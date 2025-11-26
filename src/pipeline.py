import cv2
import time
import os
import logging
from typing import Optional, Tuple
import numpy as np

# Config, Detector ve Tracker'ın import edildiğini varsayıyoruz
from config import AppConfig
from pipeline_functions import analytics

# Basit bir logger yapılandırması (Sınav için hayat kurtarır)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class Pipeline:
    """Video işleme pipeline'ı"""
    
    def __init__(self, preprocessor, detector, tracker, visualizer, analytics):
        
        logger.info(f"Pipeline başlatılıyor...")
        self.preprocessor = preprocessor
        self.detector = detector
        self.tracker = tracker   
        self.visualizer = visualizer
        self.analytics = analytics

    def process_video(self, input_path: str, output_dir: Optional[str] = None, frame_skip: int = 1, show_display: bool = False):
        
        # 1. PATH DÜZELTME (Windows için kritik)
        input_path = os.path.abspath(input_path)
        if not os.path.exists(input_path):
            raise FileNotFoundError(f"Giriş videosu bulunamadı: {input_path}")

        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            raise ValueError(f"Video açılamadı: {input_path}")
        
        # 2. VİDEO BİLGİLERİ (Integer olduğundan emin oluyoruz)
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Frame Skip Ayarı
        skip_rate = int(max(1, frame_skip)) # En az 1 olsun ve int olsun
        output_fps = fps / skip_rate
        
        # Log basalım (Boyutlar 0 gelirse hata var demektir)
        logger.info(f"Video Açıldı: {width}x{height} @ {fps:.2f}fps -> Çıktı: {output_fps:.2f}fps")
        if width == 0 or height == 0:
            raise ValueError("Video boyutları okunamadı (0x0). Video bozuk olabilir.")

        writer = None
        
        # 3. WRITER BAŞLATMA (Döngülü ve Garantili)
        if output_dir:
            # Klasör yolunu temizle ve oluştur
            output_dir = os.path.abspath(output_dir)
            os.makedirs(output_dir, exist_ok=True)

            # Denenecek codec'ler (Senin çalıştığını söylediğin liste)
            # Not: 'mp4v' Windows'ta en güvenlisidir.
            codecs = [
                ('mp4v', 'processed_video.mp4'),
                ('avc1', 'processed_video.mp4'),
                ('XVID', 'processed_video.avi'), # MP4 değil AVI deniyoruz XVID için
                ('MJPG', 'processed_video.avi')  # En son çare
            ]

            for codec_name, filename in codecs:
                fourcc = cv2.VideoWriter_fourcc(*codec_name)
                save_path = os.path.join(output_dir, filename)
                
                try:
                    temp_writer = cv2.VideoWriter(save_path, fourcc, output_fps, (width, height))
                    
                    if temp_writer.isOpened():
                        writer = temp_writer
                        logger.info(f"✅ Video Writer Başladı: {save_path} ({codec_name})")
                        break # Başarılı olduysa döngüden çık
                    else:
                        logger.warning(f"⚠️ Codec başarısız: {codec_name}")
                        
                except Exception as e:
                    logger.warning(f"⚠️ Writer hatası ({codec_name}): {e}")

            if writer is None:
                logger.error("❌ HATA: Hiçbir codec ile kayıt başlatılamadı. Çıktı klasörüne yazma izni olmayabilir.")

        # ... (Buradan sonrası aynı: frame_idx = 0, while True döngüsü...)


        frame_idx = 0
        processed_count = 0
        
        try:
            start_process_time = time.time()
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                frame_idx += 1
                if frame_idx % skip_rate != 0:
                    continue
                
                # İşlem
                t0 = time.time()
                processed_frame, dets, trks = self.process_frame(frame)
                dt = time.time() - t0
                
                processed_count += 1
                
                # Basit Loglama (Her 50 karede bir)
                if processed_count % 50 == 0:
                    logger.info(f"Frame {frame_idx}/{total_frames} | Det: {dets} | Process Time: {dt*1000:.1f}ms")

                # Görselleştirme (Headless check)
                if show_display:
                    cv2.imshow('Pipeline Stream', processed_frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        logger.info("Kullanıcı çıkışı (q)")
                        break
                
                if writer:
                    writer.write(processed_frame)
                    
        except KeyboardInterrupt:
            logger.info("İşlem manuel olarak durduruldu (Ctrl+C).")
            
        finally:
            cap.release()
            if writer: writer.release()
            cv2.destroyAllWindows()
            
            total_time = time.time() - start_process_time
            logger.info(f"🏁 İşlem Bitti. Toplam Süre: {total_time:.1f}s | Ortalama FPS: {processed_count/total_time:.2f}")
    
            self.analytics.save_report()


    def process_frame(self, frame: np.ndarray) -> Tuple[np.ndarray, int, int]:
        # Süre ölçümü başlat
        start_time = time.time()

        # 1. PREPROCESS
        proc_frame, roi_rect = self.preprocessor.process(frame)  # Bize işlenmiş küçük resim ve offset lazım
        
        # 2. DETECT
        # Dönen sonuçlar küçük resme göre (Local Coordinates)
        detections = self.detector.detect(proc_frame)
        
        # 3. TRACK
        tracks = self.tracker.update(detections)
        
        # YENİ: Analitik güncelleme (İşlem süresini hesapla)
        process_duration = time.time() - start_time
        self.analytics.update(tracks, process_duration)



        # 4. VISUALIZE
        # Çizim sınıfına "Global Frame"i, "Local Track"leri ve "Offset" bilgisini (ROI) veriyoruz.
        metrics = self.analytics.get_metrics() # {"fps": 24.5, "total": 5...}
        viz_frame = self.visualizer.draw_results(
            frame=frame, 
            tracks=tracks, 
            roi_rect=roi_rect, # İçinde (offset_x, offset_y, w, h)
            fps=metrics["fps"], # Visualizer'da bu parametreyi ekleyeceğiz
            count=metrics["total_unique_objects"] # Bunu da ekleyelim
        )

        return viz_frame, len(detections), len(tracks)