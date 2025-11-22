import cv2
import time
import os
import logging
from typing import Optional, Tuple
import numpy as np

# Config, Detector ve Tracker'ın import edildiğini varsayıyoruz
from config import AppConfig
from pipeline_functions.detector import YOLODetector
from pipeline_functions.tracker import SORTTracker

# Basit bir logger yapılandırması (Sınav için hayat kurtarır)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class Pipeline:
    """Video işleme pipeline'ı"""
    
    def __init__(self, detector, tracker):
        
        logger.info(f"Pipeline başlatılıyor...")
        self.detector = detector
        self.tracker = tracker   

    def process_video(self, input_path: str, output_path: Optional[str] = None, frame_skip: int = 1, show_display: bool = False):
        """
        Video dosyasını işle.
        show_display: False yapılırsa sunucu modunda (GUI olmadan) çalışır.
        """
        if not os.path.exists(input_path):
            raise FileNotFoundError(f"Giriş videosu bulunamadı: {input_path}")

        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            raise ValueError(f"Video açılamadı: {input_path}")
        
        # Video bilgileri
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Güvenli Frame Skip
        skip_rate = frame_skip # Config'de yoksa 1 al
        if skip_rate < 1: skip_rate = 1

        output_fps = fps / skip_rate
        
        logger.info(f"Video: {width}x{height} @ {fps:.2f}fps -> İşlenen: {output_fps:.2f}fps")
        
        writer = None
        if output_path:
            # 1. Klasör Kontrolü (KRİTİK)
            output_dir = os.path.dirname(output_path)
            if output_dir and not os.path.exists(output_dir):
                os.makedirs(output_dir)
                logger.info(f"Klasör oluşturuldu: {output_dir}")

            # 2. Codec Denemeleri
            codecs_to_try = [
                ('mp4v', cv2.VideoWriter_fourcc(*'mp4v')),
                ('avc1', cv2.VideoWriter_fourcc(*'avc1')),
                ('XVID', cv2.VideoWriter_fourcc(*'XVID')),
            ]
            
            for codec_name, fourcc in codecs_to_try:
                temp_writer = cv2.VideoWriter(output_path, fourcc, output_fps, (width, height))
                if temp_writer.isOpened():
                    writer = temp_writer
                    logger.info(f"✅ Video codec seçildi: {codec_name}")
                    break
            
            if not writer:
                logger.warning("⚠️ Video writer başlatılamadı, kayıt yapılmayacak!")

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

    def process_frame(self, frame: np.ndarray) -> Tuple[np.ndarray, int, int]:
        """Tek kare işleme mantığı"""
        # Deteksiyon
        detections = self.detector.detect(frame)
        # Takip
        tracks = self.tracker.update(detections)
        # Çizim
        viz_frame = self.tracker.draw_tracks(frame, tracks)
        
        return viz_frame, len(detections), len(tracks)

# Main kısmı aynı kalabilir...