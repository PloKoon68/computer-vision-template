"""
Ana video işleme pipeline'ı
"""
import cv2
import time
import os
from typing import Optional
import numpy as np

from config import AppConfig
from pipeline_functions.detector import YOLODetector
from pipeline_functions.tracker import SORTTracker
#from utils import MetricsLogger, setup_logger


class Pipeline:
    """Video işleme pipeline'ı"""
    
    def __init__(self, config: AppConfig):
        """
        Args:
            config: Config objesi
        """
        self.config = config
        
        # Modüller
        self.detector = YOLODetector(
            model_path=config.model_path,
            confidence_threshold=config.confidence_threshold,
            device=config.device,
   #         target_classes=config.target_classes
        )
        
        self.tracker = SORTTracker(
            max_age=config.max_age,
            min_hits=config.min_hits,
            iou_threshold=config.iou_threshold_tracker
        )
        
#        self.logger = setup_logger()
 #       self.metrics = MetricsLogger(log_interval=config.log_interval)
        
 #       self.logger.info("✅ Pipeline başlatıldı")
    
    def process_video(self, input_path: str, output_path: Optional[str] = None):
        """
        Video dosyasını işle
        
        Args:
            input_path: Giriş video dosyası
            output_path: Çıkış video dosyası (None ise sadece metrik topla)
        """
        # Video aç
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            raise ValueError(f"Video açılamadı: {input_path}")
        
        # Video bilgileri
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # FPS geçersizse varsayılan değer kullan
        if fps <= 0 or fps > 120:
            fps = 30.0
            print(f"⚠️  FPS geçersiz, varsayılan {fps} kullanılıyor")
        
        # Frame skip kullanılıyorsa FPS'i ayarla
        output_fps = fps / self.config.frame_skip if self.config.frame_skip > 1 else fps
        
        print(f"📹 Video bilgisi: {width}x{height} @ {fps:.2f}fps (çıkış: {output_fps:.2f}fps), {total_frames} frame")
        
        # Video writer
        writer = None
        if output_path:
            # Windows'ta daha uyumlu codec'ler dene
            # H264 genellikle en iyi çalışır ama sistemde codec olması gerekir
            codecs_to_try = [
                ('H264', cv2.VideoWriter_fourcc(*'H264')),
                ('XVID', cv2.VideoWriter_fourcc(*'XVID')),
                ('mp4v', cv2.VideoWriter_fourcc(*'mp4v')),
            ]
            
            writer = None
            for codec_name, fourcc in codecs_to_try:
                writer = cv2.VideoWriter(output_path, fourcc, output_fps, (width, height))
                if writer.isOpened():
                    print(f"✅ Video codec: {codec_name}")
                    break
                else:
                    writer.release()
                    writer = None
            
            if writer is None or not writer.isOpened():
                raise RuntimeError(f"Video writer başlatılamadı: {output_path}. Codec sorunu olabilir.")
            
            print(f"💾 Çıkış dosyası: {output_path}")
        
        frame_idx = 0
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                frame_idx += 1
                
                # Frame skip (performans için)
                if frame_idx % self.config.frame_skip != 0:
                    continue
                
                # Frame işle
                start_time = time.time()
                processed_frame, num_detections, num_tracks = self.process_frame(frame)
                processing_time = time.time() - start_time
                
                # Metrik kaydet
#                self.metrics.log_frame(num_detections, num_tracks, processing_time)
                
                # Ekrana göster
                cv2.imshow('Processed Video', processed_frame)
                
                # Yaz
                if writer:
                    writer.write(processed_frame)
                
                # Pencereyi güncelle ve 'q' tuşu ile çıkış kontrolü
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    print("Kullanıcı tarafından durduruldu (q tuşu)")
                    break
                
                # Progress
                if frame_idx % 100 == 0:
                    progress = (frame_idx / total_frames) * 100
                    print(f"Progress: {progress:.1f}% ({frame_idx}/{total_frames})")
#                    self.logger.info(f"Progress: {progress:.1f}% ({frame_idx}/{total_frames})")
        
        finally:
            cap.release()
            if writer:
                writer.release()
                # Video dosyasının düzgün kapatıldığından emin ol
                time.sleep(0.1)  # Kısa bir bekleme
            
            # Pencereyi kapat
            cv2.destroyAllWindows()
            
            # Final rapor
#            self.metrics.print_final_report()
#            self.metrics.save_metrics()
            
            if output_path:
                if os.path.exists(output_path):
                    file_size = os.path.getsize(output_path) / (1024 * 1024)  # MB
                    print(f"✅ İşlem tamamlandı - Dosya: {output_path} ({file_size:.2f} MB)")
                else:
                    print(f"⚠️  Uyarı: Çıkış dosyası oluşturulamadı: {output_path}")
            else:
                print("✅ İşlem tamamlandı")
#            self.logger.info("✅ İşlem tamamlandı")
    
    def process_frame(self, frame: np.ndarray):
        """
        Tek bir frame işle
        
        Args:
            frame: BGR formatında görüntü
            
        Returns:
            (processed_frame, num_detections, num_tracks)
        """
        # 1. Tespit
        detections = self.detector.detect(frame)
        
        # 2. İzleme
        tracks = self.tracker.update(detections)
        
        # 3. Görselleştirme
        processed_frame = self.tracker.draw_tracks(frame, tracks)
        
        # Bilgi yazısı
        info_text = f"Detections: {len(detections)} | Tracks: {len(tracks)}"
        cv2.putText(processed_frame, info_text, (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        return processed_frame, len(detections), len(tracks)
    
    def process_webcam(self, camera_id: int = 0):
        """
        Webcam'den gerçek zamanlı işleme
        
        Args:
            camera_id: Kamera ID'si (varsayılan 0)
        """
        cap = cv2.VideoCapture(camera_id)
        if not cap.isOpened():
            raise ValueError(f"Kamera açılamadı: {camera_id}")
        
#        self.logger.info("🎥 Webcam başlatıldı (Çıkmak için 'q')")
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Frame işle
                start_time = time.time()
                processed_frame, num_detections, num_tracks = self.process_frame(frame)
                processing_time = time.time() - start_time
                
                # FPS göster
                fps = 1.0 / processing_time if processing_time > 0 else 0
                fps_text = f"FPS: {fps:.1f}"
                cv2.putText(processed_frame, fps_text, (10, 70), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                
                # Göster
                cv2.imshow('Detection & Tracking', processed_frame)
                
                # Çıkış
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        
        finally:
            cap.release()
            cv2.destroyAllWindows()
#            self.logger.info("✅ Webcam kapatıldı")


def main():
    """Test için ana fonksiyon"""
    # Config oluştur
    config = AppConfig()
    
    # Pipeline oluştur
    pipeline = Pipeline(config)
    
    # Video işle (veya webcam)
    # pipeline.process_video("input.mp4", "outputs/output.mp4")
    pipeline.process_webcam(0)


if __name__ == "__main__":
    main()