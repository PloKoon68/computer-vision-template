"""
Yardımcı fonksiyonlar ve sınıflar
"""
import logging
import json
import os
from datetime import datetime


def setup_logger(name: str = "VideoPipeline", log_file: str = None) -> logging.Logger:
    """
    Logger oluşturur
    
    Args:
        name: Logger adı
        log_file: Log dosyası yolu (None ise sadece console)
        
    Returns:
        Logger objesi
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    
    # Eğer handler'lar zaten eklenmişse, tekrar ekleme
    if logger.handlers:
        return logger
    
    # Format
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler (varsa)
    if log_file:
        os.makedirs(os.path.dirname(log_file) if os.path.dirname(log_file) else '.', exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger


class MetricsLogger:
    """Frame bazlı metrikleri toplar ve kaydeder"""
    
    def __init__(self, log_interval: int = 30):
        """
        Args:
            log_interval: Her N frame'de bir log yaz
        """
        self.log_interval = log_interval
        self.frame_count = 0
        self.total_detections = 0
        self.total_tracks = 0
        self.total_processing_time = 0.0
        
        self.frame_metrics: list[dict] = []
        
    def log_frame(self, num_detections: int, num_tracks: int, processing_time: float):
        """
        Bir frame'in metriklerini kaydet
        
        Args:
            num_detections: Tespit sayısı
            num_tracks: Track sayısı
            processing_time: İşleme süresi (saniye)
        """
        self.frame_count += 1
        self.total_detections += num_detections
        self.total_tracks += num_tracks
        self.total_processing_time += processing_time
        
        # Frame metriklerini kaydet
        self.frame_metrics.append({
            'frame': self.frame_count,
            'detections': num_detections,
            'tracks': num_tracks,
            'processing_time': processing_time
        })
        
    def print_final_report(self):
        """Final metrik raporunu yazdır"""
        if self.frame_count == 0:
            print("⚠️  Hiç frame işlenmedi")
            return
        
        avg_detections = self.total_detections / self.frame_count
        avg_tracks = self.total_tracks / self.frame_count
        avg_processing_time = self.total_processing_time / self.frame_count
        avg_fps = 1.0 / avg_processing_time if avg_processing_time > 0 else 0
        
        print("\n" + "="*50)
        print("📊 FINAL METRİK RAPORU")
        print("="*50)
        print(f"Toplam Frame: {self.frame_count}")
        print(f"Ortalama Tespit/Frame: {avg_detections:.2f}")
        print(f"Ortalama Track/Frame: {avg_tracks:.2f}")
        print(f"Ortalama İşleme Süresi: {avg_processing_time*1000:.2f} ms")
        print(f"Ortalama FPS: {avg_fps:.2f}")
        print("="*50 + "\n")
    
    def save_metrics(self, filepath: str = "logs/metrics.json"):
        """
        Metrikleri JSON dosyasına kaydet
        
        Args:
            filepath: Kayıt dosyası yolu
        """
        os.makedirs(os.path.dirname(filepath) if os.path.dirname(filepath) else '.', exist_ok=True)
        
        metrics_data = {
            'timestamp': datetime.now().isoformat(),
            'summary': {
                'total_frames': self.frame_count,
                'total_detections': self.total_detections,
                'total_tracks': self.total_tracks,
                'total_processing_time': self.total_processing_time,
                'avg_detections_per_frame': self.total_detections / self.frame_count if self.frame_count > 0 else 0,
                'avg_tracks_per_frame': self.total_tracks / self.frame_count if self.frame_count > 0 else 0,
                'avg_processing_time': self.total_processing_time / self.frame_count if self.frame_count > 0 else 0,
            },
            'frame_metrics': self.frame_metrics
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(metrics_data, f, indent=2, ensure_ascii=False)
        
        print(f"✅ Metrikler kaydedildi: {filepath}")

