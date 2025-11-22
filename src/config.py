"""
Yapılandırma dosyası - Tüm parametreler burada
"""
import os
from dataclasses import dataclass

from torch import mode

"""
@dataclass
class Config:
    # Model ayarları
    model_path: str = "yolov8n.pt"  # YOLOv8 nano model
    confidence_threshold: float = 0.5
    iou_threshold: float = 0.45
#    device: str = "cuda"  # veya "cpu"
    device: str = "cpu" 
    
   # İşleme ayarları
    frame_width: int = 1280
    frame_height: int = 720

    # Tracker ayarları
    max_age: int = 30  # Kaybolan nesneyi kaç frame tutacak
    min_hits: int = 3   # Onaylanmış track için minimum tespit
    iou_threshold_tracker: float = 0.3
    
    # Video ayarları
    input_video: str = "input.mp4"
    output_video: str = "output.mp4"
    frame_skip: int = 1  # Her N frame'i işle (FPS artırmak için)
    target_fps: int = 30
    
 
    
    # Metrik ayarları
    log_interval: int = 30  # Her 30 frame'de bir log
    metrics_file: str = "metrics.json"
    
    # Sınıflar (COCO dataset)
    target_classes: list = None  # None = tüm sınıflar, [0, 2] = person, car    # TODO: Bu ayarı kullanıcıdan alınır
    
    def __post_init__(self):
        if self.target_classes is None:
            self.target_classes = [0]  # Sadece person (0)
        
        # Output klasörü oluştur
        os.makedirs("outputs", exist_ok=True)
        os.makedirs("logs", exist_ok=True)
   """     



from dataclasses import dataclass, field
import os
import argparse
from dotenv import load_dotenv

@dataclass
class AppConfig:
    """Uygulamanın tüm konfigürasyonlarını tutan sınıf."""
    # Model ayarları
    model_path: str = "yolov8n.pt"
    device: str = "cpu"
    
    # Tespit Parametreleri
    confidence_threshold: float = 0.5
    iou_threshold: float = 0.45
    
    # track parameters
    max_age: int = 30  # Kaybolan nesneyi kaç frame tutacak
    min_hits: int = 3   # Onaylanmış track için minimum tespit
    iou_threshold_tracker: float = 0.3
   
    # Girdi/Çıktı Ayarları
    input_video_path: str | int = 0  # 0 webcam demektir
    

    frame_skip: int = 1  # Her N frame'i işle (FPS artırmak için)
    target_fps: int = 30
    #... diğer ayarlar ...

def load_configuration() -> AppConfig:
    """
    Konfigürasyonu şu öncelik sırasıyla yükler:
    1. Komut Satırı Argümanları (en yüksek)
    2. .env Dosyası
    3. Varsayılan Değerler (en düşük)
    """
    # Adım 1: .env dosyasını yükle (eğer varsa)
    load_dotenv()

    # Adım 2: Komut satırı argümanlarını tanımla
    parser = argparse.ArgumentParser(description="Nesne Tespit ve Takip Uygulaması")
    

    # ÖNEMLİ: default=None diyerek, argümanın kullanıcı tarafından girilip girilmediğini anlarız.
    choices = {
        "device": ["cpu", "cuda"],
        "model": ["yolov8n.pt", "yolov8s.pt", "yolov8m.pt"]
    }
    parser.add_argument("--device", type=str, default=None, choices=choices["device"], help="Kullanılacak cihaz: 'cpu' veya 'cuda'")
    parser.add_argument("--model", type=str, default=None,  choices=choices["model"], help="Kullanılacak modelin yolu (örn: yolov8s.pt)")
    parser.add_argument("--input", type=str, default=None, help="İşlenecek video dosyasının ismi veya '0' (webcam)")
    parser.add_argument("--conf", type=float, default=None, help="Güven eşiği (confidence threshold)")
    
    args = parser.parse_args()

    # Adım 3: Hiyerarşiyi uygulayarak nihai konfigürasyonu oluştur
    
    # .env'den veya varsayılandan gelen değeri al, sonra komut satırı ile ez (override)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    model_name = args.model or os.getenv("MODEL_PATH", "yolov8n.pt")

    model_path = os.path.join(project_root, 'models', model_name)
    config = AppConfig(
        device=args.device or os.getenv("DEVICE", "cpu"),
        model_path=model_path,
        input_video_path=args.input or os.getenv("INPUT_VIDEO_PATH", 0),
        confidence_threshold=args.conf or float(os.getenv("CONFIDENCE_THRESHOLD", 0.5)),
    )

    # input_video_path '0' ise int'e çevir
    try:
        config.input_video_path = int(config.input_video_path)
    except (ValueError, TypeError):
        pass

    return config