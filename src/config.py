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
    output_video_path: str = ""
    

    frame_skip: int = 1  # Her N frame'i işle (FPS artırmak için)
    target_fps: int = 30

    roi_percent: tuple[float, float, float, float] = None # (20, 30, 40, 70)

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
    parser.add_argument("--input_path", type=str, default=None, help="İşlenecek video dosyasının yolu")
    parser.add_argument("--output", type=str, default=None, help="İşlenmiş videonun ismi")
    parser.add_argument("--output_path", type=str, default=None, help="İşlenmiş videonun kaydedildiği dosyasının yolu")
    parser.add_argument("--conf", type=float, default=None, help="Güven eşiği (confidence threshold)")

    # nargs=4: Kullanıcının peş peşe 4 değer girmesini zorunlu kılar.
    # type=float: Hepsinin sayı olmasını sağlar.    
    parser.add_argument("--roi", type=float, nargs=4, default=None, 
                        metavar=('X', 'Y', 'W', 'H'),
                        help="ROI alanı yüzdeleri (0.0-1.0 arası) px, py, pw, ph. Örn: --roi 0.1 0.1 0.5 0.5")

    args = parser.parse_args()

    # Adım 3: Hiyerarşiyi uygulayarak nihai konfigürasyonu oluştur
    
    # .env'den veya varsayılandan gelen değeri al, sonra komut satırı ile ez (override)
    model_path, input_video_path, output_video_path = get_paths(args)

    # ROI'yi parse et (Hem CLI hem .env desteği için helper fonksiyon)
    roi_values = get_roi_values(args)

    config = AppConfig(
        device=args.device or os.getenv("DEVICE", "cpu"),
        model_path=model_path,
        input_video_path=input_video_path,
        output_video_path=output_video_path,
        confidence_threshold=args.conf or float(os.getenv("CONFIDENCE_THRESHOLD", 0.5)),
        roi_percent=roi_values  # Tuple olarak atandı
    )

    # input_video_path '0' ise int'e çevir
    try:
        config.input_video_path = int(config.input_video_path)
    except (ValueError, TypeError):
        pass

    return config

def get_paths(args):
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    models_path = os.path.join(project_root, 'models')
    model_name = args.model or os.getenv("MODEL_NAME", "yolov8n.pt")
    model_path = os.path.join(models_path, model_name)

    input_video_path_from_root = args.input_path or os.getenv("INPUT_VIDEO_PATH", "data/input/videos")
    input_video_name = args.input or os.getenv("INPUT_VIDEO_NAME", "production_line")
    input_video_path = os.path.join(project_root, input_video_path_from_root, input_video_name+'.mp4')

    output_video_path_from_root = args.output_path or os.getenv("OUTPUT_VIDEO_PATH", "data/output/videos")
    output_video_path_name = args.output or os.getenv("OUTPUT_VIDEO_NAME", f"{input_video_name}_detected.mp4")
    output_video_path = os.path.join(project_root, output_video_path_from_root, output_video_path_name)

    return model_path, input_video_path, output_video_path


def get_roi_values(args) -> tuple[float, float, float, float] | None:
    """
    ROI değerlerini önce argümanlardan, yoksa .env dosyasından alır.
    """
    # 1. Öncelik: Komut Satırı (CLI)
    if args.roi is not None:
        # args.roi zaten [0.1, 0.1, 0.5, 0.5] şeklinde bir liste gelir (nargs=4 sayesinde)
        return tuple(args.roi)
    
    # 2. Öncelik: .env dosyası
    env_roi = os.getenv("ROI_PERCENT") # Örn: "0.2,0.2,0.6,0.6" string olarak gelir
    if env_roi:
        try:
            # String'i virgüllerden ayırıp float'a çeviriyoruz
            values = [float(x.strip()) for x in env_roi.split(',')]
            if len(values) == 4:
                return tuple(values)
            else:
                print("UYARI: .env dosyasındaki ROI formatı hatalı. 4 değer gerekli. ROI devre dışı.")
        except ValueError:
            print("UYARI: .env dosyasındaki ROI sayısal değil. ROI devre dışı.")
            
    return None