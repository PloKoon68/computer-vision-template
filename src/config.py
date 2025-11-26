from dataclasses import dataclass, field
import os
import argparse
from typing import Optional, Tuple, Dict
from dotenv import load_dotenv
from datetime import datetime

@dataclass
class AppConfig:
    """Uygulamanın tüm konfigürasyonlarını tutan sınıf."""
    
    # --- 1. ÇÖZÜM: Varsayılan değeri olmayanları EN BAŞA alabilirsin ---
    # Ama daha kolayı, hepsine varsayılan değer vermektir.
    
    # Model ayarları
    device: str = "cpu"
    
    # Tespit Parametreleri
    confidence_threshold: float = 0.5
    iou_threshold: float = 0.45
    
    # Track parameters
    max_age: int = 30
    min_hits: int = 3
    iou_threshold_tracker: float = 0.3
    
    # --- HATA BURADAYDI DÜZELTİLDİ ---
    # paths: dict  <-- Hatalıydı çünkü default değeri yoktu
    paths: Dict[str, str] = field(default_factory=dict) # Doğrusu bu
    
    frame_skip: int = 1
    target_fps: int = 30

    roi_percent: Optional[Tuple[float, float, float, float]] = None

def load_configuration() -> AppConfig:
    load_dotenv()

    parser = argparse.ArgumentParser(description="Nesne Tespit ve Takip Uygulaması")
    
    choices = {
        "device": ["cpu", "cuda"],
        "model": ["yolov8n.pt", "yolov8s.pt", "yolov8m.pt"]
    }
    
    parser.add_argument("--device", type=str, default=None, choices=choices["device"])
    parser.add_argument("--model", type=str, default=None, choices=choices["model"])
    parser.add_argument("--input", type=str, default=None)
    parser.add_argument("--input_path", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--conf", type=float, default=None)
    
    parser.add_argument("--roi", type=float, nargs=4, default=None, 
                        metavar=('X', 'Y', 'W', 'H'),
                        help="ROI alanı yüzdeleri. Örn: --roi 0.1 0.1 0.5 0.5")

    args = parser.parse_args()

    # Yolları hesapla
    paths_dict = get_paths(args)
    
    # ROI parse et
    roi_values = get_roi_values(args)

    # --- Config Oluşturma ---
    # BURAYA DİKKAT: Artık model_path vb. tek tek vermiyoruz, paths sözlüğünü veriyoruz.
    config = AppConfig(
        device=args.device or os.getenv("DEVICE", "cpu"),
        paths=paths_dict, # Sözlüğü buraya veriyoruz
        confidence_threshold=args.conf or float(os.getenv("CONFIDENCE_THRESHOLD", 0.5)),
        roi_percent=roi_values
    )
    
    # Input video path '0' kontrolü (Webcam için)
    # Artık paths sözlüğünün içinden kontrol etmeliyiz
    try:
        # Eğer '0' ise integer'a çevir, değilse string kalsın
        if config.paths["input_video_path"] == '0':
             config.paths["input_video_path"] = 0
        else:
             # Sadece sayıysa int yap (bazen string '0' gelebilir)
             try:
                 config.paths["input_video_path"] = int(config.paths["input_video_path"])
             except ValueError:
                 pass
    except (KeyError, TypeError):
        pass

    return config

def get_paths(args) -> Dict[str, str]:
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    models_path = os.path.join(project_root, 'models')
    model_name = args.model or os.getenv("MODEL_NAME", "yolov8n.pt")
    model_path = os.path.join(models_path, model_name)

    input_video_dir = args.input_path or os.getenv("INPUT_VIDEO_PATH", "data/input/videos")
    # Eğer input argümanı '0' gelirse dosya yolu oluşturmaya çalışma

    input_video_name = args.input or os.getenv("INPUT_VIDEO_NAME", "")
    input_video_path = os.path.join(project_root, input_video_dir, input_video_name+'.mp4')

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_relative_dir = args.output_dir or os.getenv("OUTPUT_VIDEO_RELATIVE_DIR", f"data/output/videos/{input_video_name}/{timestamp}")
    output_dir = os.path.join(project_root, output_relative_dir)
    #print("her: ", os.listdir(output_dir))
    return {
        "model_path": model_path, 
        "input_video_path": input_video_path, 
        "output_dir": output_dir
    }

def get_roi_values(args) -> Optional[Tuple[float, float, float, float]]:
    if args.roi is not None:
        return tuple(args.roi)
    
    env_roi = os.getenv("ROI_PERCENT")
    if env_roi:
        try:
            values = [float(x.strip()) for x in env_roi.split(',')]
            if len(values) == 4:
                return tuple(values)
        except ValueError:
            pass
    return None