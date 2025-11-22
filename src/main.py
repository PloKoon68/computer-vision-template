# main.py

from config import load_configuration
from pipeline_functions.detector import YOLODetector
from pipeline_functions.tracker import SORTTracker
from pipeline import Pipeline

# Logger ayarları...

def main():
    # 1. AYARLARI YÜKLE
    cfg = load_configuration()
    
    try:
        # 2. PARÇALARI OLUŞTUR (Dependency Injection için hazırlık)
    #    logger.info("Modüller hazırlanıyor...")
        print(f"Modüller hazırlanıyor...", cfg.model_path)
        
        my_detector = YOLODetector(
            model_path=cfg.model_path,
            confidence_threshold=cfg.confidence_threshold,
            device=cfg.device
        )
        
        my_tracker = SORTTracker(
            max_age=cfg.max_age,
            iou_threshold=cfg.iou_threshold
        )
        
        # 3. PIPELINE'I OLUŞTUR (Dependency Injection Anı!)
        # Pipeline'a "Al bunları kullan" diyoruz.
        pipeline = Pipeline(
            detector=my_detector,
            tracker=my_tracker
        )
        
        # 4. ÇALIŞTIR
        pipeline.process_video(
            input_path=cfg.input_video_path,
            output_path=cfg.output_video_path,
            frame_skip=cfg.frame_skip,  # Sadece integer değer geçiyoruz
            show_display=True
        )
 
    except Exception as e:
    #    logger.error(f"Sistem başlatılamadı: {e}")
        print(f"Sistem başlatılamadı: {e}")

if __name__ == "__main__":
    main()



"""
import os
from config import load_configuration
from pipeline import Pipeline

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
input_video = os.path.join(project_root, 'data', 'input', 'videos', 'bottle_track.mp4')
output_video = os.path.join(project_root, 'data', 'output', 'bottle_track.mp4')

config = load_configuration()
pipeline = Pipeline(config)

pipeline.process_video(input_video, output_video, True)
"""