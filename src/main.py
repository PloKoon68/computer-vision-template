# main.py

from config import load_configuration
from pipeline_functions.detector import YOLODetector
from pipeline_functions.tracker import SORTTracker
from pipeline_functions.preprocessor import FramePreprocessor
from pipeline_functions.visualizer import FrameVisualizer
from pipeline_functions.analytics import AnalyticsManager
from pipeline import Pipeline


# Logger ayarları...

def main():
    # 1. AYARLARI YÜKLE
    cfg = load_configuration()
    
    try:
        # 2. PARÇALARI OLUŞTUR (Dependency Injection için hazırlık)
    #    logger.info("Modüller hazırlanıyor...")
        print(f"Modüller hazırlanıyor...")
        preprocessor = FramePreprocessor(cfg.roi_percent)

        detector = YOLODetector(
            model_path=cfg.paths['model_path'],
            confidence_threshold=cfg.confidence_threshold,
            device=cfg.device
        )
        
        tracker = SORTTracker(
            max_age=cfg.max_age,
            min_hits=cfg.min_hits,
            iou_threshold=cfg.iou_threshold
        )

        analytics = AnalyticsManager(
            output_dir=cfg.paths['output_dir']
        )
        visualizer = FrameVisualizer()

        # 3. PIPELINE'I OLUŞTUR (Dependency Injection Anı!)
        # Pipeline'a "Al bunları kullan" diyoruz.
        pipeline = Pipeline(
            preprocessor=preprocessor,
            detector=detector,
            tracker=tracker,
            visualizer=visualizer,
            analytics=analytics
        )
        
        # 4. ÇALIŞTIR
        pipeline.process_video(
            input_path=cfg.paths['input_video_path'],
            output_dir=cfg.paths['output_dir'],
            frame_skip=cfg.frame_skip,  # Sadece integer değer geçiyoruz
            show_display=True
        )
 
    except Exception as e:
    #    logger.error(f"Sistem başlatılamadı: {e}")
        print(f"Sistem başlatılamadı: {e}")

if __name__ == "__main__":
    main()
