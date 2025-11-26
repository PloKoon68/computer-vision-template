import pytest
import numpy as np
from unittest.mock import MagicMock
from src.pipeline import Pipeline

# Tracker'ın beklediği Detection sınıfını taklit eden basit bir sınıf
class MockDetection:
    def __init__(self, bbox):
        self.bbox = bbox # [x1, y1, x2, y2]
        self.confidence = 0.9
        self.class_id = 0

def test_pipeline_flow():
    """Pipeline baştan sona hatasız akıyor mu?"""
    
    # --- 1. MOCK (SAHTE) PARÇALAR ---
    
    # Detector
    mock_detector = MagicMock()
    mock_detector.detect.return_value = [MockDetection([10, 10, 50, 50])]
    
    # Tracker
    mock_tracker = MagicMock()
    mock_tracker.update.return_value = []
    
    # Preprocessor
    mock_preprocessor = MagicMock()
    # (işlenmiş_resim, roi_rect) döner
    mock_preprocessor.process.return_value = (np.zeros((100,100,3)), (0,0,100,100))
    
    # Visualizer (Çizim yapmasın, sadece frame'i geri döndürsün)
    mock_visualizer = MagicMock()
    # draw_results çağrılınca ilk argümanı (frame) geri dönsün
    mock_visualizer.draw_results.side_effect = lambda frame, **kwargs: frame
    
    # Analytics (İstatistik tutmasın)
    mock_analytics = MagicMock()
    
    # --- 2. PIPELINE OLUŞTUR ---
    # Dependency Injection: Tüm parçaları dışarıdan veriyoruz
    pipeline = Pipeline(
        detector=mock_detector, 
        tracker=mock_tracker, 
        preprocessor=mock_preprocessor,
        visualizer=mock_visualizer,
        analytics=mock_analytics
    )
    
    # --- 3. ÇALIŞTIR ---
    fake_frame = np.zeros((100, 100, 3), dtype=np.uint8)
    
    # Hata vermeden çalışmalı
    viz_frame, n_det, n_trk = pipeline.process_frame(fake_frame)
    
    # --- 4. DOĞRULA (ASSERT) ---
    mock_detector.detect.assert_called_once()
    mock_tracker.update.assert_called_once()
    
    # Analytics update edildi mi?
    # (Mock obje olduğu için gerçek bir update yapmaz ama çağrıldığını görürüz)
    # pipeline.analytics.update.assert_called_once() 
    
    assert n_det == 1
    assert isinstance(viz_frame, np.ndarray)