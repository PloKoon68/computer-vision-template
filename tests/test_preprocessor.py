import pytest
import numpy as np
from src.pipeline_functions.preprocessor import FramePreprocessor

def test_roi_calculation():
    """Yüzdelik ROI, doğru piksel koordinatlarına dönüşüyor mu?"""
    # 100x100'lük siyah bir resim (Dummy Image)
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    
    # %50 merkez ROI: 
    # x=25 (%25), y=25 (%25), w=50 (%50), h=50 (%50)
    roi_percent = (25, 25, 50, 50)
    
    prep = FramePreprocessor(roi_percent=roi_percent)
    processed_frame, roi_rect = prep.process(frame)
    
    # 1. ROI Rect (x, y, w, h) doğru hesaplandı mı?
    assert roi_rect == (25, 25, 50, 50)
    
    # 2. Resim boyutu gerçekten küçüldü mü?
    assert processed_frame.shape == (50, 50, 3)

def test_no_roi():
    """ROI verilmezse orijinal resim dönüyor mu?"""
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    prep = FramePreprocessor(roi_percent=None)
    processed_frame, roi_rect = prep.process(frame)
    
    assert roi_rect is None
    assert processed_frame.shape == (100, 100, 3)