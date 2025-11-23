# pipeline_functions/preprocessor.py

import cv2
import numpy as np
from typing import Optional

class FramePreprocessor:
    def __init__(self,                 
                 roi_percent: Optional[tuple[float, float, float, float]] = None,  #px, py, pw, ph
                 use_clahe: bool = False):
        self.roi_percent = roi_percent
        self.use_clahe = use_clahe
        
        if self.use_clahe:
            self.clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))

    def process(self, frame: np.ndarray) -> tuple[np.ndarray, Optional[tuple]]:
        """
        Returns:
            processed_frame: İşlenmiş görüntü
            roi_rect: (x, y, w, h) pixel değerleri veya None
        """
        h, w = frame.shape[:2]
        processed_frame = frame
        roi_rect = None # Varsayılan (ROI yoksa)

        # 1. ROI Hesaplama
        if self.roi_percent:
            px, py, pw, ph = self.roi_percent
            x = int(px * w / 100)
            y = int(py * h / 100)
            width = int(pw * w / 100) 
            height = int(ph * h / 100)
            
            # Sınır kontrolleri
            x = max(0, min(x, w))
            y = max(0, min(y, h))
            
            # ROI Rect oluştur (Dönüş değeri için)
            roi_rect = (x, y, width, height)
            
            # Kırpma
            processed_frame = frame[y:y+height, x:x+width]

        # 2. CLAHE
        if self.use_clahe:
            lab = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            l_eq = self.clahe.apply(l)
            processed_frame = cv2.cvtColor(cv2.merge((l_eq, a, b)), cv2.COLOR_LAB2BGR)

        return processed_frame, roi_rect