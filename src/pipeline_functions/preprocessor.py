import cv2
import numpy as np
from typing import Tuple, Optional

class FramePreprocessor:
    """
    Görüntü ön işleme işlemlerini yöneten sınıf.
    SRP: Sadece görüntü manipülasyonundan sorumlu.
    """
    def __init__(self, 
                 roi_area: Optional[Tuple[int, int, int, int]] = None, 
                 use_clahe: bool = False, 
                 gamma_value: float = 1.0):
        """
        Args:
            roi_area: (x, y, w, h) formatında ilgi alanı. None ise tüm frame.
            use_clahe: Kontrast eşitleme açılsın mı?
            gamma_value: Parlaklık ayarı (1.0 = normal, <1 koyu, >1 parlak)
        """
        self.roi_area = roi_area
        self.use_clahe = use_clahe
        self.gamma_value = gamma_value
        
        # CLAHE objesini BİR KERE oluşturuyoruz (Loop içinde değil!)
        if self.use_clahe:
            self.clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))

    def process(self, frame: np.ndarray) -> Tuple[np.ndarray, int, int]:
        """
        Frame'i işle ve ROI offset değerlerini döndür.
        
        Returns:
            processed_frame: İşlenmiş (veya kırpılmış) görüntü
            offset_x: ROI x başlangıcı (Koordinat düzeltmesi için)
            offset_y: ROI y başlangıcı
        """
        # 1. ROI İşlemi (En başta yapılır ki diğer işlemler küçük resme uygulansın -> HIZ)
        offset_x, offset_y = 0, 0
        
        processed_frame = frame
        
        if self.roi_area:
            x, y, w, h = self.roi_area
            # Görüntü sınırlarını kontrol et
            img_h, img_w = frame.shape[:2]
            # Güvenli crop (resim dışına taşmayı önle)
            x = max(0, min(x, img_w))
            y = max(0, min(y, img_h))
            # Crop yap
            processed_frame = frame[y:y+h, x:x+w]
            offset_x, offset_y = x, y

        # 2. Gamma Correction (Eğer default 1.0 değilse)
        if self.gamma_value != 1.0:
            # Lookup table yöntemi (Hızlıdır)
            inv_gamma = 1.0 / self.gamma_value
            table = np.array([((i / 255.0) ** inv_gamma) * 255
                              for i in np.arange(0, 256)]).astype("uint8")
            processed_frame = cv2.LUT(processed_frame, table)

        # 3. CLAHE (Kontrast)
        if self.use_clahe:
            # CLAHE renkli resme direkt uygulanmaz, LAB uzayına geçilir
            lab = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            l_eq = self.clahe.apply(l)
            processed_frame = cv2.cvtColor(cv2.merge((l_eq, a, b)), cv2.COLOR_LAB2BGR)

        return processed_frame, offset_x, offset_y