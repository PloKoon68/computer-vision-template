"""
YOLO tabanlı nesne tespit modülü
"""
import numpy as np
import cv2
import torch

try:
    from ultralytics import YOLO
except ImportError:
    print("ultralytics yüklü değil: pip install ultralytics")

class Detection:
    """Tek bir tespit objesi"""
    def __init__(self, bbox: tuple[int, int, int, int], confidence: float, class_id: int):
        self.bbox = bbox  # (x1, y1, x2, y2)
        self.confidence = confidence
        self.class_id = class_id
        
    @property
    def center(self) -> tuple[int, int]:
        """Bbox merkez noktası"""
        x1, y1, x2, y2 = self.bbox
        return (int((x1 + x2) / 2), int((y1 + y2) / 2))
    
    @property
    def area(self) -> int:
        """Bbox alanı"""
        x1, y1, x2, y2 = self.bbox
        return (x2 - x1) * (y2 - y1)


class YOLODetector:
    """YOLOv8 tabanlı nesne tespit sınıfı"""
    
    def __init__(self, model_path: str, confidence_threshold: float = 0.5, 
                 device: str = None, target_classes: list[int] = None):
        """
        Args:
            model_path: YOLO model dosyası (.pt)
            confidence_threshold: Minimum güven skoru
            device: 'cuda' veya 'cpu' (None ise otomatik algılanır)
            target_classes: Tespit edilecek sınıflar [0,1,2...] veya None (hepsi)
        """
        # Cihaz otomatik algılama: CUDA varsa kullan, yoksa CPU
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Model dosyasının varlığını kontrol et (otomatik indirmeyi engellemek için)
        self.upload_model(model_path)   #self.model
        self.model.to(device)

        self.confidence_threshold = confidence_threshold
        self.device = device
        self.target_classes = target_classes
        
        
    def detect(self, frame: np.ndarray) -> list[Detection]:
        """
        Frame üzerinde nesne tespiti yapar
        
        Args:
            frame: BGR formatında görüntü
            
        Returns:
            Detection objelerinin listesi
        """
        # YOLO inference
        results = self.model.predict(
            frame, 
            conf=self.confidence_threshold,
            verbose=False,
            device=self.device
        )[0]
        
        detections = []
        
        # Her tespit için
        for box in results.boxes:
            # Koordinatlar
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            confidence = float(box.conf[0])
            class_id = int(box.cls[0])
            
            # Hedef sınıflardan değilse atla
            if self.target_classes is not None and class_id not in self.target_classes:
                continue
            
            detection = Detection(
                bbox=(int(x1), int(y1), int(x2), int(y2)),
                confidence=confidence,
                class_id=class_id
            )
            detections.append(detection)
        
        return detections
    
    def draw_detections(self, frame: np.ndarray, detections: list[Detection]) -> np.ndarray:
        """
        Tespitleri frame üzerine çizer
        
        Args:
            frame: Orijinal frame
            detections: Detection listesi
            
        Returns:
            Çizilmiş frame
        """
        frame_copy = frame.copy()
        
        for det in detections:
            x1, y1, x2, y2 = det.bbox
            
            # Bbox çiz
            cv2.rectangle(frame_copy, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Label
            label = f"ID:{det.class_id} {det.confidence:.2f}"
            cv2.putText(frame_copy, label, (x1, y1-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        return frame_copy
    
    def upload_model(self, model_path:str):
        try:
            # Şimdi modele tam ve doğru yolu veriyoruz.
            self.model = YOLO(model_path)
            print(f"✅ '{model_path}' adresindeki yerel model başarıyla yüklendi!")

        except FileNotFoundError:
            print(f"❌ HATA: Model dosyası belirtilen yolda bulunamadı: {model_path}")
            print("Lütfen 'models' klasörünün içine .pt dosyasını koyduğunuzdan emin olun.")
            exit()
        except Exception as e:
            print(f"❌ Model yüklenirken bir hata oluştu: {e}")
            exit()