"""
Nesne izleme modülü - Basitleştirilmiş SORT algoritması
(DeepSORT yerine daha basit SORT kullanıyoruz - sınav için yeterli)
"""
import numpy as np
from scipy.optimize import linear_sum_assignment
import cv2

class Track:
    """Tek bir izleme objesi"""
    
    _id_counter = 0  # Benzersiz ID atamak için
    
    def __init__(self, detection):
        self.id = Track._id_counter
        Track._id_counter += 1
        
        self.bbox = detection.bbox
        self.confidence = detection.confidence
        self.class_id = detection.class_id
        
        self.age = 0  # Kaç frame'dir hayatta
        self.hits = 1  # Toplam eşleşme sayısı
        self.time_since_update = 0  # Son güncellemeye kadar geçen frame
        
        # Kalman filtresi için basit tahmin (gerçek projede Kalman kullanılır)
        self.velocity = [0, 0]
        
    def update(self, detection):
        """Yeni tespit ile track'i güncelle"""
        # Hız tahmini
        old_center = self._get_center(self.bbox)
        new_center = self._get_center(detection.bbox)
        self.velocity = [new_center[0] - old_center[0], new_center[1] - old_center[1]]
        
        # Güncelle
        self.bbox = detection.bbox
        self.confidence = detection.confidence
        self.hits += 1
        self.time_since_update = 0
        
    def predict(self):
        """Bir sonraki konumu tahmin et"""
        x1, y1, x2, y2 = self.bbox
        x1 += self.velocity[0]
        x2 += self.velocity[0]
        y1 += self.velocity[1]
        y2 += self.velocity[1]
        self.bbox = (int(x1), int(y1), int(x2), int(y2))
        
        self.age += 1
        self.time_since_update += 1
        
    def _get_center(self, bbox):
        x1, y1, x2, y2 = bbox
        return [(x1 + x2) / 2, (y1 + y2) / 2]
    
    @property
    def is_confirmed(self):
        """Track onaylandı mı? (minimum hit sayısına ulaştı mı)"""
        return self.hits >= 3


class SORTTracker:
    """Basitleştirilmiş SORT tracker (sınav için yeterli)"""
    
    def __init__(self, max_age: int = 30, min_hits: int = 3, iou_threshold: float = 0.3):
        """
        Args:
            max_age: Kaybolan track kaç frame tutulur
            min_hits: Onaylanmış track için minimum eşleşme
            iou_threshold: Eşleştirme için minimum IoU
        """
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        
        self.tracks: list[Track] = []
        
    def update(self, detections: list) -> list[Track]:
        """
        Yeni tespitler ile track'leri güncelle
        
        Args:
            detections: Detection objelerinin listesi
            
        Returns:
            Aktif Track listesi
        """
        # Önce tüm track'leri tahmin et
        for track in self.tracks:
            track.predict()
        
        # Eğer tespit yoksa, sadece track'leri döndür
        if len(detections) == 0:
            self._remove_dead_tracks()
            return [t for t in self.tracks if t.is_confirmed]
        
        # Eşleştirme yap
        matched, unmatched_dets, unmatched_trks = self._match(detections)
        
        # Eşleşenleri güncelle
        for det_idx, trk_idx in matched:
            self.tracks[trk_idx].update(detections[det_idx])
        
        # Eşleşmeyen tespitler için yeni track oluştur
        for det_idx in unmatched_dets:
            self.tracks.append(Track(detections[det_idx]))
        
        # Ölü track'leri kaldır
        self._remove_dead_tracks()
        
        # Onaylanmış track'leri döndür
        return [t for t in self.tracks if t.is_confirmed]
    
    def _match(self, detections: list) -> tuple[list, list, list]:
        """Tespitler ile track'leri eşleştir"""
        if len(self.tracks) == 0:
            return [], list(range(len(detections))), []
        
        # IoU matrisi oluştur
        iou_matrix = np.zeros((len(detections), len(self.tracks)))
        for d, det in enumerate(detections):
            for t, trk in enumerate(self.tracks):
                iou_matrix[d, t] = self._iou(det.bbox, trk.bbox)
        
        # Hungarian algoritması ile eşleştir
        matched_indices = linear_sum_assignment(-iou_matrix)
        matched_indices = np.array(list(zip(*matched_indices)))
        
        # Düşük IoU'lu eşleşmeleri filtrele
        matches = []
        unmatched_detections = []
        unmatched_tracks = []
        
        for d in range(len(detections)):
            if d not in matched_indices[:, 0]:
                unmatched_detections.append(d)
        
        for t in range(len(self.tracks)):
            if t not in matched_indices[:, 1]:
                unmatched_tracks.append(t)
        
        for d, t in matched_indices:
            if iou_matrix[d, t] < self.iou_threshold:
                unmatched_detections.append(d)
                unmatched_tracks.append(t)
            else:
                matches.append((d, t))
        
        return matches, unmatched_detections, unmatched_tracks
    
    def _iou(self, bbox1: tuple, bbox2: tuple) -> float:
        """İki bbox arasındaki IoU hesapla"""
        x1_1, y1_1, x2_1, y2_1 = bbox1
        x1_2, y1_2, x2_2, y2_2 = bbox2
        
        # Kesişim
        xi1 = max(x1_1, x1_2)
        yi1 = max(y1_1, y1_2)
        xi2 = min(x2_1, x2_2)
        yi2 = min(y2_1, y2_2)
        inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)
        
        # Birleşim
        box1_area = (x2_1 - x1_1) * (y2_1 - y1_1)
        box2_area = (x2_2 - x1_2) * (y2_2 - y1_2)
        union_area = box1_area + box2_area - inter_area
        
        return inter_area / union_area if union_area > 0 else 0
    
    def _remove_dead_tracks(self):
        """Çok uzun süredir güncellenmeyen track'leri sil"""
        self.tracks = [t for t in self.tracks if t.time_since_update < self.max_age]
    
    def draw_tracks(self, frame: np.ndarray, tracks: list[Track]) -> np.ndarray:
        """Track'leri frame üzerine çiz"""
        frame_copy = frame.copy()
        
        for track in tracks:
            x1, y1, x2, y2 = track.bbox
            
            # Renk (ID'ye göre)
            color = self._get_color(track.id)
            
            # Bbox çiz
            cv2.rectangle(frame_copy, (x1, y1), (x2, y2), color, 2)
            
            # ID ve güven skoru
            label = f"ID:{track.id} ({track.confidence:.2f})"
            cv2.putText(frame_copy, label, (x1, y1-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            
            # Yaş bilgisi (debug için)
            age_label = f"Age:{track.age}"
            cv2.putText(frame_copy, age_label, (x1, y2+20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
        
        return frame_copy
    
    def _get_color(self, track_id: int) -> tuple[int, int, int]:
        """ID'ye göre benzersiz renk üret"""
        np.random.seed(track_id)
        return tuple(np.random.randint(0, 255, 3).tolist())