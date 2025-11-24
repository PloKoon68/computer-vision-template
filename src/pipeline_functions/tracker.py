"""
SORT (Simple Online and Realtime Tracking) Implementasyonu
OpenCV Kalman Filtresi kullanılarak "Uçan Kutu" sorunu çözülmüştür.
"""
import numpy as np
from scipy.optimize import linear_sum_assignment
import cv2

class KalmanBoxTracker:
    """
    Tek bir nesneyi izleyen, Kalman Filtresi tabanlı sınıf.
    """
    count = 0
    
    def __init__(self, bbox):
        """
        Args:
            bbox: [x1, y1, x2, y2]
        """
        # Kalman Filtresi Kurulumu (7 State, 4 Measurement)
        self.kf = cv2.KalmanFilter(7, 4) 
        
        # F (Transition Matrix)
        self.kf.transitionMatrix = np.array([
            [1,0,0,0,1,0,0],
            [0,1,0,0,0,1,0],
            [0,0,1,0,0,0,1],
            [0,0,0,1,0,0,0],  
            [0,0,0,0,1,0,0],
            [0,0,0,0,0,1,0],
            [0,0,0,0,0,0,1]
        ], np.float32)

        # H (Measurement Matrix)
        self.kf.measurementMatrix = np.array([
            [1,0,0,0,0,0,0],
            [0,1,0,0,0,0,0],
            [0,0,1,0,0,0,0],
            [0,0,0,1,0,0,0]
        ], np.float32)

        # Gürültü Kovaryansları
        self.kf.processNoiseCov = np.eye(7, dtype=np.float32) * 0.03
        self.kf.measurementNoiseCov = np.eye(4, dtype=np.float32) * 0.5
        self.kf.errorCovPost = np.eye(7, dtype=np.float32) 

        # --- DÜZELTME BURADA ---
        # 1. Önce 7 satırlık boş bir hafıza oluşturuyoruz (rastgele sayılar olmasın diye)
        # OpenCV statePost'u varsayılan olarak 0 başlatır ama biz garantiye alalım.
        self.kf.statePost = np.random.randn(7, 1).astype(np.float32) 
        
        # 2. İlk 4 satıra (Konum bilgilerine) bizim ölçümü yazıyoruz
        # Geriye kalan 3 satır (Hızlar) rastgele veya 0 kalıyor, zamanla düzelecek.
        self.kf.statePost[:4] = self.convert_bbox_to_z(bbox)
        # -----------------------
        
        self.time_since_update = 0
        self.id = KalmanBoxTracker.count
        KalmanBoxTracker.count += 1
        self.history = []
        self.hits = 0
        self.hit_streak = 0
        self.age = 0
        
        self.bbox = bbox 
        self.confidence = 0.0 
        self.class_id = -1

    def update(self, bbox, confidence=0.0, class_id=-1):
        """
        Yeni bir ölçümle (YOLO tespitiyle) filtreyi güncelle.
        """
        self.time_since_update = 0
        self.history = []
        self.hits += 1
        self.hit_streak += 1
        
        # Kalman Düzeltmesi (Correction)
        self.kf.correct(self.convert_bbox_to_z(bbox))
        
        # Bilgileri sakla
        self.bbox = bbox
        self.confidence = confidence
        self.class_id = class_id

    def predict(self):
        """
        Bir sonraki adımı tahmin et.
        """
        # Eğer alan (s) çok küçükse düzelt
        if((self.kf.statePost[6]+self.kf.statePost[2])<=0):
            self.kf.statePost[6] *= 0.0

        # Kalman Tahmini (Prediction)
        self.kf.predict()
        
        self.age += 1
        if(self.time_since_update > 0):
            self.hit_streak = 0
        self.time_since_update += 1
        
        # Tahmin edilen yeni bbox'ı döndür
        self.bbox = self.convert_x_to_bbox(self.kf.statePost)
        return self.bbox

    def get_state(self):
        """Güncel bbox'ı döndür"""
        return self.convert_x_to_bbox(self.kf.statePost)

    def convert_bbox_to_z(self, bbox):
        """
        (x1, y1, x2, y2) -> (x, y, s, r) formatına çevir
        x,y: Merkez
        s: Alan
        r: Oran (w/h)
        """
        w = bbox[2] - bbox[0]
        h = bbox[3] - bbox[1]
        x = bbox[0] + w/2.
        y = bbox[1] + h/2.
        s = w * h
        r = w / float(h) if h > 0 else 1.0
        return np.array([[x],[y],[s],[r]]).astype(np.float32)

    def convert_x_to_bbox(self, x, score=None):
        """
        (x, y, s, r) -> (x1, y1, x2, y2) formatına çevir
        """
        w = np.sqrt(x[2] * x[3])
        h = x[2] / w
        
        # Float array dönüyor olabilir, int'e çevirelim
        x_center = x[0]
        y_center = x[1]
        
        return [
            int(x_center - w/2.), 
            int(y_center - h/2.), 
            int(x_center + w/2.), 
            int(y_center + h/2.)
        ]


class SORTTracker:
    """
    Ana Tracker Sınıfı
    """
    def __init__(self, max_age=30, min_hits=3, iou_threshold=0.3):
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.trackers = []
        self.frame_count = 0

    def update(self, detections: list) -> list:
        """
        Args:
            detections: Detection objelerinin listesi (Bizim Detection sınıfımız)
        Returns:
            tracks: Güncel Track objeleri listesi (Görselleştirme için)
        """
        self.frame_count += 1
        
        # 1. MEVCUT TRACKERLARI TAHMİN ET (PREDICT)
        trks = np.zeros((len(self.trackers), 5))
        to_del = []
        
        for t, trk in enumerate(trks):
            pos = self.trackers[t].predict() # Kalman tahmini
            trk[:] = [pos[0], pos[1], pos[2], pos[3], 0]
            if np.any(np.isnan(pos)):
                to_del.append(t)
                
        # Hatalı olanları temizle (numpy tersten silme)
        for t in reversed(to_del):
            self.trackers.pop(t)
            
        # 2. EŞLEŞTİRME (MATCHING)
        # Detection objelerini basit listeye [x1,y1,x2,y2] çevir
        dets = []
        for d in detections:
            dets.append(d.bbox)
        dets = np.array(dets)

        matched, unmatched_dets, unmatched_trks = self._match(dets, trks)

        # 3. GÜNCELLEME (UPDATE)
        
        # a) Eşleşenleri güncelle (Kalman Correction)
        for d, t in matched:
            # detections[d] bizim Detection objemiz
            # trackers[t] bizim KalmanBoxTracker objemiz
            self.trackers[t].update(
                dets[d], 
                detections[d].confidence, 
                detections[d].class_id
            )

        # b) Eşleşmeyen detectionlar için YENİ TRACKER oluştur
        for i in unmatched_dets:
            trk = KalmanBoxTracker(dets[i])
            trk.confidence = detections[i].confidence
            trk.class_id = detections[i].class_id
            self.trackers.append(trk)

        # c) Eşleşmeyen (kayıp) trackerları yönet
        # Zaten predict aşamasında time_since_update artırıldı.

        # 4. TEMİZLİK VE SONUÇ DÖNDÜRME
        ret = []
        i = len(self.trackers)
        for trk in reversed(self.trackers):
            d = trk.get_state()
            
            # Tracker onaylı mı ve hala görüntüde mi?
            if (trk.time_since_update < 1) and \
               (trk.hit_streak >= self.min_hits or self.frame_count <= self.min_hits):
                # Görselleştirme için bu trackerı döndür
                ret.append(trk)
            
            i -= 1
            # Ölü track silme
            if (trk.time_since_update > self.max_age):
                self.trackers.pop(i)

        if len(ret) > 0:
            return ret
        return []

    def _match(self, dets, trks):
        """Hungarian Algorithm ile eşleştirme"""
        # IOU Matrisi
        iou_matrix = np.zeros((len(dets), len(trks)), dtype=np.float32)
        for d, det in enumerate(dets):
            for t, trk in enumerate(trks):
                iou_matrix[d, t] = self._iou(det, trk)

        # Hungarian (Linear Assignment)
        if min(iou_matrix.shape) > 0:
            a = (iou_matrix > self.iou_threshold).astype(np.int32)
            # -iou çünkü fonksiyon min cost arar, biz max iou istiyoruz
            if a.sum(1).max() == 1 and a.sum(0).max() == 1:
                matched_indices = np.stack(np.where(a), axis=1)
            else:
                matched_indices = linear_sum_assignment(-iou_matrix)
                matched_indices = np.array(list(zip(*matched_indices)))
        else:
            matched_indices = np.empty((0, 2))

        unmatched_dets = []
        for d, det in enumerate(dets):
            if d not in matched_indices[:, 0]:
                unmatched_dets.append(d)
                
        unmatched_trks = []
        for t, trk in enumerate(trks):
            if t not in matched_indices[:, 1]:
                unmatched_trks.append(t)

        # IoU Threshold Kontrolü (Zorunlu)
        matches = []
        for m in matched_indices:
            if iou_matrix[m[0], m[1]] < self.iou_threshold:
                unmatched_dets.append(m[0])
                unmatched_trks.append(m[1])
            else:
                matches.append(m.reshape(1, 2))
                
        if len(matches) == 0:
            matches = np.empty((0, 2), dtype=int)
        else:
            matches = np.concatenate(matches, axis=0)

        return matches, unmatched_dets, unmatched_trks

    def _iou(self, bb_test, bb_gt):
        """elf.kf.statePost = self.convert_bbox_to_z(bbox)
        Intersection Over Union Hesapla
        bb: [x1, y1, x2, y2]
        """
        xx1 = max(bb_test[0], bb_gt[0])
        yy1 = max(bb_test[1], bb_gt[1])
        xx2 = min(bb_test[2], bb_gt[2])
        yy2 = min(bb_test[3], bb_gt[3])
        w = max(0., xx2 - xx1)
        h = max(0., yy2 - yy1)
        wh = w * h
        o = wh / ((bb_test[2] - bb_test[0]) * (bb_test[3] - bb_test[1]) +
                  (bb_gt[2] - bb_gt[0]) * (bb_gt[3] - bb_gt[1]) - wh)
        return o