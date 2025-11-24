# pipeline_functions/visualizer.py

import cv2
import numpy as np

class FrameVisualizer:
    def __init__(self):
        self.colors = np.random.randint(0, 255, (1000, 3)).tolist()

    def draw_results(self, frame: np.ndarray, tracks: list, roi_rect: tuple = None,
                     fps: float = 0.0, count: int = 0) -> np.ndarray:
        """
        Args:
            frame: Orijinal Global Frame
            tracks: Local koordinatlardaki trackler
            roi_rect: (off_x, off_y, w, h) -> Offset bilgisi burada!
        """
        viz_frame = frame.copy()
        
        # Offsetleri hazırla (Eğer ROI yoksa offset 0'dır)
        offset_x, offset_y = 0, 0
        if roi_rect:
            offset_x, offset_y, w, h = roi_rect
            
            # ROI Alanını Çiz (Mavi Kutu)
            cv2.rectangle(viz_frame, (offset_x, offset_y), 
                         (offset_x + w, offset_y + h), (255, 0, 0), 2)
            
            cv2.putText(viz_frame, "ROI (Analiz Alani)", (offset_x, offset_y - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

        # Trackleri Çiz
        for track in tracks:
            # 1. Local Koordinatları Al
            lx1, ly1, lx2, ly2 = track.bbox
            
            # 2. Global'e Çevir (Dediğin Mantık Burada)
            gx1 = lx1 + offset_x
            gy1 = ly1 + offset_y
            gx2 = lx2 + offset_x
            gy2 = ly2 + offset_y
            
            # 3. Renk Seçimi
            color = self.colors[track.id % len(self.colors)]
            
            # 4. Çizim (Global koordinatlarla)
            cv2.rectangle(viz_frame, (gx1, gy1), (gx2, gy2), color, 2)
            
            # Etiket
            label = f"ID:{track.id} {track.confidence:.2f}"
            
            # Etiket arka planı
            (lbl_w, lbl_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
            cv2.rectangle(viz_frame, (gx1, gy1 - 20), (gx1 + lbl_w, gy1), color, -1)
            
            cv2.putText(viz_frame, label, (gx1, gy1 - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)


        #fps - counter
        cv2.rectangle(viz_frame, (0, 0), (250, 80), (0, 0, 0), -1)
        cv2.rectangle(viz_frame, (0, 0), (250, 80), (255, 255, 255), 2) # Beyaz çerçeve
        # FPS Yazısı
        cv2.putText(viz_frame, f"FPS: {fps:.1f}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        # Sayaç Yazısı
        cv2.putText(viz_frame, f"Count: {count}", (10, 65), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 200, 255), 2)

        return viz_frame