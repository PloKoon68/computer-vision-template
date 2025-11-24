import time
import json
import csv
from collections import defaultdict
from typing import Dict
import os

class AnalyticsManager:
    """
    Sistemin performans ve sayım metriklerini tutan sınıf.
    """
    def __init__(self, output_dir: str = "outputs"):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Metrikler
        self.start_time = time.time()
        self.total_frames = 0
        self.processing_times = []  # Her karenin işlem süresi
        
        # Sayım (Set kullanarak unique ID'leri tutuyoruz)
        # Örn: {0: {4, 5, 8}, 2: {1, 12}} -> Class ID 0 (İnsan): 4, 5, 8 nolu ID'ler
        self.unique_counts: Dict[int, set] = defaultdict(set)
        
        # Anlık FPS hesabı için
        self.fps_start_time = time.time()
        self.fps_frame_counter = 0
        self.current_fps = 0.0

    def update(self, tracks: list, process_time: float):
        """
        Her frame sonunda çağrılır.
        
        Args:
            tracks: Tracker'dan dönen aktif track listesi
            process_time: O karenin işlenme süresi (saniye)
        """
        self.total_frames += 1
        self.processing_times.append(process_time)
        
        # 1. Sayım İşlemi
        for track in tracks:
            # track.class_id'ye ait kümeye track.id'yi ekle
            # Küme (set) olduğu için aynı ID tekrar eklenmez (Unique Count)
            self.unique_counts[track.class_id].add(track.id)
            
        # 2. FPS Hesabı (Her 30 karede bir güncelle - okuması kolay olsun)
        self.fps_frame_counter += 1
        if self.fps_frame_counter >= 30:
            duration = time.time() - self.fps_start_time
            self.current_fps = self.fps_frame_counter / duration
            
            # Sıfırla
            self.fps_start_time = time.time()
            self.fps_frame_counter = 0

    def get_metrics(self) -> dict:
        """Ekrana yazdırmak için anlık metrikleri döndür"""
        total_objects = sum(len(ids) for ids in self.unique_counts.values())
        avg_process_time = (sum(self.processing_times) / len(self.processing_times)) * 1000 if self.processing_times else 0
        
        return {
            "fps": self.current_fps,
            "frame_count": self.total_frames,
            "total_unique_objects": total_objects,
            "avg_latency_ms": avg_process_time
        }

    def save_report(self, filename: str = "analysis_report.json"):
        """
        İşlem bitince final raporu kaydeder.
        """
        total_duration = time.time() - self.start_time
        avg_fps = self.total_frames / total_duration if total_duration > 0 else 0
        
        # Sınıf bazlı sayımları sayıya çevir (Set -> Int)
        class_counts = {cls_id: len(ids) for cls_id, ids in self.unique_counts.items()}
        
        report = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "duration_seconds": round(total_duration, 2),
            "total_frames_processed": self.total_frames,
            "average_fps": round(avg_fps, 2),
            "total_unique_objects": sum(class_counts.values()),
            "class_breakdown": class_counts,
            "performance": {
                "min_latency_ms": round(min(self.processing_times) * 1000, 2) if self.processing_times else 0,
                "max_latency_ms": round(max(self.processing_times) * 1000, 2) if self.processing_times else 0,
                "avg_latency_ms": round((sum(self.processing_times) / len(self.processing_times)) * 1000, 2) if self.processing_times else 0
            }
        }
        
        # JSON Kaydet
        file_path = os.path.join(self.output_dir, filename)
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=4, ensure_ascii=False)
            
        print(f"📊 Analiz raporu kaydedildi: {file_path}")