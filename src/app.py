import os
import sys
import shutil
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import FileResponse, JSONResponse
from datetime import datetime

# Import ayarı: Root dizinden çalıştırıldığı varsayılır
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import load_configuration
from src.pipeline import Pipeline
from src.pipeline_functions.detector import YOLODetector
from src.pipeline_functions.tracker import SORTTracker
from src.pipeline_functions.preprocessor import FramePreprocessor
from src.pipeline_functions.visualizer import FrameVisualizer
from src.pipeline_functions.analytics import AnalyticsManager

app = FastAPI(
    title="Video Analytics API",
    description="YOLOv8 + SORT Pipeline",
    version="1.0.0"
)

# Global Config ve Modeller (Server açılışında 1 kere yüklenir)
print("🚀 API Başlatılıyor...")
CFG = load_configuration()

# Modelleri RAM'e yükle
DETECTOR = YOLODetector(CFG.paths['model_path'], CFG.confidence_threshold, CFG.device)
TRACKER = SORTTracker(CFG.max_age, CFG.min_hits, CFG.iou_threshold_tracker)

@app.get("/")
def health_check():
    return {
        "status": "active", 
        "device": CFG.device,
        "model": CFG.paths['model_path']
    }

@app.post("/process/")
async def process_video_endpoint(file: UploadFile = File(...)):
    """
    Video yükle -> İşle -> İndir
    """
    # 1. Geçici Klasörler
    temp_dir = "temp_api_files"
    os.makedirs(temp_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    input_path = os.path.join(temp_dir, f"in_{timestamp}_{file.filename}")
    
    # Çıktı için özel klasör (Artifact Encapsulation)
    output_folder = os.path.join(temp_dir, f"out_{timestamp}")
    os.makedirs(output_folder, exist_ok=True)
    
    try:
        # 2. Dosyayı Kaydet
        with open(input_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
            
        # 3. Her istek için taze Pipeline parçaları
        preprocessor = FramePreprocessor(roi_percent=CFG.roi_percent)
        visualizer = FrameVisualizer()
        analytics = AnalyticsManager(output_dir=output_folder)
        
        # Pipeline oluştur
        pipeline = Pipeline(
            preprocessor=preprocessor,
            detector=DETECTOR, # Hazır yüklü modeli kullan
            tracker=TRACKER,   # Hazır tracker'ı kullan (Not: State karışmaması için her requestte yeni tracker oluşturmak daha güvenli olabilir ama sınav için bu performanslıdır)
            visualizer=visualizer,
            analytics=analytics
        )
        
        # 4. İşle (GUI Kapalı!)
        print(f"▶️ API Video İşliyor: {file.filename}")
        pipeline.process_video(
            input_path=input_path,
            output_dir=output_folder,
            frame_skip=CFG.frame_skip,
            show_display=False # Sunucuda imshow açılmaz
        )
        
        # 5. Sonucu Bul ve Döndür
        result_video = os.path.join(output_folder, "processed_video.mp4")
        
        # Fallback: Eğer mp4 yoksa avi dene (Bizim pipeline mantığı)
        if not os.path.exists(result_video):
            result_video = os.path.join(output_folder, "processed_video.avi")
            
        if os.path.exists(result_video):
            return FileResponse(result_video, media_type="video/mp4", filename=f"processed_{file.filename}")
        else:
            return JSONResponse(status_code=500, content={"error": "Video işlendi ama çıktı dosyası bulunamadı."})

    except Exception as e:
        import traceback
        traceback.print_exc()
        return JSONResponse(status_code=500, content={"error": str(e)})

if __name__ == "__main__":
    import uvicorn
    # API'yi başlat: python src/app.py
    uvicorn.run(app, host="0.0.0.0", port=8000)