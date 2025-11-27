import torch
import os
import time
from diffusers import DiffusionPipeline, DPMSolverMultistepScheduler
from diffusers.utils import export_to_video
import pynvml 

class SyntheticDataGenerator:
    def __init__(self, output_dir: str = "data/input/synthetic"):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        self.pipeline = None

    def load_model(self):
        print("🤖 Model yükleniyor... (fp16 + CPU offload)")
        self.pipeline = DiffusionPipeline.from_pretrained(
            "damo-vilab/text-to-video-ms-1.7b",
            torch_dtype=torch.float16,
            variant="fp16"
        )
        self.pipeline.scheduler = DPMSolverMultistepScheduler.from_config(
            self.pipeline.scheduler.config
        )
        # VRAM tasarrufu için kritik:
        self.pipeline.enable_model_cpu_offload()
        # Veya VRAM çok azsa şunu dene (daha yavaştır ama az yer kaplar):
        # self.pipeline.enable_sequential_cpu_offload()
        print("✅ Model yüklendi.")

    def gpu_status(self):
        try:
            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
            util = pynvml.nvmlDeviceGetUtilizationRates(handle)
            print(f"📊 GPU: {util.gpu}% | VRAM: {mem.used / 1024**2:.0f}MB / {mem.total / 1024**2:.0f}MB")
        except:
            pass

    def generate_video(self, prompt: str, filename: str, num_frames: int = 64):
        if not self.pipeline:
            self.load_model()

        print(f"🎬 Video üretiliyor (Batch Mode): '{prompt}'...")
        self.gpu_status()
        
        try:
            # --- DÜZELTME BURADA ---
            # Döngü YOK! Hepsini tek seferde istiyoruz.
            # num_frames 16 veya 24 idealdir. Çok artırırsan VRAM yetmez.
            video_frames = self.pipeline(
                prompt, 
                negative_prompt="blur, low quality, noise, distortion, watermark, text, lowres, jpeg artifacts",
                num_frames=num_frames,
                num_inference_steps=50,  # DAHA TEMİZ GÖRÜNTÜ İÇİN (Eskisi 25'ti)
                guidance_scale=7.5,      # Prompt'a ne kadar sadık kalacağı
                height=256, width=256    # VRAM varsa 320x320 deneyebilirsin ama riskli
            ).frames[0]
            
            output_path = os.path.join(self.output_dir, filename)
            export_to_video(video_frames, output_path, fps=8)
            print(f"💾 Video başarıyla kaydedildi: {output_path}")
            
        except Exception as e:
            print(f"❌ HATA: {e}")
            print("VRAM yetmemiş olabilir. num_frames değerini düşürün (örn: 10) veya Online Tool kullanın.")

if __name__ == "__main__":
    gen = SyntheticDataGenerator()
    # Fabrika ortamı için biraz daha detaylı prompt
    gen.generate_video(
#        prompt="A continuous shot of a factory worker walking in a warehouse, wearing a yellow hardhat, realistic 4k, slow motion",
        prompt="Child trying to walk",
        filename="worker_gen.mp4",
        num_frames=16 # 16 kare standarttır (yaklaşık 2 saniye)
    )