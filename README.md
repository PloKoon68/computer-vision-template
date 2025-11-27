# 👁️ Real-Time Video Analytics Pipeline

    

Bu proje, güvenlik kameraları ve video akışları için geliştirilmiş; **modüler**, **ölçeklenebilir** ve **dağıtıma hazır** bir bilgisayarlı görü (Computer Vision) hattıdır (pipeline).

**Temel Yetenekler:**

  * 🚀 **Nesne Tespiti:** YOLOv8 (State-of-the-Art)
  * 🎯 **Nesne Takibi:** SORT (Kalman Filter + Hungarian Algorithm)
  * 📊 **Analitik:** Gerçek zamanlı FPS, nesne sayımı ve ROI analizi.
  * 🤖 **Sentetik Veri:** Text-to-Video modelleri ile uç durum (edge-case) testi.
  * 🌐 **Servis:** FastAPI tabanlı REST API.

-----

## 🏗️ Mimari ve Tasarım Prensipleri

Proje, **"Separation of Concerns"** (İlgi Alanlarının Ayrımı) ve **"Dependency Injection"** prensipleri gözetilerek geliştirilmiştir.

### 1\. Modüler Yapı (Dependency Injection)

`Pipeline` sınıfı, `Detector` veya `Tracker` sınıflarına sıkı sıkıya bağlı (tightly coupled) değildir. Bu bileşenler `main.py` veya `app.py` içerisinde oluşturulup Pipeline'a enjekte edilir.

  * **Avantajı:** Bu sayede gelecekte YOLO yerine **Faster-RCNN** veya SORT yerine **DeepSORT** kullanmak istenirse, sadece ilgili sınıfı değiştirmek yeterlidir; Pipeline mantığına dokunulmaz. Ayrıca **Unit Test** yazarken Mock objelerle test etmeyi kolaylaştırır.

### 2\. Görüntü İşleme (Preprocessing) Yaklaşımı

  * **ROI (Region of Interest):** Kullanıcı, videonun sadece belirli bir yüzdelik alanını (örn: `%20` ile `%80` arası) işleyebilir.
      * *Neden?* FPS artışı sağlar, işlemciyi rahatlatır ve modelin ilgisiz arka plana odaklanmasını engelleyerek **False Positive** oranını düşürür.
  * **Neden Blur/Grayscale Yok?** YOLO gibi modern CNN (Convolutional Neural Network) modelleri, özellik çıkarımı (feature extraction) sırasında gürültüye karşı zaten dirençlidir. Geleneksel yöntemlerdeki gibi (Canny Edge vb.) ön işlemeye ihtiyaç duymazlar. Gereksiz işlem yükünden kaçınılmıştır.

### 3\. Takip Algoritması (SORT Implementation)

Bu projede Alex Bewley'in orijinal SORT algoritması referans alınmıştır. Ancak kod kopyalanmamış, projenin OOP yapısına ve **`Detection`** veri sınıfına uygun şekilde **refactor** edilmiştir.

  * **Bilinen Kısıtlar:** SORT sadece konum ve hız (Motion-based) takibi yapar. Uzun süreli kapanmalarda (Occlusion) görsel hafızası olmadığı için **ID Switching** (Kimlik değişimi) yaşanabilir.
  * *Not:* Production ortamında donanım elverirse DeepSORT entegrasyonu planlanmıştır.

-----

## 🛠️ Kurulum (Installation)

Proje hem yerel geliştirme ortamında (Local Developer) hem de Docker konteynerinde çalışacak şekilde tasarlanmıştır.

### Ön Gereksinimler

  * **Python:** 3.9 veya üzeri (Type hinting `list[str]` ve `| None` desteği için).
  * **GPU:** NVIDIA GPU ve güncel sürücüler (Önerilen).

-----

### A. Geliştirici Modu (Local Setup)

Bu mod, kodu geliştirmek ve debug etmek içindir.

1.  **Repo'yu klonlayın ve dizine girin:**

    ```bash
    git clone https://github.com/yourusername/vision-pipeline.git
    cd vision-pipeline
    ```

2.  **Sanal ortam oluşturun:**

    ```bash
    python -m venv venv
    # Windows:
    venv\Scripts\activate
    # Linux/Mac:
    source venv/bin/activate
    ```

3.  **⚠️ Kritik Adım: GPU Desteği (PyTorch)**
    `requirements.txt` dosyası donanım bağımsızdır. GPU hızlandırmasından faydalanmak için bilgisayarınızdaki CUDA sürümüne uygun PyTorch'u manuel kurmalısınız.
    *(Örnek: CUDA 12.x için)*

    ```bash
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
    ```

    *Eğer GPU yoksa bu adımı atlayabilirsiniz, CPU sürümü otomatik kurulacaktır.*

4.  **Diğer bağımlılıkları yükleyin:**

    ```bash
    pip install -r requirements.txt
    ```

-----

### B. Konteyner Modu (Docker Deployment)

Bu mod, API servisini izole bir ortamda sunmak içindir.

**⚠️ ÖNEMLİ:** Docker içinde GPU kullanabilmek için host makinede sadece Docker değil, **NVIDIA Container Toolkit** de kurulu olmalıdır.

1.  **İmajı Oluşturun (Build):**
    Docker imajı, CUDA 12.1 destekli resmi PyTorch imajını baz alır. Manuel Torch kurulumu gerekmez.

    ```bash
    docker build -t vision-app .
    ```

2.  **Konteyneri Başlatın (Run):**
    GPU erişimi vererek API'yi 8000 portunda başlatın.

    ```bash
    docker run --gpus all -p 8000:8000 vision-app
    ```

    *Eğer `nvidia-container-toolkit` yoksa `--gpus all` parametresini kaldırın (CPU modunda çalışır).*

-----

## 🚀 Kullanım (Usage)

### 1\. Komut Satırı Arayüzü (CLI)

Videoları terminal üzerinden işlemek için `main.py` kullanılır. Çıktılar `data/output/` altına, tarih damgalı klasörler halinde kaydedilir (**Artifact Encapsulation**).

```bash
# Temel Kullanım
python src/main.py --input_path data/input/videos/sample.mp4

# Gelişmiş Kullanım (ROI Belirleme)
# ROI Formatı: x_start y_start width height (Yüzdelik: 0.0 - 1.0 arası)
# Örn: %10 soldan, %20 üstten başla, %50 genişlik ve %50 yükseklik al.
python src/main.py --input my_video --roi 0.1 0.2 0.5 0.5 --conf 0.6
```

### 2\. REST API (FastAPI)

Sistemi bir mikroservis olarak kullanmak için `app.py` veya Docker kullanılır.

  * **Başlatma:** `python src/app.py`
  * **Dokümantasyon:** Tarayıcıda `http://localhost:8000/docs` adresine giderek Swagger UI üzerinden video yükleyip test edebilirsiniz.

-----

## 🤖 Sentetik Veri Üretimi (GenAI)

Modelin zorlu koşullardaki (örn: karanlık, sisli fabrika ortamı) başarısını test etmek için **Text-to-Video** teknolojisi kullanılmıştır.

  * **Script:** `src/data_generation/generator.py`
  * **Model:** ModelScope (damo-vilab/text-to-video-ms-1.7b)
  * **Kullanım:**
    ```python
    python src/data_generation/generator.py
    ```
    *Not: Bu işlem yüksek VRAM (GPU Hafızası) gerektirir. Donanım kısıtlarında Luma/RunwayML gibi online araçlar alternatif olarak kullanılabilir.*

-----

## 🧪 Testler

Proje, birim (unit) ve entegrasyon testlerini içerir. Mocking kullanılarak, ağır modelleri yüklemeden pipeline mantığı test edilmiştir.

```bash
pytest
```

-----

## 🔮 Gelecek Planları (Future Work)

  * [ ] **DeepSORT / ByteTrack:** ID Switching sorununu azaltmak için görsel özellik (Re-ID) kullanan tracker entegrasyonu.
  * [ ] **Model Seçimi:** Kullanıcının API isteğiyle model boyutunu (yolov8n, yolov8x) seçebilmesi.
  * [ ] **Database:** Analitik verilerinin (AnalyticsManager) PostgreSQL/MongoDB'ye yazılması.

-----

## ⚖️ Lisans ve Etik

  * Bu proje MIT lisansı ile sunulmuştur.
  * Kullanılan YOLOv8 modeli AGPL-3.0, SORT algoritması GPL-3.0 lisanslarına tabidir.
  * Sentetik veriler test amaçlı üretilmiştir, gerçek kişilerin gizliliğini ihlal etmez.