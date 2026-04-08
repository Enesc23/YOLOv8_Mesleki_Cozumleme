from ultralytics import YOLO
import os

def train_model():
    # 1. Model Seçimi: YOLOv8 Nano (hızlı ve hafif olması için)
    # Eğer elinde eğitilmiş bir model varsa onun ismini de yazabilirsin
    model = YOLO('yolov8n.pt') 

    # 2. Eğitim Parametreleri
    # data: data.yaml dosyasının yolu
    # epochs: Eğitim tur sayısı (50 idealdir)
    # imgsz: Görsel boyutu (640 standarttır)
    # device: Eğer ekran kartın varsa 0, yoksa 'cpu' yazabilirsin
    results = model.train(
        data='data.yaml', 
        epochs=50, 
        imgsz=640, 
        device=0,  # GPU yoksa 'cpu' yap
        plots=True, # Grafiklerin (runs klasörü) oluşması için
        save=True,  # En iyi modelin kaydedilmesi için
        name='kask_sapka_egitim' # Klasör ismi
    )

    print("Eğitim başarıyla tamamlandı. Sonuçlar 'runs' klasörüne kaydedildi.")

if __name__ == '__main__':
    train_model()