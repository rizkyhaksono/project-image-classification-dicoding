# 🐾 Klasifikasi Gambar Animals-10 — Submission Akhir Dicoding

Proyek akhir kelas **Belajar Pengembangan Machine Learning** (Dicoding). Membangun model CNN
untuk mengklasifikasikan **10 jenis hewan**, lalu mengekspornya ke **SavedModel**, **TF-Lite**,
dan **TensorFlow.js**. Disusun untuk menargetkan **rating 5** (seluruh kriteria wajib + semua saran).

> 🖥️ Notebook dirancang untuk dijalankan di **Google Colab (T4 GPU)** — dataset ~26.000 gambar
> membutuhkan akselerasi GPU.

---

## 📁 Struktur Repositori

```
project-image-classification-dicoding/
├── README.md            ← Anda di sini (ikhtisar repo)
├── INSTRUCTIONS.md      ← panduan menjalankan di Colab & merakit + zip submission (TIDAK dikumpulkan)
└── submission/          ← folder yang di-zip & dikirim ke Dicoding
    ├── notebook.ipynb   ← notebook utama (harus sudah dijalankan & berisi output)
    ├── notebook.py      ← sumber jupytext (Dicoding mewajibkan .py & .ipynb)
    ├── README.md        ← README yang ikut dinilai
    ├── requirements.txt
    ├── saved_model/     ← saved_model.pb + variables/   (diisi setelah run di Colab)
    ├── tflite/          ← model.tflite + label.txt
    └── tfjs_model/      ← model.json + group1-shard*.bin
```

> ✏️ **Catatan untuk pengembang:** sumber kebenaran notebook adalah
> [`submission/notebook.py`](submission/notebook.py) (format *jupytext percent*). Setelah
> mengeditnya, regenerasi `.ipynb` dengan:
> ```bash
> jupytext --to ipynb submission/notebook.py
> ```

---

## 🚀 Cara Cepat (Google Colab)

1. Buka [`submission/notebook.ipynb`](submission/notebook.ipynb) di Colab →
   **Runtime → Change runtime type → T4 GPU**.
2. Siapkan kredensial Kaggle (salah satu): **`kaggle.json`** (Settings → API → *Create Legacy API
   Key*) **atau** token **`KGAT_...`** (*Generate New Token*). Notebook menerima keduanya.
3. **Runtime → Run all.** Notebook akan mengunduh dataset, melatih model, membuat plot, melakukan
   inferensi, dan mengekspor tiga format model.
4. Rakit & kompres submission sesuai [`INSTRUCTIONS.md`](INSTRUCTIONS.md).

---

## 🧠 Model & Dataset

- **Dataset:** [Animals-10 (Kaggle)](https://www.kaggle.com/datasets/alessiocorrado99/animals10) —
  ~26.000 gambar, **10 kelas** (dog, horse, elephant, butterfly, chicken, cat, cow, sheep, spider,
  squirrel), **resolusi tidak seragam**.
- **Split:** train **80%** / validation **10%** / test **10%**.
- **Arsitektur** (`tf.keras.Sequential`):

  ```
  Input(224×224×3)
   → Data Augmentation (RandomFlip / RandomRotation / RandomZoom)   # aktif hanya saat training
   → Rescaling ke [-1, 1]                                            # preprocessing MobileNetV2
   → MobileNetV2 (include_top=False, bobot ImageNet)                # transfer learning
   → Conv2D(256) → MaxPooling2D → Conv2D(128)                        # Conv2D + Pooling eksplisit
   → GlobalAveragePooling2D → Dropout → Dense(128) → Dropout
   → Dense(10, softmax)
  ```

- **Pelatihan:** dua tahap — *feature extraction* (basis dibekukan) lalu *fine-tuning*
  (membuka ~40 lapisan teratas, *learning rate* kecil) untuk menembus akurasi 95%.

---

## ✅ Pemenuhan Kriteria

| Kriteria / Saran | Status |
|---|---|
| Dataset ≥ 1.000 gambar, bukan RPS/X-Ray | ✅ Animals-10 (~26k) |
| Split train / validation / test | ✅ 80 / 10 / 10 |
| Sequential + Conv2D + Pooling | ✅ `Conv2D` + `MaxPooling2D` eksplisit di head |
| Akurasi train & test ≥ 85% | ✅ Target ≥ 95% |
| Plot akurasi & loss | ✅ |
| Ekspor SavedModel + TF-Lite + TFJS | ✅ |
| *(Saran)* Callback | ✅ EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, callback kustom |
| *(Saran)* Resolusi tidak seragam | ✅ Dibuktikan di EDA |
| *(Saran)* ≥ 10.000 gambar | ✅ ~26.000 |
| *(Saran)* Akurasi ≥ 95% | ✅ Target arsitektur |
| *(Saran)* ≥ 3 kelas | ✅ 10 kelas |
| *(Saran)* Inferensi + bukti | ✅ Demo TF-Lite di notebook |

---

## 🛠️ Dependensi

Lingkungan acuan: **Python 3.12, TensorFlow 2.19, Keras 3** (default Colab). Lihat
[`submission/requirements.txt`](submission/requirements.txt). Dapat dibuat ulang dengan
`pip freeze > requirements.txt` atau `pipreqs` agar lebih ringkas.

## 📄 Lisensi & Atribusi

Dataset Animals-10 merupakan dataset publik untuk keperluan edukasi; hak cipta gambar milik
pemilik aslinya. Proyek ini dibuat sebagai submission pembelajaran di Dicoding.
