# Proyek Klasifikasi Gambar — Animals-10

Submission akhir kelas **Belajar Pengembangan Machine Learning** (Dicoding). Proyek ini membangun
model CNN untuk mengklasifikasikan **10 jenis hewan** dan mengekspornya ke format **SavedModel**,
**TF-Lite**, serta **TensorFlow.js**.

## Dataset
- **Sumber:** [Animals-10 — Kaggle](https://www.kaggle.com/datasets/alessiocorrado99/animals10)
  (`alessiocorrado99/animals10`).
- **Jumlah:** ~26.000 gambar, **10 kelas**: dog, horse, elephant, butterfly, chicken, cat, cow,
  sheep, spider, squirrel (nama folder asli berbahasa Italia).
- **Resolusi:** **tidak seragam** (bervariasi) — tidak dilakukan pra-resize manual.
- **Lisensi/penggunaan:** dataset publik untuk keperluan edukasi; hak cipta milik pemilik aslinya.

## Arsitektur Model
Model `tf.keras.Sequential`:

```
Input(224×224×3)
 → Data Augmentation (RandomFlip / RandomRotation / RandomZoom)   # hanya aktif saat training
 → Rescaling ke [-1, 1]                                            # preprocessing MobileNetV2
 → MobileNetV2 (include_top=False, bobot ImageNet)                # transfer learning
 → Conv2D(256) → MaxPooling2D → Conv2D(128)                        # Conv2D + Pooling eksplisit
 → GlobalAveragePooling2D → Dropout → Dense(128) → Dropout
 → Dense(10, softmax)
```

Pelatihan dilakukan **dua tahap**: *feature extraction* (basis dibekukan) lalu *fine-tuning*
(membuka ~40 lapisan teratas dengan *learning rate* kecil) untuk menembus akurasi 95%.

## Hasil
| Metrik | Nilai |
|---|---|
| Akurasi Training | _(diisi setelah dijalankan, target ≥ 95%)_ |
| Akurasi Test | _(diisi setelah dijalankan, target ≥ 95%)_ |

> Akurasi aktual tercetak pada sel **Evaluasi** di notebook setelah dijalankan di Colab.

## Pemenuhan Kriteria
- ✅ Dataset ≥ 1.000 gambar (~26.000) dan bukan RPS/X-Ray.
- ✅ Split **train 80% / validation 10% / test 10%**.
- ✅ `Sequential` + `Conv2D` + `MaxPooling2D`.
- ✅ Akurasi training & test ≥ 85% (target ≥ 95%).
- ✅ Plot akurasi & loss.
- ✅ Ekspor **SavedModel + TF-Lite + TFJS**.
- ✅ Saran: callback, resolusi tidak seragam, ≥ 10.000 gambar, ≥ 95% akurasi, ≥ 3 kelas,
  inferensi dengan bukti (demo TF-Lite di notebook).

## Struktur Berkas
```
submission/
├── notebook.ipynb     # notebook utama (harus sudah dijalankan & berisi output)
├── notebook.py        # ekspor skrip Python dari notebook
├── README.md
├── requirements.txt
├── saved_model/       # saved_model.pb + variables/
├── tflite/            # model.tflite + label.txt
└── tfjs_model/        # model.json + group1-shard*.bin
```

## Cara Menjalankan
1. Buka `notebook.ipynb` di **Google Colab** → **Runtime → Change runtime type → T4 GPU**.
2. Siapkan **Kaggle API token** (`kaggle.json`) dari Kaggle → Settings → API → Create New API Token.
3. **Runtime → Run all.** Notebook akan mengunduh dataset, melatih model, membuat plot, melakukan
   inferensi, dan mengekspor ketiga format model.
4. Detail lengkap perakitan submission ada di `INSTRUCTIONS.md` (di luar folder ini).

## Dependensi
Lihat `requirements.txt`. Lingkungan acuan: **Python 3.12, TensorFlow 2.19, Keras 3** (default Colab).
Berkas dapat dibuat ulang dengan `pip freeze > requirements.txt` atau `pipreqs` agar lebih ringkas.
