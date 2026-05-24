# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.4
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Proyek Klasifikasi Gambar — Animals-10
#
# **Submission Akhir — Belajar Pengembangan Machine Learning (Dicoding)**
#
# Notebook ini membangun model *Convolutional Neural Network* (CNN) untuk mengklasifikasikan
# gambar **10 jenis hewan** menggunakan dataset publik **[Animals-10](https://www.kaggle.com/datasets/alessiocorrado99/animals10)**
# dari Kaggle (~26.000 gambar, resolusi tidak seragam).
#
# Model dibangun dengan **`tf.keras.Sequential`** yang menggabungkan *transfer learning*
# (basis **MobileNetV2**) dengan lapisan **`Conv2D`** dan **`MaxPooling2D`** buatan sendiri,
# lalu diekspor ke tiga format: **SavedModel**, **TF-Lite**, dan **TensorFlow.js**.
#
# ## Pemenuhan Kriteria
#
# | Kriteria / Saran | Pemenuhan |
# |---|---|
# | Dataset ≥ 1.000 gambar | Animals-10 (~26.000 gambar) |
# | Bukan dataset RPS / X-Ray | Animals-10 (10 kelas hewan) |
# | Split train / validation / test | 80% / 10% / 10% |
# | Sequential + Conv2D + Pooling | Lihat sel **Arsitektur Model** |
# | Akurasi train & test ≥ 85% | Target ≥ 95% (transfer learning + fine-tuning) |
# | Plot akurasi & loss | Lihat sel **Visualisasi** |
# | Ekspor SavedModel, TF-Lite, TFJS | Lihat sel-sel **Ekspor Model** |
# | *(Saran)* Callback | EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, callback kustom |
# | *(Saran)* Resolusi tidak seragam | Dibuktikan pada EDA (tanpa pra-resize manual) |
# | *(Saran)* ≥ 10.000 gambar | ~26.000 gambar |
# | *(Saran)* Akurasi ≥ 95% | Target arsitektur ini |
# | *(Saran)* ≥ 3 kelas | 10 kelas |
# | *(Saran)* Inferensi + bukti | Demo inferensi model TF-Lite di akhir notebook |
#
# > **Catatan eksekusi:** jalankan di **Google Colab** dengan **Runtime → T4 GPU**
# > (`Runtime → Change runtime type → T4 GPU`), lalu `Runtime → Run all`.

# %% [markdown]
# ## 1. Persiapan & Unduh Dataset
#
# ### Autentikasi Kaggle
# Kaggle mendukung **dua jenis kredensial** — sel di bawah menerima keduanya:
#
# 1. **`kaggle.json` (Legacy API Key)** — di halaman **Kaggle → Settings → API**, scroll ke
#    *Legacy API Credentials* → klik **Create Legacy API Key** → berkas `kaggle.json` terunduh.
# 2. **Token akses baru `KGAT_...`** — tombol **Generate New Token** di bagian *API Tokens*.
#
# Saat sel dijalankan: **unggah `kaggle.json`** bila punya, atau **batalkan unggahan** lalu
# **tempel token `KGAT_...`** saat diminta (token tidak ikut tersimpan di notebook).

# %%
# Perlu kaggle CLI >= 1.8.0 agar token KGAT_ didukung; --upgrade memastikan versinya cukup baru.
!pip install -q --upgrade kaggle

# %%
import os
import getpass

KAGGLE_DIR = os.path.expanduser("~/.kaggle")
os.makedirs(KAGGLE_DIR, exist_ok=True)


def _have_kaggle_creds():
    return (os.path.exists(f"{KAGGLE_DIR}/kaggle.json")
            or os.path.exists(f"{KAGGLE_DIR}/access_token")
            or os.environ.get("KAGGLE_API_TOKEN"))


if not _have_kaggle_creds():
    uploaded = {}
    try:
        from google.colab import files
        print("Unggah kaggle.json (Legacy API Key). "
              "Klik 'Cancel' jika ingin memakai token KGAT_... sebagai gantinya.")
        uploaded = files.upload()
    except Exception:
        pass  # bukan di Colab atau unggahan dibatalkan

    if "kaggle.json" in uploaded or os.path.exists("kaggle.json"):
        # Opsi 1: kaggle.json (legacy)
        os.replace("kaggle.json", f"{KAGGLE_DIR}/kaggle.json")
        os.chmod(f"{KAGGLE_DIR}/kaggle.json", 0o600)
    else:
        # Opsi 2: token akses baru (KGAT_...) — input aman, tidak tercetak / tidak tersimpan.
        token = getpass.getpass("Tempel API token KGAT_... lalu Enter: ").strip()
        with open(f"{KAGGLE_DIR}/access_token", "w") as f:
            f.write(token)
        os.chmod(f"{KAGGLE_DIR}/access_token", 0o600)

print("Kredensial Kaggle siap.")

# %%
# Unduh & ekstrak dataset (~26k gambar). Diunduh sekali; lewati jika folder sudah ada.
if not os.path.isdir("data/raw-img"):
    !kaggle datasets download -d alessiocorrado99/animals10
    !unzip -q animals10.zip -d data
print("Isi folder data/:", os.listdir("data"))

# %%
# Import seluruh library dan tetapkan seed agar hasil reproducible.
import random
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras import layers

SEED = 123
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

print("TensorFlow :", tf.__version__)
print("Keras      :", tf.keras.__version__)
print("GPU        :", tf.config.list_physical_devices("GPU") or "TIDAK ADA — gunakan Runtime T4 GPU!")

# %% [markdown]
# ## 2. Exploratory Data Analysis (EDA)
#
# Kita memeriksa jumlah gambar per kelas dan **ragam resolusi gambar asli**. Dataset ini
# berisi gambar dengan ukuran yang **tidak seragam** (memenuhi salah satu saran penilaian):
# kita tidak melakukan pra-resize manual — proses *resize* hanya dilakukan di dalam pipeline
# pelatihan ke ukuran input model.

# %%
from PIL import Image

DATA_DIR = "data/raw-img"

# Pemetaan nama folder (Bahasa Italia) -> Bahasa Inggris (dari translate.py milik dataset).
IT_TO_EN = {
    "cane": "dog",       "cavallo": "horse",  "elefante": "elephant",
    "farfalla": "butterfly", "gallina": "chicken", "gatto": "cat",
    "mucca": "cow",      "pecora": "sheep",   "ragno": "spider",
    "scoiattolo": "squirrel",
}


def explore_dataset(directory):
    """Cetak jumlah gambar per kelas + ragam resolusi (membuktikan resolusi tidak seragam)."""
    total_images = 0
    all_sizes = set()
    print(f"{'Kelas (IT)':<12}{'Kelas (EN)':<12}{'Jumlah':>8}")
    print("-" * 32)
    for subdir in sorted(os.listdir(directory)):
        subdir_path = os.path.join(directory, subdir)
        if not os.path.isdir(subdir_path):
            continue
        image_files = os.listdir(subdir_path)
        total_images += len(image_files)
        print(f"{subdir:<12}{IT_TO_EN.get(subdir, '?'):<12}{len(image_files):>8}")
        # Sampel 200 gambar per kelas untuk mencatat ragam ukuran (cepat, cukup representatif).
        for img_file in image_files[:200]:
            try:
                with Image.open(os.path.join(subdir_path, img_file)) as img:
                    all_sizes.add(img.size)
            except Exception:
                pass
    print("-" * 32)
    print(f"TOTAL gambar          : {total_images}")
    print(f"Ragam resolusi unik   : {len(all_sizes)} ukuran berbeda (tidak seragam)")
    widths = [w for w, h in all_sizes]
    heights = [h for w, h in all_sizes]
    print(f"Lebar  (min..max)     : {min(widths)} .. {max(widths)} px")
    print(f"Tinggi (min..max)     : {min(heights)} .. {max(heights)} px")
    print(f"Contoh ukuran         : {list(sorted(all_sizes))[:8]} ...")
    return total_images


total = explore_dataset(DATA_DIR)

# %%
# Tampilkan beberapa contoh gambar dari tiap kelas.
fig, axes = plt.subplots(2, 5, figsize=(16, 7))
for ax, subdir in zip(axes.ravel(), sorted(os.listdir(DATA_DIR))):
    subdir_path = os.path.join(DATA_DIR, subdir)
    if not os.path.isdir(subdir_path):
        continue
    sample = os.path.join(subdir_path, os.listdir(subdir_path)[0])
    ax.imshow(Image.open(sample))
    ax.set_title(f"{IT_TO_EN.get(subdir, subdir)}\n({subdir})")
    ax.axis("off")
plt.suptitle("Contoh gambar tiap kelas (resolusi asli berbeda-beda)", fontsize=14)
plt.tight_layout()
plt.show()

# %% [markdown]
# ## 3. Membagi Dataset: Train / Validation / Test (80 / 10 / 10)
#
# `image_dataset_from_directory` hanya mendukung pembagian dua arah (`validation_split`).
# Karena itu kita ambil 20% sebagai himpunan validasi penuh, lalu **membaginya lagi**
# menjadi *validation* dan *test* menggunakan `take`/`skip` — pola resmi dari tutorial
# *transfer learning* TensorFlow. Hasilnya: **train 80% / validation 10% / test 10%**.

# %%
IMG_SIZE = (224, 224)   # ukuran input MobileNetV2
BATCH = 32

train_ds = tf.keras.utils.image_dataset_from_directory(
    DATA_DIR, validation_split=0.2, subset="training",
    seed=SEED, image_size=IMG_SIZE, batch_size=BATCH, label_mode="int",
)
val_full = tf.keras.utils.image_dataset_from_directory(
    DATA_DIR, validation_split=0.2, subset="validation",
    seed=SEED, image_size=IMG_SIZE, batch_size=BATCH, label_mode="int",
)

# Nama kelas mengikuti urutan alfabetis folder (Italia) -> indeks 0..9.
class_names = train_ds.class_names
en_class_names = [IT_TO_EN[c] for c in class_names]
num_classes = len(class_names)
assert num_classes == 10, f"Diharapkan 10 kelas, ditemukan {num_classes}: {class_names}"
print("Indeks kelas:", dict(enumerate(en_class_names)))

# Bagi himpunan validasi penuh menjadi validation (50%) dan test (50%).
val_batches = val_full.cardinality().numpy()
test_ds = val_full.take(val_batches // 2)
val_ds = val_full.skip(val_batches // 2)

# Optimasi pipeline. Validation/test di-cache (kecil) agar evaluasi deterministik & cepat.
AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.prefetch(AUTOTUNE)               # train: decode ulang tiap epoch (hemat RAM)
val_ds = val_ds.cache().prefetch(AUTOTUNE)
test_ds = test_ds.cache().prefetch(AUTOTUNE)

print(f"Batch  -> train: {train_ds.cardinality().numpy()}, "
      f"val: {val_ds.cardinality().numpy()}, test: {test_ds.cardinality().numpy()}")

# %% [markdown]
# ## 4. Arsitektur Model (Sequential + Conv2D + Pooling)
#
# Model `Sequential` terdiri atas:
# 1. **Lapisan augmentasi** (`RandomFlip`, `RandomRotation`, `RandomZoom`) — hanya aktif saat
#    *training* (otomatis non-aktif saat evaluasi/inferensi), sehingga augmentasi hanya
#    diterapkan pada data latih.
# 2. **`Rescaling`** ke rentang `[-1, 1]` sesuai kebutuhan MobileNetV2 (lebih bersih untuk
#    diekspor dibanding `Lambda preprocess_input`).
# 3. **Basis MobileNetV2** (`include_top=False`, bobot ImageNet) untuk *transfer learning*.
# 4. **Lapisan `Conv2D` + `MaxPooling2D` buatan sendiri** — memenuhi Kriteria 4 secara eksplisit.
# 5. Kepala klasifikasi `Dense` + `Dropout` + `softmax`.

# %%
base_model = tf.keras.applications.MobileNetV2(
    input_shape=IMG_SIZE + (3,), include_top=False, weights="imagenet"
)
base_model.trainable = False  # tahap awal: bekukan basis

data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.15),
    layers.RandomZoom(0.15),
], name="data_augmentation")

model = tf.keras.Sequential([
    tf.keras.Input(shape=IMG_SIZE + (3,)),
    data_augmentation,                                   # augmentasi (train-only)
    layers.Rescaling(1.0 / 127.5, offset=-1),            # preprocessing MobileNetV2 [-1, 1]
    base_model,                                          # transfer learning
    layers.Conv2D(256, 3, padding="same", activation="relu"),   # Conv2D eksplisit
    layers.MaxPooling2D(),                                       # Pooling eksplisit
    layers.Conv2D(128, 3, padding="same", activation="relu"),
    layers.GlobalAveragePooling2D(),
    layers.Dropout(0.3),
    layers.Dense(128, activation="relu"),
    layers.Dropout(0.3),
    layers.Dense(num_classes, activation="softmax"),
], name="animals10_classifier")

model.summary()

# %% [markdown]
# ## 5. Callback
#
# Kita memakai empat callback (memenuhi saran "mengimplementasikan callback"), termasuk satu
# **callback kustom** yang menghentikan pelatihan saat akurasi validasi mencapai target.

# %%
class StopAtAccuracy(tf.keras.callbacks.Callback):
    """Hentikan pelatihan ketika val_accuracy mencapai ambang target."""

    def __init__(self, target=0.96):
        super().__init__()
        self.target = target

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        if logs.get("val_accuracy", 0.0) >= self.target:
            print(f"\n✅ val_accuracy {logs['val_accuracy']:.4f} ≥ {self.target} — hentikan pelatihan.")
            self.model.stop_training = True


callbacks = [
    tf.keras.callbacks.EarlyStopping(
        monitor="val_accuracy", patience=5, restore_best_weights=True, verbose=1),
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss", factor=0.2, patience=3, min_lr=1e-6, verbose=1),
    tf.keras.callbacks.ModelCheckpoint(
        "best_model.keras", monitor="val_accuracy", save_best_only=True, verbose=0),
    StopAtAccuracy(target=0.97),
]

# %% [markdown]
# ## 6. Pelatihan Dua Tahap
#
# **Tahap 1 — *feature extraction*:** basis MobileNetV2 dibekukan, hanya kepala model yang dilatih.
#
# **Tahap 2 — *fine-tuning*:** sebagian lapisan atas basis di-*unfreeze* dengan *learning rate*
# kecil untuk mendorong akurasi melewati 95%. Lapisan `BatchNormalization` tetap dibekukan agar
# statistik berjalannya tidak rusak.

# %%
# --- Tahap 1: feature extraction ---
EPOCHS_HEAD = 12
model.compile(optimizer=tf.keras.optimizers.Adam(1e-3),
              loss="sparse_categorical_crossentropy", metrics=["accuracy"])
history_head = model.fit(train_ds, validation_data=val_ds,
                         epochs=EPOCHS_HEAD, callbacks=callbacks)

# %%
# --- Tahap 2: fine-tuning ---
base_model.trainable = True
FINE_TUNE_AT = len(base_model.layers) - 40            # buka ~40 lapisan teratas
for layer in base_model.layers[:FINE_TUNE_AT]:
    layer.trainable = False
for layer in base_model.layers:                        # BatchNorm tetap beku
    if isinstance(layer, layers.BatchNormalization):
        layer.trainable = False

EPOCHS_FINE = 12
model.compile(optimizer=tf.keras.optimizers.Adam(1e-5),  # LR kecil untuk fine-tuning
              loss="sparse_categorical_crossentropy", metrics=["accuracy"])
history_fine = model.fit(train_ds, validation_data=val_ds,
                         epochs=EPOCHS_FINE, callbacks=callbacks)

# %%
# Gabungkan riwayat kedua tahap untuk pembuatan plot.
def merge_history(h1, h2):
    out = {}
    for k in h1.history:
        out[k] = h1.history[k] + h2.history.get(k, [])
    return out

history = merge_history(history_head, history_fine)
phase1_epochs = len(history_head.history["accuracy"])

# %% [markdown]
# ## 7. Evaluasi Model
#
# Kita mengukur akurasi pada **training set** dan **test set** (keduanya wajib ≥ 85%, target ≥ 95%),
# lalu menampilkan *classification report* dan *confusion matrix*.

# %%
train_loss, train_acc = model.evaluate(train_ds, verbose=0)
test_loss, test_acc = model.evaluate(test_ds, verbose=0)
print(f"Akurasi Training : {train_acc:.4f}")
print(f"Akurasi Test     : {test_acc:.4f}")
assert train_acc >= 0.85 and test_acc >= 0.85, "Akurasi di bawah ambang wajib 85%!"
print("\n✅ Memenuhi syarat wajib ≥85%." +
      ("  ✅ Memenuhi saran ≥95%." if train_acc >= 0.95 and test_acc >= 0.95 else
       "  ⚠️ Belum mencapai 95% — tambah epoch / lapisan fine-tuning."))

# %%
from sklearn.metrics import classification_report, confusion_matrix

# Kumpulkan label asli & prediksi dalam SATU lintasan agar urutan konsisten.
y_true, y_pred = [], []
for images, labels in test_ds:
    probs = model.predict(images, verbose=0)
    y_true.extend(labels.numpy())
    y_pred.extend(np.argmax(probs, axis=1))

print(classification_report(y_true, y_pred, target_names=en_class_names))

cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(9, 7))
plt.imshow(cm, cmap="Blues")
plt.title("Confusion Matrix (Test Set)")
plt.colorbar()
ticks = np.arange(num_classes)
plt.xticks(ticks, en_class_names, rotation=45, ha="right")
plt.yticks(ticks, en_class_names)
for i in range(num_classes):
    for j in range(num_classes):
        plt.text(j, i, cm[i, j], ha="center", va="center",
                 color="white" if cm[i, j] > cm.max() / 2 else "black", fontsize=8)
plt.ylabel("Label Asli")
plt.xlabel("Prediksi")
plt.tight_layout()
plt.show()

# %% [markdown]
# ## 8. Visualisasi Akurasi & Loss
#
# Plot di bawah memperlihatkan perkembangan akurasi dan loss (train vs validation) sepanjang
# pelatihan. Garis putus-putus menandai transisi dari Tahap 1 (*feature extraction*) ke
# Tahap 2 (*fine-tuning*).

# %%
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

ax1.plot(history["accuracy"], label="Train")
ax1.plot(history["val_accuracy"], label="Validation")
ax1.axvline(phase1_epochs - 1, color="gray", ls="--", label="Mulai fine-tuning")
ax1.axhline(0.95, color="green", ls=":", alpha=0.6, label="Target 95%")
ax1.set_title("Akurasi Model"); ax1.set_xlabel("Epoch"); ax1.set_ylabel("Akurasi"); ax1.legend()

ax2.plot(history["loss"], label="Train")
ax2.plot(history["val_loss"], label="Validation")
ax2.axvline(phase1_epochs - 1, color="gray", ls="--", label="Mulai fine-tuning")
ax2.set_title("Loss Model"); ax2.set_xlabel("Epoch"); ax2.set_ylabel("Loss"); ax2.legend()

plt.tight_layout()
plt.show()

# %% [markdown]
# ## 9. Ekspor — SavedModel
#
# Pada Keras 3 (default Colab), `model.save()` menulis format `.keras`. Untuk menghasilkan
# **direktori SavedModel** (`saved_model.pb` + `variables/`) digunakan `model.export()`.
#
# ⚠️ **Penting:** lapisan augmentasi (`RandomRotation` / `RandomZoom`) memakai op
# `ImageProjectiveTransformV3` yang **tidak didukung TF-Lite**, sehingga konversi `.tflite`
# akan gagal bila lapisan ini ikut diekspor. Karena augmentasi hanya diperlukan saat *training*
# (bukan saat deployment), kita ekspor **model inferensi tanpa lapisan augmentasi** —
# bobot hasil pelatihan tetap dipakai, tanpa perlu melatih ulang.

# %%
SAVED_MODEL_DIR = "saved_model"

# Buang lapisan augmentasi (indeks 0); lapisan lain memakai bobot hasil pelatihan.
inference_model = tf.keras.Sequential(
    [tf.keras.Input(shape=IMG_SIZE + (3,))] + model.layers[1:],
    name="animals10_inference",
)
inference_model.export(SAVED_MODEL_DIR)
print("Lapisan model inferensi :", [l.name for l in inference_model.layers])
print("Isi SavedModel          :", os.listdir(SAVED_MODEL_DIR))

# %% [markdown]
# ## 10. Ekspor — TF-Lite
#
# Konversi dilakukan dari direktori SavedModel (jalur paling andal di Keras 3). Kita juga
# menulis `label.txt` sesuai urutan indeks kelas model.

# %%
os.makedirs("tflite", exist_ok=True)

converter = tf.lite.TFLiteConverter.from_saved_model(SAVED_MODEL_DIR)
converter.optimizations = [tf.lite.Optimize.DEFAULT]   # kuantisasi bobot -> berkas lebih kecil
tflite_model = converter.convert()

with open("tflite/model.tflite", "wb") as f:
    f.write(tflite_model)

# label.txt: satu nama kelas (Inggris) per baris, sesuai urutan indeks 0..9.
with open("tflite/label.txt", "w") as f:
    f.write("\n".join(en_class_names))

print("Ukuran model.tflite : %.2f MB" % (os.path.getsize("tflite/model.tflite") / 1e6))
print("label.txt           :", en_class_names)

# %% [markdown]
# ## 11. Inferensi (Bukti Menggunakan Model TF-Lite)
#
# Sebagai bukti inferensi (salah satu saran penilaian), kita memuat **model TF-Lite** dan
# memprediksi beberapa gambar dari **test set**. Karena lapisan *rescaling* berada di dalam
# model, masukan TF-Lite adalah piksel mentah rentang `[0, 255]`.

# %%
interpreter = tf.lite.Interpreter(model_path="tflite/model.tflite")
interpreter.allocate_tensors()
in_idx = interpreter.get_input_details()[0]["index"]
out_idx = interpreter.get_output_details()[0]["index"]


def tflite_predict(image):
    """Prediksi satu gambar (H,W,3) float32 -> (indeks_kelas, confidence)."""
    x = np.expand_dims(image.astype(np.float32), axis=0)
    interpreter.set_tensor(in_idx, x)
    interpreter.invoke()
    probs = interpreter.get_tensor(out_idx)[0]
    return int(np.argmax(probs)), float(np.max(probs))


# Ambil satu batch dari test set lalu tampilkan 9 prediksi.
images, labels = next(iter(test_ds))
plt.figure(figsize=(12, 12))
for i in range(9):
    pred_idx, conf = tflite_predict(images[i].numpy())
    true_idx = int(labels[i].numpy())
    correct = pred_idx == true_idx
    ax = plt.subplot(3, 3, i + 1)
    ax.imshow(images[i].numpy().astype("uint8"))
    ax.set_title(
        f"Prediksi: {en_class_names[pred_idx]} ({conf:.0%})\nAsli: {en_class_names[true_idx]}",
        color="green" if correct else "red", fontsize=11)
    ax.axis("off")
plt.suptitle("Bukti Inferensi Model TF-Lite (hijau = benar, merah = salah)", fontsize=14)
plt.tight_layout()
plt.show()

# %% [markdown]
# ## 12. Ekspor — TensorFlow.js
#
# `tensorflowjs` mem-*pin* versi TF/jax/tf-keras lama yang dapat mengganggu TF bawaan Colab.
# Mitigasi: jalankan langkah ini **paling akhir** dan instal dengan `--no-deps`. Konversi
# membaca direktori `saved_model/` di disk sehingga tetap aman walau runtime perlu di-restart.

# %%
# Instalasi terisolasi untuk menghindari konflik dependensi dengan TF bawaan Colab.
!pip install -q tensorflowjs --no-deps
!pip install -q tensorflow-hub

# %%
# Konversi SavedModel -> TensorFlow.js graph model.
!tensorflowjs_converter \
    --input_format=tf_saved_model \
    --output_format=tfjs_graph_model \
    --signature_name=serving_default \
    --saved_model_tags=serve \
    saved_model tfjs_model

print("Isi tfjs_model:", os.listdir("tfjs_model"))

# %% [markdown]
# ## 13. requirements.txt
#
# Hasilkan daftar dependensi dari lingkungan eksekusi. (Sebagai alternatif yang lebih ringkas,
# dapat digunakan `pipreqs` yang hanya mencantumkan paket yang benar-benar di-*import*.)

# %%
!pip freeze > requirements.txt
print("requirements.txt dibuat. Cuplikan paket inti:")
!grep -Ei "^(tensorflow|tensorflowjs|keras|numpy|matplotlib|pillow|scikit-learn|kaggle)" requirements.txt

# %% [markdown]
# ## Ringkasan
#
# - Model `Sequential` (MobileNetV2 + `Conv2D`/`MaxPooling2D` + kepala `Dense`) dilatih dua tahap.
# - Akurasi training & test ditampilkan pada sel Evaluasi (target ≥ 95%).
# - Plot akurasi & loss, *confusion matrix*, dan **bukti inferensi TF-Lite** tersedia di atas.
# - Model diekspor ke **SavedModel** (`saved_model/`), **TF-Lite** (`tflite/`), dan
#   **TensorFlow.js** (`tfjs_model/`).
#
# **Langkah berikutnya (lihat README / INSTRUCTIONS):** unduh `notebook.ipynb` (sudah berisi
# output) **dan** `notebook.py` dari Colab, lalu susun semuanya ke dalam folder `submission/`
# dan kompres menjadi `.zip` sebelum dikirim ke Dicoding.
