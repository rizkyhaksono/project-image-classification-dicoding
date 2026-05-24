#!/usr/bin/env bash
# Validasi kelengkapan submission lalu buat submission.zip sesuai ketentuan Dicoding.
set -euo pipefail
cd "$(dirname "$0")"

SUB="submission"
ZIP="submission.zip"
ok=1

check() {  # check <path> <deskripsi>
  if [ -e "$SUB/$1" ]; then
    echo "  ✅ $1"
  else
    echo "  ❌ HILANG: $1  ($2)"
    ok=0
  fi
}

echo "== Memeriksa berkas wajib di $SUB/ =="
check "notebook.ipynb"            "notebook sudah dijalankan (berisi output)"
check "notebook.py"               "ekspor .py (wajib bersama .ipynb)"
check "README.md"                 "dokumentasi"
check "requirements.txt"          "dependensi"
check "saved_model/saved_model.pb" "format SavedModel"
check "saved_model/variables"      "bobot SavedModel"
check "tflite/model.tflite"        "format TF-Lite"
check "tflite/label.txt"           "label TF-Lite"
check "tfjs_model/model.json"      "format TFJS"
# Minimal satu shard .bin untuk TFJS
if ls "$SUB"/tfjs_model/group1-shard*.bin >/dev/null 2>&1; then
  echo "  ✅ tfjs_model/group1-shard*.bin"
else
  echo "  ❌ HILANG: tfjs_model/group1-shard*.bin  (bobot TFJS)"
  ok=0
fi

# Cek notebook benar-benar punya output sel
if [ -e "$SUB/notebook.ipynb" ]; then
  if python3 -c "import json,sys; nb=json.load(open('$SUB/notebook.ipynb')); sys.exit(0 if any(c.get('outputs') for c in nb['cells'] if c['cell_type']=='code') else 1)"; then
    echo "  ✅ notebook.ipynb berisi output sel"
  else
    echo "  ❌ notebook.ipynb TIDAK punya output — jalankan dulu di Colab lalu unduh ulang"
    ok=0
  fi
fi

if [ "$ok" -ne 1 ]; then
  echo
  echo "⛔ Submission belum lengkap. Lengkapi berkas yang hilang lalu jalankan lagi skrip ini."
  exit 1
fi

echo
echo "== Membersihkan placeholder & cache =="
find "$SUB" -name ".gitkeep" -delete -print | sed 's/^/  hapus /' || true
find "$SUB" -name "__pycache__" -type d -prune -exec rm -rf {} + 2>/dev/null || true
find "$SUB" -name ".ipynb_checkpoints" -type d -prune -exec rm -rf {} + 2>/dev/null || true

echo
echo "== Membuat $ZIP =="
rm -f "$ZIP"
zip -r "$ZIP" "$SUB" -x '*.gitkeep' '*/__pycache__/*' '*/.ipynb_checkpoints/*' >/dev/null
echo "  dibuat: $(du -h "$ZIP" | cut -f1)  $ZIP"
echo
echo "== Isi $ZIP =="
unzip -l "$ZIP" | awk 'NR>3 && $4!="" {print "  "$4}' | head -60
echo
echo "✅ Selesai. Unggah '$ZIP' ke halaman submission Dicoding."
