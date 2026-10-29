# Verifikasi di Colab SEBELUM deploy: modul inferensi aplikasi harus memberi prediksi yang sama dengan notebook.
# Langkah: unggah han_inference.py ke /content (panel file di kiri), lalu jalankan sel ini.
# Ganti MODEL_PATH ke "han_model_slim.pt" bila memakai versi ramping.
import numpy as np
from han_inference import HANPredictor

MODEL_PATH = "han_model.pt"
pred = HANPredictor(MODEL_PATH)
print("Model dimuat:", pred.meta, "| mode segmentasi:", pred.segment_mode, "| threshold:", pred.thresholds)

contoh = [
    "Selamat pagi semuanya, semoga harimu menyenangkan.",
    "Dasar kafir bejat! Otak lu isinya sampah semua. Pergi sana.",
    "Muka lo jelek banget kayak monyet. Badan gendut pula, ngaca dulu.",
    "Kaum itu semuanya sama, penjilat dan tukang bohong. Usir saja dari negeri ini.",
    "😂😂 https://t.co/abc",
]
out = pred.predict(contoh)
for t, p, lab in zip(contoh, out["P"], out["prediksi"]):
    print(f"{t[:50]!r:55s} -> {lab}   P={np.round(p, 3)}")

# Bandingkan dengan notebook (butuh sesi yang masih punya predict_texts dari bagian 14 & final_models dari bagian 11).
try:
    ref = predict_texts(contoh[:4])
    ref_P = ref[["p_" + l for l in LABELS]].values
    diff = np.abs(ref_P - out["P"][:4]).max()
    print("\nSelisih maksimum probabilitas vs notebook:", round(float(diff), 4))
    print("OK (selisih <= 0,002 karena pembulatan 3 desimal di notebook)" if diff <= 0.002 else
          "PERHATIAN: selisih besar -> jangan deploy dulu, kirim output ini ke saya")
except NameError:
    print("\npredict_texts (bagian 14) belum tersedia di sesi ini; perbandingan dengan notebook dilewati.")
