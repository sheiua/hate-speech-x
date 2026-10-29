# Deteksi Hate Tweet Bahasa Indonesia: Hierarchical BiLSTM-Attention (word → sentence)

Aplikasi Streamlit untuk model multi-label (hate tweet individu, grup, fisik, agama).

## Isi
| File | Fungsi |
|---|---|
| `app.py` | aplikasi Streamlit (analisis teks + attention, batch CSV, tentang model) |
| `han_inference.py` | preprocessing + model + prediksi (disalin identik dari notebook training) |
| `models/han_model.pt`, `models/metrics_han.json` | **dari Anda**: hasil notebook bagian 16 |
| `data/ablasi_cv_hasil.csv` | hasil ablation CV (untuk halaman "Tentang Model") |
| `slim_model_colab.py` | (opsional) memperkecil `han_model.pt` ±66 MB → ±23 MB |
| `uji_han_inference_colab.py` | verifikasi di Colab bahwa prediksi aplikasi = notebook |

## Langkah
1. **Colab:** jalankan notebook sampai bagian 16 → dapat `han_model.pt` dan `metrics_han.json`.
2. **(Opsional) perkecil model:** jalankan `slim_model_colab.py` di Colab → `han_model_slim.pt`; ganti namanya menjadi `han_model.pt`.
3. **Verifikasi di Colab (disarankan):** unggah `han_inference.py` ke Colab, jalankan `uji_han_inference_colab.py`.
   Selisih probabilitas vs notebook harus ≤ 0,002.
4. Letakkan kedua file dari langkah 1 di folder `models/`.
5. **Uji lokal (opsional):** `pip install -r requirements.txt` lalu `streamlit run app.py`.
6. **GitHub:** unggah seluruh isi folder. Catatan ukuran file: unggah lewat web GitHub dibatasi ±25 MB per file, sedangkan `git push`
   menerima sampai 100 MB per file. Karena itu versi ramping (langkah 2) memudahkan; atau gunakan `git push` / Git LFS.
7. **Streamlit Community Cloud:** share.streamlit.io → *Create app* → pilih repo & branch → *Main file path*: `app.py` →
   *Advanced settings*: pilih Python 3.11/3.12 → *Deploy*.

## Catatan & pemecahan masalah
- `requirements.txt` memakai `torch` versi CPU lewat `--extra-index-url`. Jika build gagal atau kehabisan memori/waktu, kabari saya
  beserta pesan lognya; alternatifnya memperkecil model atau mengonversinya ke format yang lebih ringan.
- Model diimpor dengan `torch.load(weights_only=False)` karena file memuat kamus dan vocab. Muat hanya file model buatan Anda sendiri.
- Aplikasi tidak menyimpan teks masukan (kode tidak menulis ke disk). Infrastruktur hosting dapat memiliki log sendiri.
- Label **fisik** paling lemah (F1 ±0,5): anotasi data latih tidak konsisten.
