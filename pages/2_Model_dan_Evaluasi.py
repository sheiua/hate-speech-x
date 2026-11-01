import io
import json

import numpy as np
import pandas as pd
import streamlit as st

from common import MODEL_PATH, get_ablation, get_metrics, get_predictor, model_missing_error

st.set_page_config(page_title="Model dan Evaluasi", page_icon="🛡️", layout="wide")
st.title("📊 Tentang Model & Evaluasi")
if not MODEL_PATH.exists():
    model_missing_error()
    st.stop()
predictor = get_predictor()
st.subheader("Arsitektur")
st.markdown(
    "Tweet dipecah menjadi kalimat; tiap kalimat dienkode oleh **BiLSTM level kata + attention kata** menjadi vektor kalimat; "
    "seluruh vektor kalimat dienkode oleh **BiLSTM level kalimat + attention kalimat** menjadi vektor tweet, "
    "lalu 4 keluaran sigmoid (multi-label). Embedding kata: fastText Indonesia (dibekukan). "
    f"Prediksi akhir = rata-rata **{predictor.meta['n_models']} model** (seed berbeda); "
    "threshold tiap label dituning pada data validasi."
)
m = get_metrics()
if m is None:
    st.info("`models/metrics_han.json` tidak ditemukan.")
else:
    st.subheader("Hasil pada data uji (hold-out)")
    per = pd.DataFrame(m["test_per_label"]).T
    keep = [c for c in ["precision", "recall", "f1", "support", "pr_auc", "threshold"] if c in per.columns]
    per = per[keep].astype(float).round(3)
    per.index = [predictor.display.get(i, i) for i in per.index]
    st.dataframe(per)
    tm = pd.DataFrame({"semua tweet": m["test_metrics"], "hanya tweet berlabel": m["test_metrics_hanya_berlabel"]}).round(3)
    st.dataframe(tm)
    st.caption("Data uji hanya punya ±46 sampel untuk label fisik, sehingga F1 label itu bisa bergeser ±0,1 pada pembagian data lain.")

ab = get_ablation()
if ab is not None:
    st.subheader("Perbandingan model (CV 5-fold)")
    cols_m = ["F1_Individual", "F1_Group", "F1_Physical", "F1_Religion", "macro_f1", "subset_acc"]
    fm = ab.groupby(["variant", "fold"])[cols_m].mean()
    mean, std = fm.groupby("variant").mean(), fm.groupby("variant").std(ddof=1)
    txt = mean.round(3).astype(str) + " ± " + std.round(3).astype(str)
    st.dataframe(txt)
    st.caption("Rata-rata ± simpangan baku antar-fold (seed dirata-ratakan dalam fold). HAN setara dengan BiLSTM+attention flat "
               "dan sedikit di bawah baseline TF-IDF + Logistic Regression pada data ini.")

st.subheader("Keterbatasan")
st.markdown(
    "- Dilatih pada satu dataset tweet Indonesia; kinerja pada platform/topik/periode lain belum diukur.\n"
    "- Label **fisik** paling lemah (kelas kecil + anotasi tidak konsisten pada data latih).\n"
    "- Hasil berupa probabilitas yang dibandingkan dengan threshold; bukan keputusan moderasi final.\n"
    "- Kode aplikasi ini tidak menyimpan teks yang Anda masukkan."
)
st.caption(f"Vocab {predictor.meta['vocab_size']:,} kata · dim embedding {predictor.meta['emb_dim']} · "
           f"torch (training) {predictor.meta['torch_version_training']}")
