import io
import json
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

from han_inference import HANPredictor, attention_html

st.set_page_config(page_title="Deteksi Hate Tweet (Hierarchical BiLSTM-Attention)", page_icon="🛡️", layout="wide")

BASE = Path(__file__).resolve().parent
MODEL_PATH = BASE / "models" / "han_model.pt"
METRICS_PATH = BASE / "models" / "metrics_han.json"
ABLATION_PATH = BASE / "data" / "ablasi_cv_hasil.csv"
MAX_BATCH_ROWS = 2000
MAX_CHARS = 2000

EXAMPLES = {
    "Netral": "Selamat pagi semuanya, semoga harimu menyenangkan.",
    "Fisik": "Muka lo jelek banget kayak monyet. Badan gendut pula, ngaca dulu.",
    "Agama": "Dasar kafir bejat! Otak lu isinya sampah semua. Pergi sana.",
    "Kelompok": "Kaum itu semuanya sama, penjilat dan tukang bohong. Usir saja dari negeri ini.",
}


@st.cache_resource(show_spinner="Memuat model...")
def get_predictor():
    return HANPredictor(MODEL_PATH)


@st.cache_data
def get_metrics():
    return json.loads(METRICS_PATH.read_text()) if METRICS_PATH.exists() else None


@st.cache_data
def get_ablation():
    return pd.read_csv(ABLATION_PATH) if ABLATION_PATH.exists() else None


st.title("🛡️ Deteksi Hate Tweet Bahasa Indonesia")
st.caption("Model: **Hierarchical BiLSTM-Attention (word → sentence)** · multi-label: hate tweet individu, grup, fisik, agama")
st.warning(
    "Model statistik ini bisa salah, terutama untuk label **fisik** (F1 sekitar 0,5). Gunakan sebagai alat bantu, "
    "bukan satu-satunya dasar keputusan moderasi; libatkan peninjauan manusia.",
    icon="⚠️",
)

if not MODEL_PATH.exists():
    st.error("File model `models/han_model.pt` tidak ditemukan. Jalankan notebook training sampai bagian 16, "
             "lalu letakkan `han_model.pt` (dan `metrics_han.json`) di folder `models/`.")
    st.stop()

predictor = get_predictor()
tab1, tab2, tab3 = st.tabs(["🔎 Analisis Teks", "📁 Analisis Batch (CSV)", "📊 Tentang Model & Evaluasi"])

# ------------------------------------------------------------------ TAB 1
with tab1:
    st.write("Pilih contoh atau tulis teks sendiri:")
    cols = st.columns(len(EXAMPLES))
    for col, (name, txt) in zip(cols, EXAMPLES.items()):
        if col.button(name, key=f"ex_{name}"):
            st.session_state["input_text"] = txt

    text = st.text_area("Teks yang akan dianalisis", key="input_text", height=130, max_chars=MAX_CHARS,
                        placeholder="Tempel tweet / komentar di sini...")

    if st.button("Analisis", type="primary", key="analisis"):
        if not text.strip():
            st.info("Masukkan teks terlebih dahulu.")
        else:
            out = predictor.predict([text])
            if out["empty"][0]:
                st.info("Teks tidak mengandung kata yang bisa dianalisis (mungkin hanya emoji, tautan, atau mention).")
            else:
                P, flags = out["P"][0], out["flags"][0]
                detected = [predictor.display[l] for l, f in zip(predictor.labels, flags) if f]
                if detected:
                    st.error("**Terdeteksi:** " + ", ".join(detected))
                else:
                    st.success("**Tidak terdeteksi hate tweet.**")
                if out["truncated"][0]:
                    st.caption(f"Teks panjang dipotong: maksimal {predictor.max_sent} kalimat dan "
                               f"{predictor.max_words} kata per kalimat yang dianalisis.")

                rows = [{"Label": predictor.display[l], "Probabilitas": f"{p:.1%}",
                         "Threshold": f"{predictor.thresholds[l]:.2f}", "Terdeteksi": "ya" if f else "tidak"}
                        for l, p, f in zip(predictor.labels, P, flags)]
                st.dataframe(pd.DataFrame(rows), hide_index=True)
                st.caption("Satu teks bisa memiliki lebih dari satu label. Threshold tiap label dituning pada data validasi.")

                if "HS_Physical" in predictor.labels and flags[predictor.labels.index("HS_Physical")]:
                    st.warning("Label **fisik** paling kurang andal (F1 ±0,5): anotasi data latih untuk label ini tidak konsisten.")

                st.subheader("Perhatian (attention) model")
                exp = predictor.explain(text)
                st.markdown(attention_html(exp), unsafe_allow_html=True)
                st.caption("Warna lebih pekat = kata/kalimat yang lebih diperhatikan model (bobot attention kata × kalimat, "
                           "dari model pertama dalam ensemble). Attention membantu interpretasi, tetapi bukan bukti sebab-akibat.")

# ------------------------------------------------------------------ TAB 2
with tab2:
    st.write(f"Unggah CSV berisi kolom teks (maksimal {MAX_BATCH_ROWS:,} baris diproses).")
    up = st.file_uploader("File CSV", type=["csv"], key="batch_file")
    if up is not None:
        raw_bytes = up.getvalue()
        df_in = None
        try:
            df_in = pd.read_csv(io.BytesIO(raw_bytes))
        except UnicodeDecodeError:
            df_in = pd.read_csv(io.BytesIO(raw_bytes), encoding="latin-1")
        except Exception as e:
            st.error(f"Gagal membaca CSV: {e}")

        if df_in is not None and len(df_in):
            default_idx = list(df_in.columns).index("Tweet") if "Tweet" in df_in.columns else 0
            col_text = st.selectbox("Kolom teks", list(df_in.columns), index=default_idx)
            if len(df_in) > MAX_BATCH_ROWS:
                st.warning(f"File punya {len(df_in):,} baris; hanya {MAX_BATCH_ROWS:,} baris pertama diproses.")
                df_in = df_in.head(MAX_BATCH_ROWS)

            if st.button("Proses file", type="primary", key="proses"):
                texts = df_in[col_text].fillna("").astype(str).tolist()
                P_all, F_all, preds, empties = [], [], [], []
                bar = st.progress(0.0)
                for i in range(0, len(texts), 64):
                    o = predictor.predict(texts[i:i + 64])
                    P_all.append(o["P"]); F_all.append(o["flags"]); preds += o["prediksi"]; empties.append(o["empty"])
                    bar.progress(min(1.0, (i + 64) / len(texts)))
                P_all, F_all, empties = np.vstack(P_all), np.vstack(F_all), np.concatenate(empties)

                out_df = df_in.copy()
                for j, l in enumerate(predictor.labels):
                    out_df["p_" + l] = P_all[:, j].round(3)
                    out_df[l + "_terdeteksi"] = F_all[:, j]
                out_df["prediksi"] = preds

                c1, c2 = st.columns(2)
                c1.metric("Jumlah teks", f"{len(out_df):,}")
                c2.metric("Terdeteksi hate tweet (≥1 label)", f"{int(F_all.any(axis=1).sum()):,}")
                counts = {predictor.display[l]: int(F_all[:, j].sum()) for j, l in enumerate(predictor.labels)}
                st.bar_chart(pd.Series(counts, name="Jumlah"))
                if empties.any():
                    st.caption(f"{int(empties.sum())} teks tidak dapat dianalisis (kosong setelah dibersihkan).")
                st.dataframe(out_df.head(200))
                st.download_button("Unduh hasil (CSV)", out_df.to_csv(index=False).encode("utf-8"),
                                   file_name="hasil_deteksi_hate_tweet.csv", mime="text/csv")

# ------------------------------------------------------------------ TAB 3
with tab3:
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
