import io
import json

import numpy as np
import pandas as pd
import streamlit as st

from common import MODEL_PATH, get_ablation, get_metrics, get_predictor, model_missing_error

st.set_page_config(page_title="Analisis Batch", page_icon="🛡️", layout="wide")
st.title("📁 Analisis Batch (CSV)")
if not MODEL_PATH.exists():
    model_missing_error()
    st.stop()
predictor = get_predictor()
MAX_BATCH_ROWS = 2000
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

