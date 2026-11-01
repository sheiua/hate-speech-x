"""Bagian yang dipakai bersama oleh halaman utama dan halaman tambahan."""
import json
from pathlib import Path

import pandas as pd
import streamlit as st

from han_inference import HANPredictor

BASE = Path(__file__).resolve().parent
MODEL_PATH = BASE / "models" / "han_model.pt"
METRICS_PATH = BASE / "models" / "metrics_han.json"
ABLATION_PATH = BASE / "data" / "ablasi_cv_hasil.csv"


@st.cache_resource(show_spinner="Memuat model...")
def get_predictor():
    return HANPredictor(MODEL_PATH)


@st.cache_data
def get_metrics():
    return json.loads(METRICS_PATH.read_text()) if METRICS_PATH.exists() else None


@st.cache_data
def get_ablation():
    return pd.read_csv(ABLATION_PATH) if ABLATION_PATH.exists() else None


def model_missing_error():
    st.error("File model `models/han_model.pt` tidak ditemukan. Letakkan `han_model.pt` "
             "(dan `metrics_han.json`) di folder `models/`.")
