import streamlit as st

import ui
from common import MODEL_PATH, get_predictor, model_missing_error

MAX_CHARS = 500

st.set_page_config(page_title="Deteksi Ujaran Kebencian di Aplikasi X", page_icon="🛡️", layout="centered",
                   initial_sidebar_state="collapsed")
st.markdown(ui.CSS + """
<style>
div[data-testid="stVerticalBlockBorderWrapper"]{border-radius:16px;border:1px solid #e6e6f4;box-shadow:0 2px 10px rgba(79,70,229,.06);padding:6px 10px}
div[data-testid="stTextArea"] textarea{background:#fff}
</style>""", unsafe_allow_html=True)

st.markdown(ui.hero_html(), unsafe_allow_html=True)

if not MODEL_PATH.exists():
    model_missing_error()
    st.stop()
predictor = get_predictor()


def _clear():
    st.session_state["input_text"] = ""
    st.session_state.pop("hasil", None)


# ---------------------------------------------------------------- input
with st.container(border=True):
    st.markdown(ui.input_head_html(), unsafe_allow_html=True)
    c_text, c_btn = st.columns([4.2, 1.3], vertical_alignment="center")
    with c_text:
        text = st.text_area("Teks", key="input_text", height=130, max_chars=MAX_CHARS, label_visibility="collapsed",
                            placeholder="Tempel cuitan / komentar di sini...")
        st.button("🗑  Bersihkan", key="bersihkan", on_click=_clear)
    with c_btn:
        go = st.button("🔍  Deteksi", type="primary", key="deteksi", use_container_width=True)
    st.markdown(ui.model_note_html(), unsafe_allow_html=True)

if go:
    if not text.strip():
        st.session_state.pop("hasil", None)
        st.info("Masukkan cuitan terlebih dahulu.")
    else:
        out = predictor.predict([text])
        st.session_state["hasil"] = {
            "empty": bool(out["empty"][0]), "P": out["P"][0], "flags": out["flags"][0],
            "truncated": bool(out["truncated"][0]), "exp": None if out["empty"][0] else predictor.explain(text),
        }

# ---------------------------------------------------------------- hasil
h = st.session_state.get("hasil")
if h is not None:
    if h["empty"]:
        st.info("Cuitan tidak mengandung kata yang bisa dianalisis (mungkin hanya emoji, tautan, atau mention).")
    else:
        n = int(h["flags"].sum())
        st.markdown(
            '<div class="han-card">' + ui.result_head_html(True) + ui.banner_html(n)
            + ui.cards_html(h["P"], h["flags"], predictor.labels, predictor.thresholds) + "</div>",
            unsafe_allow_html=True)
        if h["truncated"]:
            st.caption(f"Cuitan panjang dipotong: maksimal {predictor.max_sent} kalimat dan "
                       f"{predictor.max_words} kata per kalimat yang dianalisis.")
        st.markdown(ui.attention_card_html(h["exp"]), unsafe_allow_html=True)
        st.markdown(ui.wise_html(), unsafe_allow_html=True)
else:
    st.markdown('<div class="han-card">' + ui.result_head_html(False)
                + '<div class="han-empty">Tulis cuitan di atas lalu tekan <b>Deteksi</b> untuk melihat hasil analisis.</div></div>',
                unsafe_allow_html=True)

st.markdown(ui.disclaimer_html() + ui.footer_html(), unsafe_allow_html=True)
