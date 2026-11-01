"""Pembangun HTML/CSS tampilan aplikasi (tanpa dependensi streamlit agar mudah diuji).
Semua string HTML ditulis tanpa indentasi/baris kosong agar tidak dianggap blok kode oleh Markdown."""
import base64
import html
from pathlib import Path

ASSETS = Path(__file__).resolve().parent / "assets"

# Urutan kartu sesuai desain: agama, fisik, individu, kelompok
CARD_ORDER = [
    ("HS_Religion", "AGAMA", "agama", "yang berkaitan dengan agama."),
    ("HS_Physical", "FISIK", "fisik", "yang berkaitan dengan fisik."),
    ("HS_Individual", "INDIVIDU", "individu", "yang menyerang individu secara personal."),
    ("HS_Group", "KELOMPOK", "kelompok", "yang menargetkan kelompok tertentu."),
]

ICONS = {
    "HS_Religion": '<svg viewBox="0 0 24 24" width="26" height="26" fill="currentColor"><path d="M12 2 9.5 5H7v3L3 11v2h1v8h6v-5h4v5h6v-8h1v-2l-4-3V5h-2.5z"/></svg>',
    "HS_Physical": '<svg viewBox="0 0 24 24" width="26" height="26" fill="currentColor"><circle cx="12" cy="7" r="4.2"/><path d="M4 21c0-4.4 3.6-7.5 8-7.5s8 3.1 8 7.5z"/></svg>',
    "HS_Individual": '<svg viewBox="0 0 24 24" width="26" height="26" fill="currentColor"><circle cx="12" cy="7" r="4.2"/><path d="M4 21c0-4.4 3.6-7.5 8-7.5s8 3.1 8 7.5z"/></svg>',
    "HS_Group": '<svg viewBox="0 0 24 24" width="28" height="26" fill="currentColor"><circle cx="12" cy="8" r="3.4"/><circle cx="5" cy="9.5" r="2.6"/><circle cx="19" cy="9.5" r="2.6"/><path d="M6 20c0-3.6 2.7-6 6-6s6 2.4 6 6z"/><path d="M0.5 19c0-2.8 2-4.7 4.5-4.7 1 0 1.8.2 2.5.7-1.4 1-2.3 2.5-2.5 4z"/><path d="M23.5 19c0-2.8-2-4.7-4.5-4.7-1 0-1.8.2-2.5.7 1.4 1 2.3 2.5 2.5 4z"/></svg>',
}

SVG_WARN = '<svg viewBox="0 0 48 48" width="52" height="52"><path d="M24 4 2 42h44z" fill="#dc2626" stroke="#dc2626" stroke-width="4" stroke-linejoin="round"/><rect x="21.5" y="16" width="5" height="14" rx="2.5" fill="#fff"/><circle cx="24" cy="35.5" r="3" fill="#fff"/></svg>'
SVG_SHIELD_RED = '<svg viewBox="0 0 40 46" width="42" height="48"><path d="M20 2 3 8v13c0 11 7.5 20 17 23 9.5-3 17-12 17-23V8z" fill="#dc2626"/><rect x="18" y="12" width="4" height="14" rx="2" fill="#fff"/><circle cx="20" cy="32" r="2.6" fill="#fff"/></svg>'
SVG_SHIELD_GREEN = '<svg viewBox="0 0 40 46" width="42" height="48"><path d="M20 2 3 8v13c0 11 7.5 20 17 23 9.5-3 17-12 17-23V8z" fill="#16a34a"/><path d="m12 23 6 6 11-12" fill="none" stroke="#fff" stroke-width="4" stroke-linecap="round" stroke-linejoin="round"/></svg>'
SVG_ALERT_OK = '<svg viewBox="0 0 48 48" width="52" height="52"><circle cx="24" cy="24" r="21" fill="#16a34a"/><path d="m14 25 7 7 14-15" fill="none" stroke="#fff" stroke-width="5" stroke-linecap="round" stroke-linejoin="round"/></svg>'
SVG_INFO = '<svg viewBox="0 0 24 24" width="18" height="18" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round"><circle cx="12" cy="12" r="9.5"/><path d="M12 11v6"/><circle cx="12" cy="7.5" r=".6" fill="currentColor"/></svg>'
SVG_EDIT = '<svg viewBox="0 0 24 24" width="24" height="24" fill="none" stroke="#4f46e5" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M12 4H5a2 2 0 0 0-2 2v13a2 2 0 0 0 2 2h13a2 2 0 0 0 2-2v-7"/><path d="M18.5 2.5a2.1 2.1 0 0 1 3 3L12 15l-4 1 1-4z"/></svg>'
SVG_BARS = '<svg viewBox="0 0 24 24" width="24" height="24" fill="#4f46e5"><rect x="3" y="12" width="4.5" height="9" rx="1"/><rect x="9.8" y="6" width="4.5" height="15" rx="1"/><rect x="16.5" y="9" width="4.5" height="12" rx="1"/></svg>'
SVG_EYE = '<svg viewBox="0 0 24 24" width="24" height="24" fill="none" stroke="#4f46e5" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M1.5 12S5.5 5 12 5s10.5 7 10.5 7-4 7-10.5 7S1.5 12 1.5 12z"/><circle cx="12" cy="12" r="3.2" fill="#4f46e5"/></svg>'
SVG_CHECK_SM = '<svg viewBox="0 0 24 24" width="20" height="20"><circle cx="12" cy="12" r="11" fill="#16a34a"/><path d="m6.5 12.5 3.8 3.8 7-7.6" fill="none" stroke="#fff" stroke-width="2.6" stroke-linecap="round" stroke-linejoin="round"/></svg>'


def img_uri(name):
    p = ASSETS / f"{name}.png"
    if not p.exists():
        return ""
    return "data:image/png;base64," + base64.b64encode(p.read_bytes()).decode()


def img_tag(name, width, cls=""):
    uri = img_uri(name)
    return f'<img src="{uri}" width="{width}" class="{cls}" alt="">' if uri else ""


CSS = """
<style>
:root{--indigo:#4f46e5;--indigo-d:#3730a3;--navy:#1e1b4b;--red:#dc2626;--red-bg:#fff1f1;--red-bd:#fecaca;--slate:#475569;--line:#e6e6f4;--muted:#64748b}
html,body,[data-testid="stAppViewContainer"],.stApp{background:#fff!important;color:var(--navy)}
#MainMenu,footer,[data-testid="stToolbar"],[data-testid="stDecoration"]{visibility:hidden;height:0}
header[data-testid="stHeader"]{background:transparent}
.block-container{max-width:1000px!important;padding:1.2rem 1rem 2rem!important}
.han-hero{display:flex;align-items:center;justify-content:space-between;gap:8px;padding:8px 0 4px}
.han-hero .mid{text-align:center;flex:1}
.han-hero .logo{display:block;margin:0 auto 6px}
.han-hero h1{font-size:2.35rem;line-height:1.15;font-weight:800;color:var(--navy);margin:0;padding:0}
.han-hero h1 span{color:var(--indigo)}
.han-hero p{color:var(--indigo-d);font-size:1.12rem;font-weight:500;margin:10px 0 6px}
.han-hero .dash{width:22px;height:2px;background:var(--indigo-d);margin:8px auto 0;border-radius:2px}
.han-card{background:#fff;border:1px solid var(--line);border-radius:16px;box-shadow:0 2px 10px rgba(79,70,229,.06);padding:20px 22px;margin:18px 0}
.han-h{display:flex;align-items:center;gap:12px;font-size:1.2rem;font-weight:700;color:var(--navy);margin:0 0 4px}
.han-note{display:flex;align-items:center;gap:10px;color:var(--indigo-d);font-size:.92rem;margin-top:8px}
.han-badge-ok{margin-left:auto;display:flex;align-items:center;gap:8px;background:#ecfdf3;border:1px solid #bbf7d0;color:#166534;border-radius:10px;padding:7px 16px;font-size:.92rem;font-weight:600}
.han-banner{display:flex;align-items:center;gap:18px;border-radius:12px;padding:16px 22px;margin:14px 0}
.han-banner.bad{background:var(--red-bg);border:1px solid var(--red-bd)}
.han-banner.good{background:#f0fdf4;border:1px solid #bbf7d0}
.han-banner h3{margin:0;font-size:1.2rem;font-weight:800;padding:0}
.han-banner.bad h3{color:var(--red)}
.han-banner.good h3{color:#166534}
.han-banner p{margin:2px 0 0;color:var(--navy)}
.han-banner .end{margin-left:auto}
.han-grid{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:14px;margin:6px 0 14px}
.han-cat{min-width:0;box-sizing:border-box;border-radius:14px;padding:18px 12px 16px;border:1px solid var(--line);background:#fff;min-height:200px}
.han-cat.on{border-color:var(--red-bd);background:linear-gradient(180deg,#fff7f7,#fff)}
.han-cat .top{display:flex;align-items:center;gap:5px;font-weight:700;font-size:.86rem;letter-spacing:.1px}.han-cat .top svg{width:22px;height:22px;flex:none}
.han-cat.on .top{color:var(--red)}
.han-cat.off .top{color:var(--slate)}
.han-cat .tag{margin-left:auto;font-size:.64rem;font-weight:700;padding:4px 7px;border-radius:7px;color:#fff;white-space:nowrap}
.han-cat.on .tag{background:var(--red)}
.han-cat.off .tag{background:#64748b}
.han-cat .pct{font-size:2.5rem;font-weight:800;margin:10px 0 0;line-height:1.1}
.han-cat.on .pct{color:var(--red)}
.han-cat.off .pct{color:#475569}
.han-cat .lbl{color:var(--navy);margin:2px 0 8px}
.han-cat .bar{height:10px;border-radius:99px;background:#f1e4e7;overflow:hidden}
.han-cat.off .bar{background:#e6e8ef}
.han-cat .bar i{display:block;height:100%;border-radius:99px}
.han-cat.on .bar i{background:var(--red)}
.han-cat.off .bar i{background:#5b6478}
.han-cat .desc{font-size:.8rem;color:var(--navy);margin-top:12px;line-height:1.6}
.han-att{display:flex;gap:16px;align-items:center}
.han-att .l{flex:1}
.han-att .sub{color:var(--muted);font-size:.92rem;margin:2px 0 14px}
.han-chips{display:flex;flex-wrap:wrap;gap:10px}
.han-chip{padding:7px 14px;border-radius:8px;font-size:.88rem;font-weight:600}
.han-leg{display:flex;align-items:center;gap:10px;font-size:.85rem;margin:14px 0 6px;color:var(--navy)}
.han-leg i{display:block;width:200px;height:8px;border-radius:99px;background:linear-gradient(90deg,#b91c1c,#fde2e2)}
.han-att .cap{color:var(--muted);font-size:.82rem}
.han-wise{display:flex;align-items:center;gap:18px;background:#f1faf4;border:1px solid #d3efdc;border-radius:12px;padding:16px 22px;margin:14px 0}
.han-wise .t{flex:1}
.han-wise h4{margin:0 0 4px;color:#166534;font-size:1.05rem;font-weight:800;padding:0}
.han-wise p{margin:0;font-size:.92rem}
.han-wise .q{border-left:1px solid #cfe6d7;padding-left:22px;color:var(--indigo-d);font-weight:700;font-size:1.05rem;line-height:1.5;max-width:260px}
.han-disc{display:flex;align-items:center;gap:12px;background:#f3f3ff;border:1px solid #e0e0fa;border-radius:12px;padding:14px 20px;color:var(--indigo-d);font-size:.92rem;margin:14px 0}
.han-disc .end{margin-left:auto}
.han-foot{text-align:center;color:var(--muted);font-size:.78rem;margin-top:10px}
.han-foot b{color:var(--indigo-d);font-size:.82rem;display:inline-flex;align-items:center;gap:6px}
.han-empty{color:var(--muted);padding:18px 4px}
/* widget Streamlit */
div[data-testid="stTextArea"] textarea{border-radius:12px!important;border:1px solid var(--line)!important;font-size:1rem;min-height:120px}
div[data-testid="stTextArea"] label{display:none}
div.stButton>button{border-radius:12px;font-weight:700}
div.stButton>button[kind="primary"]{background:var(--indigo)!important;border:none;color:#fff!important;height:78px;font-size:1.15rem;box-shadow:0 6px 16px rgba(79,70,229,.3)}
div.stButton>button[kind="primary"]:hover{background:var(--indigo-d)!important}
div.stButton>button[kind="secondary"]{border:1px solid var(--line);background:#fff;color:var(--navy)}
@media (max-width:820px){
 .han-grid{grid-template-columns:repeat(2,minmax(0,1fr))}
 .han-hero h1{font-size:1.6rem}
 .han-hero img.side{display:none}
 .han-att img{display:none}
 .han-wise{flex-wrap:wrap}.han-wise .q{border-left:none;padding-left:0;max-width:none}
}
@media (max-width:480px){.han-grid{grid-template-columns:1fr}.han-cat .pct{font-size:2rem}}
</style>
"""


def hero_html():
    return (
        '<div class="han-hero">'
        + img_tag("megafon", 130, "side")
        + '<div class="mid">' + img_tag("logo", 62, "logo")
        + '<h1>Deteksi Ujaran Kebencian di <span>Aplikasi X</span></h1>'
        '<p>Kenali Ujaran Kebencian, Bangun Ruang Digital yang Lebih Aman.</p><div class="dash"></div></div>'
        + img_tag("perisai", 110, "side")
        + "</div>"
    )


def input_head_html():
    return f'<div class="han-h">{SVG_EDIT}<span>Masukkan Cuitan...</span></div>'


def model_note_html():
    return (f'<div class="han-note">{SVG_INFO}<span>Aplikasi ini menggunakan model Hierarchical BiLSTM + '
            "Attention untuk klasifikasi multi-label.</span></div>")


def result_head_html(done=True):
    badge = f'<div class="han-badge-ok">{SVG_CHECK_SM}<span>Analisis selesai</span></div>' if done else ""
    return f'<div class="han-h" style="margin-bottom:10px">{SVG_BARS}<span>Hasil Analisis</span>{badge}</div>'


def banner_html(n_detected):
    if n_detected:
        return (f'<div class="han-banner bad">{SVG_WARN}<div><h3>Indikasi ujaran kebencian terdeteksi</h3>'
                f'<p>Model menemukan indikasi pada {n_detected} kategori.</p></div>'
                f'<div class="end">{SVG_SHIELD_RED}</div></div>')
    return (f'<div class="han-banner good">{SVG_ALERT_OK}<div><h3>Tidak ditemukan indikasi ujaran kebencian</h3>'
            '<p>Model tidak menemukan indikasi pada keempat kategori.</p></div>'
            f'<div class="end">{SVG_SHIELD_GREEN}</div></div>')


def cards_html(P, flags, labels, thresholds):
    idx = {l: i for i, l in enumerate(labels)}
    out = ['<div class="han-grid">']
    for key, title, _, tail in CARD_ORDER:
        if key not in idx:
            continue
        p, on = float(P[idx[key]]), bool(flags[idx[key]])
        pct = int(round(p * 100))
        desc = ("Terdeteksi indikasi ujaran kebencian " if on else "Tidak ditemukan indikasi ujaran kebencian ") + tail
        out.append(
            f'<div class="han-cat {"on" if on else "off"}" title="Ambang label ini: {thresholds[key]:.2f}">'
            f'<div class="top">{ICONS[key]}<span>{title}</span><span class="tag">{"Terdeteksi" if on else "Tidak terdeteksi"}</span></div>'
            f'<div class="pct">{pct}%</div><div class="lbl">Skor Analisis</div>'
            f'<div class="bar"><i style="width:{max(2, min(100, pct))}%"></i></div>'
            f'<div class="desc">{html.escape(desc)}</div></div>')
    out.append("</div>")
    return "".join(out)


def top_attention_words(exp, k=9):
    """Ambil k kata dengan bobot (attention kata x kalimat) tertinggi; kembalikan menurut urutan kemunculan."""
    if not exp:
        return []
    items, pos = {}, 0
    for s in exp:
        for tok, w in zip(s["tokens"], s["word_w"]):
            v = w * s["sent_w"]
            if tok not in items or v > items[tok][1]:
                items[tok] = (pos, v)
            pos += 1
    if not items:
        return []
    top = sorted(items.items(), key=lambda kv: -kv[1][1])[:k]
    mx = max(v for _, (_, v) in top) or 1e-9
    top.sort(key=lambda kv: kv[1][0])
    return [(tok, min(1.0, v / mx)) for tok, (_, v) in top]


def _chip_color(a):
    # a=1 -> merah tua (#b91c1c, teks putih); a=0 -> merah muda (#fde2e2, teks merah tua)
    r1, g1, b1, r2, g2, b2 = 253, 226, 226, 185, 28, 28
    a = max(0.0, min(1.0, a)) ** 1.4
    r, g, b = (int(r1 + (r2 - r1) * a), int(g1 + (g2 - g1) * a), int(b1 + (b2 - b1) * a))
    fg = "#fff" if a > 0.55 else "#9f1239"
    return f"rgb({r},{g},{b})", fg


def attention_card_html(exp):
    words = top_attention_words(exp)
    if words:
        chips = "".join(
            f'<span class="han-chip" style="background:{_chip_color(a)[0]};color:{_chip_color(a)[1]}">{html.escape(t)}</span>'
            for t, a in words)
    else:
        chips = '<span class="han-empty">Tidak ada kata yang dapat ditampilkan.</span>'
    return (
        '<div class="han-card"><div class="han-att"><div class="l">'
        f'<div class="han-h">{SVG_EYE}<span>Kata yang mendapat perhatian model (Attention)</span></div>'
        '<div class="sub">Kata dengan warna lebih gelap memiliki bobot perhatian lebih tinggi terhadap hasil prediksi.</div>'
        f'<div class="han-chips">{chips}</div>'
        '<div class="han-leg"><span>Tinggi</span><i></i><span>Rendah</span></div>'
        '<div class="cap">Visualisasi ini menunjukkan fokus model, bukan bukti bahwa kata tersebut merupakan penyebab ujaran kebencian.</div>'
        "</div>" + img_tag("kaca", 140) + "</div></div>"
    )


def wise_html():
    return (
        '<div class="han-wise">' + img_tag("tangan", 58)
        + '<div class="t"><h4>Bijak Berkomunikasi di Ruang Digital</h4>'
        "<p>Perbedaan pendapat adalah hal yang wajar. Mari gunakan kata-kata yang sopan dan menghargai agar media sosial menjadi tempat yang aman dan nyaman untuk semua.</p></div>"
        '<div class="q">“Jari boleh mengetik,<br>hati tetap harus beretika.”</div></div>'
    )


def disclaimer_html():
    return (f'<div class="han-disc">{SVG_INFO}<div>Hasil ini merupakan prediksi model AI dan bukan keputusan mutlak.<br>'
            f'Penilaian manusia tetap diperlukan.</div><div class="end">{img_tag("logo_kecil", 40)}</div></div>')


def footer_html():
    return ('<div class="han-foot"><b>Deteksi Ujaran Kebencian di Aplikasi X</b><br>'
            "Ciptakan ruang digital yang lebih aman dan sehat bersama.</div>")
