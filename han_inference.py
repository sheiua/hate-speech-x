"""Inferensi Hierarchical BiLSTM-Attention (word -> sentence) untuk hate tweet multi-label.

Preprocessing, encoding, aturan keputusan, dan kelas model di sini disalin identik dari notebook training
supaya prediksi di aplikasi sama dengan di notebook. Hanya bagian model yang membutuhkan torch.
"""
import html
import re

import numpy as np
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

# ------------------------------------------------------------------ preprocessing (identik dengan notebook)
RE_BYTES = re.compile(r"(?:\\x[0-9a-fA-F]{2})+")
RE_ESC = re.compile(r"\\[nrt]")
RE_URL = re.compile(r"https?://\S+|www\.\S+")
RE_PLACEHOLDER = re.compile(r"\b(?:USER|URL)\b|\bRT\b")
RE_NONALNUM = re.compile(r"[^a-z0-9\s]")
RE_REPEAT = re.compile(r"(.)\1{2,}")
SEG_PATTERNS = {
    "sentence": re.compile(r"[.!?]+(?:\s+|$)|[\n\r]+"),
    "clause": re.compile(r"[.!?;:]+(?:\s+|$)|,+\s*|[\n\r]+|\s+-+\s+|\.{2,}"),
}


def normalize_segment(seg, kamus):
    out = []
    for tok in RE_NONALNUM.sub(" ", seg.lower()).split():
        if tok in kamus:
            out.append(kamus[tok])
            continue
        tok2 = RE_REPEAT.sub(r"\1", tok)
        out.append(kamus.get(tok2, tok2))
    return " ".join(w for w in " ".join(out).split() if not w.isdigit())


def segment_text(text, kamus, mode="sentence"):
    """Tweet mentah -> daftar kalimat ternormalisasi (kalimat kosong dibuang)."""
    t = str(text)
    t = RE_ESC.sub("\n", t)
    t = RE_BYTES.sub(" ", t)
    t = RE_URL.sub(" ", t)
    t = RE_PLACEHOLDER.sub(" ", t)
    segs = [normalize_segment(p, kamus) for p in SEG_PATTERNS[mode].split(t)]
    return [s for s in segs if s]


def encode_han(seg_texts, vocab, max_sent, max_words):
    """List string berpemisah ' | ' -> array (N, max_sent, max_words)."""
    ids = np.zeros((len(seg_texts), max_sent, max_words), dtype=np.int64)
    for i, t in enumerate(seg_texts):
        for s, seg in enumerate([x for x in t.split(" | ") if x.strip()][:max_sent]):
            toks = seg.split()[:max_words]
            ids[i, s, :len(toks)] = [vocab.get(w, 1) for w in toks]
        if not ids[i].any():
            ids[i, 0, 0] = 1
    return ids


def decide(P, thr, labels, mode="none"):
    """Probabilitas (n,4) -> label biner. mode 'none' = tiap label diputuskan sendiri."""
    T = np.array([thr[l] for l in labels])
    Fl = P >= T
    if mode != "none":
        ii, gi = labels.index("HS_Individual"), labels.index("HS_Group")
        ratio = P / T
        ind_wins = ratio[:, ii] >= ratio[:, gi]
        both = Fl[:, ii] & Fl[:, gi]
        Fl[both & ind_wins, gi] = False
        Fl[both & ~ind_wins, ii] = False
        if mode == "full":
            others = [labels.index(l) for l in labels if l not in ("HS_Individual", "HS_Group")]
            orphan = Fl[:, others].any(axis=1) & ~Fl[:, ii] & ~Fl[:, gi]
            Fl[orphan & ind_wins, ii] = True
            Fl[orphan & ~ind_wins, gi] = True
    return Fl


# ------------------------------------------------------------------ model (identik dengan notebook)
class AdditiveAttention(nn.Module):
    def __init__(self, dim, att_dim=64):
        super().__init__()
        self.proj = nn.Linear(dim, att_dim)
        self.v = nn.Linear(att_dim, 1, bias=False)

    def forward(self, H, mask):
        s = self.v(torch.tanh(self.proj(H))).squeeze(-1)
        s = s.masked_fill(~mask, -1e9)
        a = torch.softmax(s, dim=-1)
        ctx = torch.bmm(a.unsqueeze(1), H).squeeze(1)
        return ctx, a


def run_lstm(lstm, x, lengths):
    packed = pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
    out, _ = lstm(packed)
    out, _ = pad_packed_sequence(out, batch_first=True, total_length=x.size(1))
    return out


class HAN(nn.Module):
    def __init__(self, vocab_size, emb_dim, w_hidden, s_hidden, dropout, emb_matrix=None, emb_trainable=True,
                 n_out=4, att_dim=64):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        if emb_matrix is not None:
            self.emb.weight.data.copy_(torch.from_numpy(emb_matrix))
        self.emb.weight.requires_grad = bool(emb_trainable)
        self.emb_drop = nn.Dropout(dropout)
        self.drop = nn.Dropout(dropout)
        self.word_lstm = nn.LSTM(emb_dim, w_hidden, batch_first=True, bidirectional=True)
        self.word_att = AdditiveAttention(2 * w_hidden, att_dim)
        self.sent_lstm = nn.LSTM(2 * w_hidden, s_hidden, batch_first=True, bidirectional=True)
        self.sent_att = AdditiveAttention(2 * s_hidden, att_dim)
        self.out = nn.Linear(2 * s_hidden, n_out)

    def forward(self, ids, return_att=False):
        B, S, W = ids.shape
        flat = ids.reshape(B * S, W)
        real = (flat != 0).any(dim=1)
        pos = real.nonzero(as_tuple=False).squeeze(1)
        words = flat[pos]
        wmask = words != 0
        wlen = wmask.sum(1).clamp(min=1)
        Hw = self.drop(run_lstm(self.word_lstm, self.emb_drop(self.emb(words)), wlen))
        sv, aw = self.word_att(Hw, wmask)
        D = sv.size(1)
        sent_vecs = torch.zeros(B * S, D, device=ids.device, dtype=sv.dtype).index_copy(0, pos, sv).view(B, S, D)
        smask = real.view(B, S)
        slen = smask.sum(1).clamp(min=1)
        Hs = self.drop(run_lstm(self.sent_lstm, self.drop(sent_vecs), slen))
        doc, a_sent = self.sent_att(Hs, smask)
        logits = self.out(self.drop(doc))
        if return_att:
            aw_full = torch.zeros(B * S, W, device=ids.device, dtype=aw.dtype).index_copy(0, pos, aw).view(B, S, W)
            return logits, {"word": aw_full, "sentence": a_sent}
        return logits


def trim_batch(xb):
    s = max(1, int((xb != 0).any(dim=2).sum(1).max()))
    w = max(1, int((xb != 0).sum(2).max()))
    return xb[:, :s, :w].contiguous()


# ------------------------------------------------------------------ predictor
class HANPredictor:
    """Memuat han_model.pt (ensemble beberapa model) dan melakukan prediksi."""

    def __init__(self, path):
        try:
            ck = torch.load(str(path), map_location="cpu", weights_only=False)
        except TypeError:                      # torch lama tanpa argumen weights_only
            ck = torch.load(str(path), map_location="cpu")
        cfg = ck["cfg"]
        if cfg.get("arch", "han") != "han":
            raise ValueError("Checkpoint bukan arsitektur HAN (arch=%r)" % cfg.get("arch"))
        self.cfg = cfg
        self.labels = list(ck["labels"])
        self.display = dict(ck["display"])
        self.thresholds = dict(ck["thresholds"])
        self.mode = ck.get("consistency_mode", "none")
        self.kamus = ck["kamus"]
        self.vocab = ck["vocab"]
        self.max_sent, self.max_words = int(ck["max_sent"]), int(ck["max_words"])
        self.segment_mode = cfg.get("segment_mode", "sentence")
        self.meta = {"torch_version_training": ck.get("torch_version", "?"), "n_models": len(ck["state_dicts"]),
                     "emb_dim": ck["emb_dim"], "vocab_size": len(self.vocab)}
        shared = ck.get("shared_emb")          # format ramping: embedding frozen disimpan sekali
        self.models = []
        for sd in ck["state_dicts"]:
            sd = dict(sd)
            if shared is not None:
                sd["emb.weight"] = shared
            m = HAN(len(self.vocab), int(ck["emb_dim"]), int(cfg["w_hidden"]), int(cfg["s_hidden"]),
                    float(cfg["dropout"]), None, False, len(self.labels))
            m.load_state_dict(sd)
            m.eval()
            self.models.append(m)

    # -- hanya dua metode ini yang menjalankan jaringan
    def _forward_probs(self, ids, batch_size=64):
        outs = []
        with torch.no_grad():
            for m in self.models:
                ps = []
                for i in range(0, len(ids), batch_size):
                    b = trim_batch(torch.from_numpy(ids[i:i + batch_size]))
                    ps.append(torch.sigmoid(m(b)).numpy())
                outs.append(np.vstack(ps))
        return np.mean(outs, axis=0)

    def _forward_attention(self, ids):
        with torch.no_grad():
            _, att = self.models[0](torch.from_numpy(ids), return_att=True)
        return att["word"][0].numpy(), att["sentence"][0].numpy()

    def _segments(self, text):
        return segment_text(text, self.kamus, self.segment_mode)

    def predict(self, texts):
        """Mengembalikan dict: P (n,4), flags (n,4 bool), empty (n bool), prediksi (n str), truncated (n bool)."""
        segs = [self._segments(t) for t in texts]
        empty = np.array([len(s) == 0 for s in segs])
        truncated = np.array([len(s) > self.max_sent or any(len(x.split()) > self.max_words for x in s) for s in segs])
        ids = encode_han([" | ".join(s) for s in segs], self.vocab, self.max_sent, self.max_words)
        P = self._forward_probs(ids)
        flags = decide(P, self.thresholds, self.labels, self.mode)
        flags[empty] = False
        prediksi = ["tidak dapat dianalisis" if e else (", ".join(self.display[l] for l, f in zip(self.labels, row) if f)
                                                         or "bukan hate tweet") for e, row in zip(empty, flags)]
        return {"P": P, "flags": flags, "empty": empty, "prediksi": prediksi, "truncated": truncated}

    def explain(self, text):
        """Bobot attention (kata x kalimat) dari model pertama ensemble. None bila teks kosong."""
        segs = self._segments(text)[: self.max_sent]
        if not segs:
            return None
        ids = encode_han([" | ".join(segs)], self.vocab, self.max_sent, self.max_words)
        aw, a_s = self._forward_attention(ids)
        out = []
        for i, seg in enumerate(segs):
            toks = seg.split()[: self.max_words]
            out.append({"tokens": toks, "word_w": [float(x) for x in aw[i, :len(toks)]], "sent_w": float(a_s[i])})
        return out


def attention_html(exp):
    """HTML sorotan kata (bobot = attention kata x attention kalimat, dinormalisasi).
    Chip opak dengan teks gelap agar terbaca di tema gelap maupun terang."""
    if not exp:
        return ""
    mx = max((w * s["sent_w"] for s in exp for w in s["word_w"]), default=1e-9) + 1e-9
    rows = []
    for i, s in enumerate(exp):
        chips = []
        for tok, w in zip(s["tokens"], s["word_w"]):
            a = min(1.0, (w * s["sent_w"]) / mx)
            bg = f"rgb(255,{int(255 - 120 * a)},{int(255 - 225 * a)})"
            chips.append(f"<span style='background:{bg};color:#111;padding:2px 5px;margin:1px;border-radius:4px;"
                         f"display:inline-block'>{html.escape(tok)}</span>")
        rows.append(f"<div style='margin-bottom:8px'><b>Kalimat {i + 1}</b> "
                    f"<span style='opacity:.7'>(bobot kalimat {s['sent_w']:.2f})</span><br>{' '.join(chips)}</div>")
    return "".join(rows)
