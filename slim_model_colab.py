# Jalankan di Colab SETELAH bagian 16 (han_model.pt sudah ada). OPSIONAL: memperkecil ukuran file.
# Embedding fastText dibekukan, jadi tiap model dalam ensemble menyimpan salinan embedding yang IDENTIK.
# Sel ini menyimpannya satu kali saja (prediksi tetap sama persis; tidak ada pembulatan).
import os, torch

ck = torch.load("han_model.pt", map_location="cpu", weights_only=False)
sds = ck["state_dicts"]
emb = sds[0]["emb.weight"]
same = all(torch.equal(emb, s["emb.weight"]) for s in sds)
print("Embedding identik di semua model:", same)
if not same:
    print("Embedding tidak identik (mungkin emb_trainable=True) -> pengecilan DIBATALKAN, pakai han_model.pt apa adanya.")
else:
    ck["shared_emb"] = emb.clone()
    for s in sds:
        s.pop("emb.weight")
    torch.save(ck, "han_model_slim.pt")
    a, b = os.path.getsize("han_model.pt") / 1e6, os.path.getsize("han_model_slim.pt") / 1e6
    print(f"{a:.1f} MB -> {b:.1f} MB. Pakai han_model_slim.pt: ganti namanya menjadi han_model.pt di folder models/ aplikasi.")
