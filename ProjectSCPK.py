import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

st.set_page_config(page_title="SCPK", layout="wide")

CSV_PATH = "IDX_Stock_Summary_2020-2024.csv"
KRITERIA = ["Return", "Likuiditas", "Volume", "Frekuensi", "Volatilitas"]
BENEFIT  = [True, True, True, True, False]
RI       = {1:0.00, 2:0.00, 3:0.58, 4:0.90, 5:1.12}
SKALA    = {
    "1 – Sama penting":1, "3 – Sedikit lebih penting":3,
    "5 – Lebih penting":5, "7 – Sangat lebih penting":7,
    "9 – Mutlak lebih penting":9,
    "1/3":1/3, "1/5":1/5, "1/7":1/7, "1/9":1/9,
}

@st.cache_data
def load():
    df = pd.read_csv(CSV_PATH, low_memory=False)
    df.columns = df.columns.str.strip()

    needed = ["Stock Code","Company Name","Previous","Change",
              "Value","Volume","Frequency","High","Low"]
    col_map = {c.lower().replace(" ","").replace("_",""): c for c in df.columns}
    rename, missing = {}, []
    for n in needed:
        key = n.lower().replace(" ","").replace("_","")
        if n in df.columns: pass
        elif key in col_map: rename[col_map[key]] = n
        else: missing.append(n)
    if missing:
        st.error(f"Kolom tidak ditemukan: {missing}\nTersedia: {df.columns.tolist()}")
        st.stop()
    if rename:
        df = df.rename(columns=rename)

    # Deteksi kolom tanggal
    date_col = None
    for c in df.columns:
        if "date" in c.lower() or "tanggal" in c.lower():
            date_col = c
            break
    if date_col:
        df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
        df = df.rename(columns={date_col: "Date"})

    num_cols = ["Previous","Change","Value","Volume","Frequency","High","Low"]
    df[num_cols] = df[num_cols].apply(pd.to_numeric, errors="coerce").fillna(0)

    # Agregasi groupby Stock Code saja, ambil nama terakhir
    agg = df.groupby("Stock Code", as_index=False)[num_cols].mean()
    agg["Company Name"] = df.groupby("Stock Code")["Company Name"].last().values
    agg["Return"]      = np.where(agg["Previous"]>0, agg["Change"]/agg["Previous"]*100, 0)
    agg["Likuiditas"]  = agg["Value"]
    agg["Frekuensi"]   = agg["Frequency"]
    agg["Volatilitas"] = agg["High"] - agg["Low"]
    return df, agg[(agg["Volume"]>0) & (agg["Value"]>0)].reset_index(drop=True)

df_raw, df_agg = load()

#region AHP
def hitung_ahp(pcm):
    n  = pcm.shape[0]
    nm = pcm / pcm.sum(0)
    pv = nm.mean(1)
    lmx = float(np.mean((pcm @ pv) / pv))
    ci  = (lmx - n) / (n - 1)
    cr  = ci / RI[n]
    return pv, round(lmx,4), round(ci,4), round(cr,4), nm

def normalisasi(s, benefit):
    mn, mx = s.min(), s.max()
    if mx == mn: return pd.Series(np.ones(len(s)), index=s.index)
    return (s-mn)/(mx-mn) if benefit else (mx-s)/(mx-mn)

def skoring(df, bobot, top_n):
    d = df.copy()
    for i, k in enumerate(KRITERIA):
        d[f"N_{k}"] = normalisasi(d[k], BENEFIT[i])
    d["Skor"] = sum(d[f"N_{k}"] * bobot[i] for i, k in enumerate(KRITERIA))
    return d.sort_values("Skor", ascending=False).head(top_n).reset_index(drop=True)
#endregion

st.sidebar.title("SCPK Saham dengan AHP")
st.sidebar.caption(f"`{CSV_PATH}`")
hal = st.sidebar.radio("Menu", ["Data","Hitung AHP","Hasil","Profil Kelompok"])

#region Page 1
if hal == "Data":
    st.title("Pemilihan Saham Terbaik dengan Metode AHP")

    tab1, tab2 = st.tabs(["Data Mentah","Data Agregasi per Saham"])
    with tab1:
        # Filter tahun
        if "Date" in df_raw.columns:
            tahun_list = sorted(df_raw["Date"].dt.year.dropna().unique().astype(int))
            tahun = st.selectbox("Pilih Tahun", ["Semua"] + tahun_list)
            if tahun != "Semua":
                tampil = df_raw[df_raw["Date"].dt.year == tahun]
            else:
                tampil = df_raw
        else:
            tampil = df_raw

        st.caption(f"Total: {len(df_raw):,} baris keseluruhan | Ditampilkan: {min(len(tampil), 10_000):,} baris")
        st.dataframe(tampil.head(10_000), use_container_width=True, height=500, hide_index=True)

    with tab2:
        view = df_agg[["Stock Code","Company Name","Return","Likuiditas","Volume","Frekuensi","Volatilitas"]].copy()
        view.columns = ["Kode","Perusahaan","Return (%)","Nilai Transaksi","Volume","Frekuensi","Volatilitas"]
        st.caption(f"Total emiten: {len(view):,}")
        st.dataframe(
            view.style.format({"Return (%)":"{:.2f}%","Nilai Transaksi":"{:,.0f}",
                               "Volume":"{:,.0f}","Frekuensi":"{:,.0f}","Volatilitas":"{:,.1f}"}),
            use_container_width=True, height=400, hide_index=True
        )
#endregion

#region Page 2
elif hal == "Hitung AHP":
    st.title("Perhitungan AHP")

    c1, c2 = st.columns(2)
    top_n   = c1.slider("Jumlah saham terbaik", 5, 30, 10)
    min_val = c2.number_input("Min. Nilai Transaksi (Juta)", 0, 500_000, 1_000, 500)

    st.markdown("---")
    st.subheader("Matriks Perbandingan Berpasangan")
    n   = len(KRITERIA)
    pcm = np.ones((n, n))
    pairs = [(i,j) for i in range(n) for j in range(i+1,n)]
    for r in range(0, len(pairs), 2):
        cols = st.columns(2)
        for ci2, (i,j) in enumerate(pairs[r:r+2]):
            v = cols[ci2].selectbox(f"{KRITERIA[i]} vs {KRITERIA[j]}", list(SKALA.keys()), key=f"p{i}{j}")
            pcm[i,j], pcm[j,i] = SKALA[v], 1/SKALA[v]

    st.dataframe(pd.DataFrame(pcm, index=KRITERIA, columns=KRITERIA).style.format("{:.3f}"),
                 use_container_width=True)

    if st.button("Jalankan Perhitungan AHP", type="primary", use_container_width=True):
        pv, lmx, ci, cr, nm = hitung_ahp(pcm)

        st.subheader("Langkah 1 Matriks Ternormalisasi & Priority Vector")
        df_nm = pd.DataFrame(nm, index=KRITERIA, columns=KRITERIA)
        df_nm["Priority Vector"] = pv
        st.dataframe(df_nm.style.format("{:.4f}"), use_container_width=True)

        st.subheader("Langkah 2 Uji Konsistensi")
        for label, val in zip(["λ_max","CI","RI","CR"], [lmx, ci, RI[n], cr]):
            st.columns(4)[["λ_max","CI","RI","CR"].index(label)].metric(label, val)
        if cr > 0.1:
            st.error(f"CR = {cr} > 0.10 — TIDAK KONSISTEN. Revisi perbandingan!"); st.stop()
        st.success(f"CR = {cr} ≤ 0.10 — Matriks KONSISTEN.")

        st.subheader("Langkah 3 Bobot Kriteria")
        df_bobot = pd.DataFrame({"Kriteria":KRITERIA,"Bobot":pv,"Bobot (%)":pv*100})
        df_bobot.index = range(1, len(df_bobot) + 1)
        st.dataframe(df_bobot.style.format({"Bobot":"{:.4f}","Bobot (%)":"{:.2f}%"}),
                     use_container_width=True)

        st.subheader("Langkah 4 Skoring & Perangkingan")
        dfw = df_agg[df_agg["Likuiditas"] >= min_val * 1_000_000].copy()
        if dfw.empty:
            st.warning("Tidak ada saham memenuhi filter. Longgarkan filter."); st.stop()

        hasil = skoring(dfw, pv, top_n)
        hasil.insert(0, "Peringkat", range(1, len(hasil)+1))
        st.session_state["hasil"]    = hasil
        st.session_state["pv"]       = pv
        st.session_state["cr"]       = cr
        st.session_state["df_bobot"] = df_bobot
        st.session_state["done"]     = True
        st.success("Selesai! Lihat hasil di halaman Hasil.")
#endregion

#region Page 3
elif hal == "Hasil":
    st.title("Hasil Perangkingan Saham Terbaik")
    if not st.session_state.get("done"):
        st.error("Belum ada hasil. Jalankan perhitungan di halaman Hitung AHP dulu."); st.stop()

    hasil = st.session_state["hasil"]
    cr    = st.session_state["cr"]

    c1,c2,c3 = st.columns(3)
    c1.metric("Saham Terbaik",     hasil.iloc[0]["Stock Code"])
    c2.metric("Skor Tertinggi",    f"{hasil.iloc[0]['Skor']:.4f}")
    c3.metric("Consistency Ratio", f"{cr:.4f}")

    st.markdown("---")
    st.subheader("Tabel Perangkingan")
    disp = hasil[["Peringkat","Stock Code","Company Name",
                  "Return","Likuiditas","Volume","Frekuensi","Volatilitas","Skor"]].copy()
    disp.columns = ["#","Kode","Perusahaan","Return (%)","Nilai Transaksi",
                    "Volume","Frekuensi","Volatilitas","Skor AHP"]

    WARNA = {1:"background-color:#36AA5D", 2:"background-color:#1565C0",
             3:"background-color:#FF6F00"}
    def warna_baris(row):
        return [WARNA.get(int(row.name), "")] * len(row)

    st.dataframe(
        disp.set_index("#").style
            .apply(warna_baris, axis=1)
            .format({"Return (%)":"{:.2f}%","Nilai Transaksi":"{:,.0f}",
                     "Volume":"{:,.0f}","Frekuensi":"{:,.0f}",
                     "Volatilitas":"{:,.1f}","Skor AHP":"{:.4f}"}),
        use_container_width=True, height=450
    )

    st.markdown("---")
    st.subheader("Grafik Skor AHP")
    COLORS = ["#36AA5D","#1565C0","#FF6F00"] + ["#FF817A"] * len(hasil)
    rev    = hasil.iloc[::-1].reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(10, max(5, len(rev)*0.65 + 1.5)))
    bars = ax.barh(range(len(rev)), rev["Skor"],
                   color=COLORS[:len(rev)][::-1], edgecolor="white", height=0.55)
    ax.set_yticks(range(len(rev)))
    ax.set_yticklabels(rev["Stock Code"], fontsize=9)
    for bar, val in zip(bars, rev["Skor"]):
        ax.text(bar.get_width() + 0.006, bar.get_y() + bar.get_height()/2,
                f"{val:.4f}", ha="left", va="center", fontsize=8)
    ax.set(title="Peringkat Saham Berdasarkan Skor AHP", xlabel="Skor AHP", ylabel="Kode Saham",
           xlim=(0, rev["Skor"].max() * 1.20), facecolor="#f9f9f9")
    ax.title.set_fontsize(13); ax.title.set_fontweight("bold")
    fig.patch.set_facecolor("white")
    plt.tight_layout()
    st.pyplot(fig); plt.close(fig)
#endregion

#region Page 4
elif hal == "Profil Kelompok":
    st.title("Profil Kelompok")
    st.info("Mata Kuliah: Praktikum SCPK\n"
            "\nTopik: Pemilihan Saham Terbaik dengan Metode AHP")
    st.markdown("### Anggota Kelompok")
    st.dataframe(pd.DataFrame({
        "No":[1,2],
        "Nama":["Indra Naufal Firdaus","Rio Adhi Permana"],
        "NIM":["123240195","123240206"]
    }), use_container_width=True, hide_index=True)
#endregion