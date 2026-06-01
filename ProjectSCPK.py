import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

st.set_page_config(page_title="SCPK Saham AHP", layout="wide")

CSV_PATH = "IDX_Stock_Summary_2020-2024.csv"

SEKTOR_MAP = {
    "S1": "Agriculture",
    "S2": "Mining",
    "S3": "Basic Industry & Chemicals",
    "S4": "Miscellaneous Industry",
    "S5": "Consumer Goods Industry",
    "S6": "Property, Real Estate & Building Construction",
    "S7": "Infrastructure, Utilities & Transportation",
    "S8": "Finance",
    "S9": "Trade, Services & Investment",
    "M1": "Energy", "U1": "Energy",
    "M2": "Basic Materials", "U2": "Basic Materials",
    "M3": "Industrials", "U3": "Industrials",
    "M4": "Consumer Non-Cyclicals", "U4": "Consumer Non-Cyclicals",
    "M5": "Consumer Cyclicals", "U5": "Consumer Cyclicals",
    "M6": "Healthcare", "U6": "Healthcare",
    "M7": "Financials", "U7": "Financials",
    "M8": "Properties & Real Estate", "U8": "Properties & Real Estate",
    "M9": "Technology", "U9": "Technology",
    "MO": "Infrastructure", "UO": "Infrastructure",
}

def parse_sektor(remarks):
    if pd.isna(remarks): return "Unknown"
    r = str(remarks).strip()
    if len(r) < 3: return "Unknown"
    prefix = r[2].upper()
    if prefix not in ("S", "M", "U"): return "Unknown"
    import re
    m = re.search(r"([1-9O])", r[3:])
    if not m: return "Unknown"
    kode = prefix + m.group(1)
    return SEKTOR_MAP.get(kode, f"Unknown ({kode})")

ALL_KRITERIA = {
    "Return":           {"label": "Return (%)",          "benefit": True,  "col": "Return"},
    "Likuiditas":       {"label": "Nilai Transaksi",      "benefit": True,  "col": "Likuiditas"},
    "Volume":           {"label": "Volume",               "benefit": True,  "col": "Volume"},
    "Frekuensi":        {"label": "Frekuensi",            "benefit": True,  "col": "Frekuensi"},
    "Volatilitas":      {"label": "Volatilitas",          "benefit": False, "col": "Volatilitas"},
    "Market Cap":       {"label": "Market Cap",           "benefit": True,  "col": "MarketCap"},
    "Foreign Net Buy":  {"label": "Foreign Net Buy",      "benefit": True,  "col": "ForeignNetBuy"},
    "Free Float":       {"label": "Free Float (%)",       "benefit": True,  "col": "FreeFloat"},
    "Index Individual": {"label": "Index Individual",     "benefit": True,  "col": "IndexIndividual"},
    "Bid-Ask Spread":   {"label": "Bid-Ask Spread (%)",   "benefit": False, "col": "BidAskSpread"},
}

DEFAULT_KRITERIA = ["Return", "Likuiditas", "Volume", "Frekuensi", "Volatilitas"]

RI_TABLE = {1:0.00, 2:0.00, 3:0.58, 4:0.90, 5:1.12,
            6:1.24, 7:1.32, 8:1.41, 9:1.45, 10:1.49}

SKALA = {
    "1 – Sama penting": 1,
    "3 – Sedikit lebih penting": 3,
    "5 – Lebih penting": 5,
    "7 – Sangat lebih penting": 7,
    "9 – Mutlak lebih penting": 9,
    "1/3": 1/3, "1/5": 1/5, "1/7": 1/7, "1/9": 1/9,
}

@st.cache_data(show_spinner="Memuat data...")
def load():
    df = pd.read_csv(CSV_PATH, low_memory=False)
    df.columns = df.columns.str.strip()

    needed = ["Stock Code","Company Name","Previous","Change",
              "Value","Volume","Frequency","High","Low",
              "Close","Listed Shares","Tradeble Shares",
              "Foreign Sell","Foreign Buy","Index Individual",
              "Offer","Bid","Remarks"]
    col_map = {c.lower().replace(" ","").replace("_",""): c for c in df.columns}
    rename = {}
    for n in needed:
        key = n.lower().replace(" ","").replace("_","")
        if n not in df.columns and key in col_map:
            rename[col_map[key]] = n
    if rename:
        df = df.rename(columns=rename)

    for c in df.columns:
        if "date" in c.lower() or "tanggal" in c.lower():
            df[c] = pd.to_datetime(df[c], errors="coerce")
            df = df.rename(columns={c: "Date"})
            break

    num_cols = ["Previous","Change","Value","Volume","Frequency","High","Low",
                "Close","Listed Shares","Tradeble Shares",
                "Foreign Sell","Foreign Buy","Index Individual","Offer","Bid"]
    for c in num_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)

    import re as _re
    def _parse_kode(remarks):
        if pd.isna(remarks): return None
        r = str(remarks).strip()
        if len(r) < 3: return None
        prefix = r[2].upper()
        if prefix not in ("S","M","U"): return None
        m = _re.search(r"([1-9O])", r[3:])
        if not m: return None
        kode = prefix + m.group(1)
        return kode if kode in SEKTOR_MAP else None

    if "Remarks" in df.columns:
        df["_kode_sektor"] = df["Remarks"].apply(_parse_kode)
    else:
        df["_kode_sektor"] = None

    def _pilih_kode(group):
        unik = group.dropna().unique()
        if len(unik) == 0: return None
        s_kodes = [k for k in unik if k.startswith("S")]
        if s_kodes:
            return group[group.isin(s_kodes)].mode().iloc[0]
        mo_kodes = [k for k in unik if k in ("MO","UO")]
        if mo_kodes:
            return group[group.isin(mo_kodes)].mode().iloc[0]
        return group.mode().iloc[0]

    sektor_per_saham = (
        df[df["_kode_sektor"].notna()]
        .groupby("Stock Code")["_kode_sektor"]
        .apply(_pilih_kode)
        .map(SEKTOR_MAP)
        .fillna("Unknown")
    )

    agg_cols = ["Previous","Change","Value","Volume","Frequency","High","Low",
                "Close","Listed Shares","Tradeble Shares",
                "Foreign Sell","Foreign Buy","Index Individual","Offer","Bid"]
    agg = df.groupby("Stock Code", as_index=False)[agg_cols].mean()
    agg["Company Name"] = df.groupby("Stock Code")["Company Name"].last().values
    agg["Sektor"] = agg["Stock Code"].map(sektor_per_saham).fillna("Unknown")

    agg["Return"]          = np.where(agg["Previous"]>0, agg["Change"]/agg["Previous"]*100, 0)
    agg["Likuiditas"]      = agg["Value"]
    agg["Frekuensi"]       = agg["Frequency"]
    agg["Volatilitas"]     = agg["High"] - agg["Low"]
    agg["MarketCap"]       = agg["Close"] * agg["Listed Shares"]
    agg["ForeignNetBuy"]   = agg["Foreign Buy"] - agg["Foreign Sell"]
    agg["FreeFloat"]       = np.where(agg["Listed Shares"]>0,
                                       agg["Tradeble Shares"]/agg["Listed Shares"]*100, 0)
    agg["IndexIndividual"] = agg["Index Individual"]
    agg["BidAskSpread"]    = np.where(agg["Close"]>0,
                                       (agg["Offer"]-agg["Bid"])/agg["Close"]*100, 0)

    valid = agg[(agg["Volume"]>0) & (agg["Value"]>0)].reset_index(drop=True)
    return df, valid

df_raw, df_agg = load()

def hitung_ahp(pcm):
    n   = pcm.shape[0]
    nm  = pcm / pcm.sum(0)
    pv  = nm.mean(1)
    lmx = float(np.mean((pcm @ pv) / pv))
    ci  = (lmx - n) / (n - 1)
    cr  = ci / RI_Table(n)
    return pv, round(lmx,4), round(ci,4), round(cr,4), nm

def RI_Table(n):
    return RI_TABLE.get(n, 1.49)

def normalisasi(s, benefit):
    mn, mx = s.min(), s.max()
    if mx == mn:
        return pd.Series(np.ones(len(s)), index=s.index)
    return (s-mn)/(mx-mn) if benefit else (mx-s)/(mx-mn)

def skoring(df, kriteria_aktif, bobot, top_n):
    d = df.copy()
    for k in kriteria_aktif:
        info = ALL_KRITERIA[k]
        d[f"N_{k}"] = normalisasi(d[info["col"]], info["benefit"])
    d["Skor"] = sum(d[f"N_{k}"] * bobot[i] for i, k in enumerate(kriteria_aktif))
    return d.sort_values("Skor", ascending=False).head(top_n).reset_index(drop=True)

st.sidebar.title("SCPK Saham AHP")
hal = st.sidebar.radio("Menu", ["Data", "Hitung AHP", "Hasil", "Profil Kelompok"])

# Page 1
if hal == "Data":
    st.title("Pemilihan Saham Terbaik dengan Metode AHP")

    tab1, tab2 = st.tabs(["Data Mentah", "Data Agregasi per Saham"])

    with tab1:
        if "Date" in df_raw.columns:
            tahun_list = sorted(df_raw["Date"].dt.year.dropna().unique().astype(int))
            tahun = st.selectbox("Pilih Tahun", ["Semua"] + tahun_list)
            tampil = df_raw[df_raw["Date"].dt.year == tahun] if tahun != "Semua" else df_raw
        else:
            tampil = df_raw
        st.caption(f"Total: {len(df_raw):,} baris | Ditampilkan: {min(len(tampil), 10_000):,} baris")
        st.dataframe(tampil.head(10_000), use_container_width=True, height=500, hide_index=True)

    with tab2:
        view = df_agg[["Stock Code","Company Name","Sektor","Return","Likuiditas",
                        "Volume","Frekuensi","Volatilitas",
                        "MarketCap","ForeignNetBuy","FreeFloat",
                        "IndexIndividual","BidAskSpread"]].copy()
        view.columns = ["Kode","Perusahaan","Sektor","Return (%)","Nilai Transaksi",
                        "Volume","Frekuensi","Volatilitas","Market Cap","Foreign Net Buy",
                        "Free Float (%)","Index Individual","Bid-Ask Spread (%)"]
        st.caption(f"Total emiten: {len(view):,}")
        st.dataframe(
            view.style.format({
                "Return (%)": "{:.2f}%",
                "Nilai Transaksi": "{:,.0f}",
                "Volume": "{:,.0f}",
                "Frekuensi": "{:,.0f}",
                "Volatilitas": "{:,.1f}",
                "Market Cap": "{:,.0f}",
                "Foreign Net Buy": "{:,.0f}",
                "Free Float (%)": "{:.1f}%",
                "Index Individual": "{:,.2f}",
                "Bid-Ask Spread (%)": "{:.2f}%",
            }),
            use_container_width=True, height=450, hide_index=True
        )

# Page 2
elif hal == "Hitung AHP":
    st.title("Perhitungan AHP")

    st.subheader("1. Pilih Kriteria")
    st.caption("Minimal 2, maksimal 10 kriteria. Default: 5 kriteria standar.")
    col_k1, col_k2 = st.columns(2)
    kriteria_aktif = []
    items = list(ALL_KRITERIA.items())
    for i, (k, info) in enumerate(items):
        col = col_k1 if i < len(items)//2 + 1 else col_k2
        checked = col.checkbox(
            f"{k} ({'benefit' if info['benefit'] else 'cost'})",
            value=(k in DEFAULT_KRITERIA),
            key=f"chk_{k}"
        )
        if checked:
            kriteria_aktif.append(k)
 
    if len(kriteria_aktif) < 2:
        st.error("Pilih minimal 2 kriteria."); st.stop()
    if len(kriteria_aktif) > 10:
        st.error("Maksimal 10 kriteria."); st.stop()
 
    st.info(f"Kriteria aktif ({len(kriteria_aktif)}): **{', '.join(kriteria_aktif)}**")
    st.markdown("---")

    st.subheader("2. Filter Saham")
    mode_filter = st.radio("Mode filter", ["Filter Sektor", "Pilih Saham Manual", "Semua Saham"],
                            horizontal=True)

    dfw = df_agg.copy()

    if mode_filter == "Filter Sektor":
        sektor_list = sorted([s for s in dfw["Sektor"].unique() if s != "Unknown"])
        pilih_sektor = st.multiselect("Pilih Sektor", sektor_list, default=sektor_list[:3])
        if pilih_sektor:
            dfw = dfw[dfw["Sektor"].isin(pilih_sektor)]
        st.caption(f"Saham tersedia: {len(dfw):,}")

    elif mode_filter == "Pilih Saham Manual":
        semua_saham = sorted(df_agg["Stock Code"].tolist())
        label_map = {f"{row['Stock Code']} – {row['Company Name']}": row['Stock Code']
                     for _, row in df_agg.iterrows()}
        pilihan = st.multiselect(
            "Pilih saham (ketik kode atau nama)",
            options=list(label_map.keys()),
            max_selections=50,
            placeholder="Cari kode saham..."
        )
        if pilihan:
            kode_dipilih = [label_map[p] for p in pilihan]
            dfw = dfw[dfw["Stock Code"].isin(kode_dipilih)]
        else:
            st.warning("Pilih minimal 1 saham."); st.stop()
        st.caption(f"Saham dipilih: {len(dfw)}")

    else:
        min_val = st.number_input("Min. Nilai Transaksi (Juta)", 0, 500_000, 1_000, 500)
        dfw = dfw[dfw["Likuiditas"] >= min_val * 1_000_000]
        st.caption(f"Saham tersedia setelah filter: {len(dfw):,}")

    if dfw.empty:
        st.warning("Tidak ada saham memenuhi filter."); st.stop()

    _max_n = max(1, min(50, len(dfw)))
    _def_n = min(10, _max_n)
    top_n = st.slider("Jumlah saham terbaik ditampilkan", 1, _max_n, _def_n)
    st.markdown("---")

    st.subheader("3. Matriks Perbandingan Berpasangan")
    n   = len(kriteria_aktif)
    pcm = np.ones((n, n))
    pairs = [(i,j) for i in range(n) for j in range(i+1,n)]

    for r in range(0, len(pairs), 2):
        cols = st.columns(2)
        for ci2, (i,j) in enumerate(pairs[r:r+2]):
            v = cols[ci2].selectbox(
                f"{kriteria_aktif[i]} vs {kriteria_aktif[j]}",
                list(SKALA.keys()), key=f"p{i}{j}"
            )
            pcm[i,j], pcm[j,i] = SKALA[v], 1/SKALA[v]

    st.dataframe(
        pd.DataFrame(pcm, index=kriteria_aktif, columns=kriteria_aktif).style.format("{:.3f}"),
        use_container_width=True
    )

    if st.button("Jalankan Perhitungan AHP", type="primary", use_container_width=True):
        pv, lmx, ci, cr, nm = hitung_ahp(pcm)

        st.subheader("Langkah 1 Matriks Ternormalisasi & Priority Vector")
        df_nm = pd.DataFrame(nm, index=kriteria_aktif, columns=kriteria_aktif)
        df_nm["Priority Vector"] = pv
        st.dataframe(df_nm.style.format("{:.4f}"), use_container_width=True)

        st.subheader("Langkah 2 Uji Konsistensi")
        cols4 = st.columns(4)
        for label, val in zip(["λ_max","CI","RI","CR"], [lmx, ci, RI_Table(n), cr]):
            cols4[["λ_max","CI","RI","CR"].index(label)].metric(label, val)

        if cr > 0.1:
            st.error(f"CR = {cr} > 0.10 — TIDAK KONSISTEN. Revisi perbandingan!")
            st.stop()
        st.success(f"CR = {cr} ≤ 0.10 — Matriks KONSISTEN ✓")

        st.subheader("Langkah 3 Bobot Kriteria")
        df_bobot = pd.DataFrame({
            "Kriteria": kriteria_aktif,
            "Tipe": ["Benefit" if ALL_KRITERIA[k]["benefit"] else "Cost" for k in kriteria_aktif],
            "Bobot": pv,
            "Bobot (%)": pv * 100
        })
        df_bobot.index = range(1, len(df_bobot)+1)
        st.dataframe(df_bobot.style.format({"Bobot":"{:.4f}","Bobot (%)":"{:.2f}%"}),
                     use_container_width=True)

        st.subheader("Langkah 4 Skoring & Perangkingan")
        hasil = skoring(dfw, kriteria_aktif, pv, top_n)
        hasil.insert(0, "Peringkat", range(1, len(hasil)+1))

        st.session_state["hasil"] = hasil
        st.session_state["pv"] = pv
        st.session_state["cr"] = cr
        st.session_state["df_bobot"] = df_bobot
        st.session_state["kriteria_aktif"] = kriteria_aktif
        st.session_state["done"] = True
        st.success("Selesai! Lihat hasil di halaman Hasil.")

# Page 3
elif hal == "Hasil":
    st.title("Hasil Perangkingan Saham Terbaik")

    if not st.session_state.get("done"):
        st.error("Belum ada hasil. Jalankan perhitungan di halaman Hitung AHP dulu.")
        st.stop()

    hasil = st.session_state["hasil"]
    cr = st.session_state["cr"]
    kriteria_aktif= st.session_state["kriteria_aktif"]

    c1, c2, c3 = st.columns(3)
    c1.metric("Saham Terbaik", hasil.iloc[0]["Stock Code"])
    c2.metric("Skor Tertinggi", f"{hasil.iloc[0]['Skor']:.4f}")
    c3.metric("Consistency Ratio", f"{cr:.4f}")

    st.markdown("---")
    st.subheader("Tabel Perangkingan")

    tampil_cols = ["Peringkat","Stock Code","Company Name","Sektor"] + \
                  [ALL_KRITERIA[k]["col"] for k in kriteria_aktif] + ["Skor"]
    tampil_cols = [c for c in tampil_cols if c in hasil.columns]
    disp = hasil[tampil_cols].copy()

    rename_disp = {"Peringkat":"#","Stock Code":"Kode","Company Name":"Perusahaan","Skor":"Skor AHP"}
    for k in kriteria_aktif:
        rename_disp[ALL_KRITERIA[k]["col"]] = ALL_KRITERIA[k]["label"]
    disp = disp.rename(columns=rename_disp)

    WARNA = {1:"background-color:#36AA5D;color:white",
             2:"background-color:#1565C0;color:white",
             3:"background-color:#FF6F00;color:white"}

    def warna_baris(row):
        rank = row.name
        style = WARNA.get(int(rank), "")
        return [style] * len(row)

    fmt = {"Skor AHP": "{:.4f}"}
    for k in kriteria_aktif:
        lbl = ALL_KRITERIA[k]["label"]
        if "%" in lbl:
            fmt[lbl] = "{:.2f}%"
        else:
            fmt[lbl] = "{:,.2f}"

    st.dataframe(
        disp.set_index("#").style.apply(warna_baris, axis=1).format(fmt, na_rep="-"),
        use_container_width=True, height=450
    )

    st.markdown("---")
    st.subheader("Grafik Skor AHP")
    COLORS = ["#36AA5D","#1565C0","#FF6F00"] + ["#FF817A"] * len(hasil)
    rev = hasil.iloc[::-1].reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(10, max(5, len(rev)*0.65 + 1.5)))
    bars = ax.barh(range(len(rev)), rev["Skor"],
                   color=COLORS[:len(rev)][::-1], edgecolor="white", height=0.55)
    ax.set_yticks(range(len(rev)))
    ax.set_yticklabels(rev["Stock Code"], fontsize=9)
    for bar, val in zip(bars, rev["Skor"]):
        ax.text(bar.get_width()+0.005, bar.get_y()+bar.get_height()/2,
                f"{val:.4f}", ha="left", va="center", fontsize=8)
    ax.set(title="Peringkat Saham Berdasarkan Skor AHP",
           xlabel="Skor AHP", ylabel="Kode Saham",
           xlim=(0, rev["Skor"].max()*1.20), facecolor="#f9f9f9")
    ax.title.set_fontsize(13); ax.title.set_fontweight("bold")
    fig.patch.set_facecolor("white")
    plt.tight_layout()
    st.pyplot(fig); plt.close(fig)

#region Page 4
elif hal == "Profil Kelompok":
    st.title("Profil Kelompok")
    st.info("Mata Kuliah: Praktikum SCPK\n\nTopik: Pemilihan Saham Terbaik dengan Metode AHP")
    st.markdown("### Anggota Kelompok")
    st.dataframe(pd.DataFrame({
        "No": [1, 2],
        "Nama": ["Indra Naufal Firdaus", "Rio Adhi Permana"],
        "NIM": ["123240195", "123240206"]
    }), use_container_width=True, hide_index=True)
#endregion