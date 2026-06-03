---
jupytext:
  formats: md:myst
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.11.5
kernelspec:
  display_name: Python 3
  language: python
  name: python3
---
# Peramalan Kadar NO₂ di Daerah Nganjuk

## Daftar Isi

- [Latar Belakang](#latar-belakang)
- [1. Pengumpulan Data](#1-pengumpulan-data)
- [2. Preprocessing Data](#2-preprocessing-data)
  - [a. Membaca File NetCDF (.nc)](#a-membaca-file-netcdf-nc)
  - [b. Mengatasi Missing Value menggunakan Interpolasi Linear](#b-mengatasi-missing-value-menggunakan-interpolasi-linear)
  - [c. Rata-ratakan Data dan Ubah Datetime](#c-rata-ratakan-data-dan-ubah-datetime)
  - [d. Simpan Data dalam Bentuk CSV](#d-simpan-data-dalam-bentuk-csv)
  - [e. Pengecekan Missing Value Data Harian pada CSV](#e-pengecekan-missing-value-data-harian-pada-csv)
  - [f. Deteksi dan Penanganan Outlier (IQR)](#f-deteksi-dan-penanganan-outlier-iqr)
- [3. Modeling menggunakan KNN Regression](#3-modeling-menggunakan-knn-regression)
  - [a. Normalisasi Data](#a-normalisasi-data)
  - [b. Uji Korelasi Data](#b-uji-korelasi-data)
  - [c. Mengubah Data menjadi Supervised](#c-mengubah-data-menjadi-supervised)
  - [d. Modeling dan Evaluasi](#d-modeling-dan-evaluasi)
  - [e. Plotting](#e-plotting)
- [Kesimpulan](#kesimpulan)

---

## Latar Belakang

Peningkatan aktivitas industri, transportasi, serta pertumbuhan populasi yang pesat telah menyebabkan peningkatan signifikan terhadap tingkat pencemaran udara di berbagai wilayah. Salah satu polutan udara utama yang menjadi perhatian adalah **Nitrogen Dioksida (NO₂)**, yaitu gas beracun yang dihasilkan terutama dari proses pembakaran bahan bakar fosil seperti kendaraan bermotor, pembangkit listrik, dan kegiatan industri.

NO₂ memiliki dampak serius terhadap kesehatan manusia, seperti gangguan pernapasan, iritasi paru-paru, serta memperburuk penyakit asma dan bronkitis. Selain itu, NO₂ juga berkontribusi terhadap pembentukan hujan asam dan penurunan kualitas lingkungan secara keseluruhan.

Pada proyek ini, dilakukan peramalan kadar NO₂ harian di daerah **Nganjuk** menggunakan data Time Series dari satelit Sentinel-5P dan model **KNN Regression**.

---

## 1. Pengumpulan Data

Data Time Series harian kadar NO₂ di daerah Nganjuk diambil dari sumber:

- **Website**: [https://dataspace.copernicus.eu/](https://dataspace.copernicus.eu/)
- **Dokumentasi pengambilan data**: [https://documentation.dataspace.copernicus.eu/notebook-samples/openeo/NO2Covid.html](https://documentation.dataspace.copernicus.eu/notebook-samples/openeo/NO2Covid.html)


Buat akun terlebih dahulu di website Copernicus, lalu akses JupyterLab dan pilih kernel **Python 3 (ipykernel)**.

### Install Library

```bash
pip install openeo
pip install netCDF4
```

### Autentikasi dan Pengambilan Data

```python
import openeo

connection = openeo.connect("openeo.dataspace.copernicus.eu").authenticate_oidc()
```

Saat menjalankan baris di atas, akan muncul permintaan autentikasi:

```
Visit (link authentikasi) 📋 to authenticate.
✅ Authorized successfully
Authenticated using device code flow.
```

Klik link autentikasi lalu login menggunakan akun Copernicus.

### Definisi Area dan Pengambilan Data NO₂

Koordinat area Nganjuk diperoleh dari [https://geojson.io](https://geojson.io) dengan menggambar kotak di atas wilayah yang diinginkan.

```python
aoi = {
    "type": "Polygon",
    "coordinates": [
        [
            [111.80, -7.40],
            [112.10, -7.40],
            [112.10, -7.70],
            [111.80, -7.70],
            [111.80, -7.40],
        ]
    ]
}

s5post = connection.load_collection(
    "SENTINEL_5P_L2",
    temporal_extent=["2023-10-01", "2026-06-01"],
    spatial_extent={
        "west": 111.80,
        "south": -7.70,
        "east": 112.10,
        "north": -7.40
    },
    bands=["NO2"],
)

# Agregasi harian agar tidak ada lebih dari satu data per hari
s5p_no2_daily = s5post.aggregate_temporal_period(reducer="mean", period="day")

# Agregasi spasial untuk menghasilkan rata-rata time series per AOI
s5p_no2_aoi = s5p_no2_daily.aggregate_spatial(reducer="mean", geometries=aoi)
```

### Eksekusi Job dan Download

```python
job = s5post.execute_batch(title="NO2 in Nganjuk", outputfile="NO2Nganjuk.nc")
```

Tunggu proses selesai — status bisa dipantau di [https://editor.openeo.org](https://editor.openeo.org/?server=https%3A%2F%2Fopeneo.dataspace.copernicus.eu%2Fopeneo%2F1.2). Output berupa file **`NO2Nganjuk.nc`**.

---

## 2. Preprocessing Data

### a. Membaca File NetCDF (.nc)

File hasil unduhan berbentuk `.nc` (NetCDF). Kita perlu mengekstrak kolom `date` dan `NO2` menggunakan library `netCDF4`.

```python
import netCDF4

file_path = "NO2Nganjuk.nc"
ds = netCDF4.Dataset(file_path)

# Lihat seluruh variabel yang tersedia
print("📦 Variabel dalam file:")
print(ds.variables.keys())
# dict_keys(['t', 'x', 'y', 'crs', 'NO2'])

# Ambil NO2
no2 = ds.variables["NO2"][:]

# Ambil Time
time = ds.variables["t"][:]

# Konversi waktu ke format tanggal
try:
    time_units = ds.variables["t"].units
    dates = netCDF4.num2date(time, units=time_units)
except Exception:
    dates = time  # fallback jika tidak ada units

# Tampilkan struktur data NO2
print(type(no2))
# <class 'numpy.ma.core.MaskedArray'>

print(len(no2))       # banyaknya record (misal: 725)
print(len(no2[0]))    # panjang data perbaris (misal: 9)
print(len(no2[0][0])) # panjang perdata (misal: 8)
print(no2[0][0][0])   # contoh nilai: 3.7701793e-05
```

Struktur data NO2 perbaris adalah array 3 dimensi:

```
NO2.shape = (n_hari, 9, 8)
```

Contoh data mengandung missing value (`--`):

```
[2.9651806e-05  4.1052295e-05  --  5.6563803e-05  --  --  6.3487379e-05  --]
```

---

### b. Mengatasi Missing Value menggunakan Interpolasi Linear

Missing value pada tiap grid spasial diatasi dengan **Interpolasi Linear** per cell grid:

```python
import numpy as np
import pandas as pd

# Inisialisasi array kosong
no2_filled = np.zeros_like(no2)
no2_filled = no2_filled.filled(0)

# Loop tiap grid (y, x)
for i in range(no2.shape[1]):     # 9 baris
    for j in range(no2.shape[2]): # 8 kolom
        series = pd.Series(no2[:, i, j])
        no2_filled[:, i, j] = series.interpolate(
            method='linear', limit_direction='both'
        ).to_numpy()
```

---

### c. Rata-ratakan Data dan Ubah Datetime

Setelah missing value diatasi, data NO₂ di-rata-rata per hari agar satu record hanya memiliki satu nilai. Format datetime juga diubah dari `2023-10-04 00:00:00` menjadi `2023-10-04`.

```python
new_dates = []
new_no2 = []

for i in range(len(dates)):
    new_date = dates[i].strftime('%Y-%m-%d')
    new_dates.append(new_date)
    new_no2.append(np.mean(no2_filled[i]))
```

---

### d. Simpan Data dalam Bentuk CSV

```python
df = pd.DataFrame({
    "date": new_dates,
    "NO2": new_no2
})

# Simpan ke CSV
df.to_csv("NO2_Nganjuk_timeseries.csv", index=False)
```

Contoh hasil CSV (`NO2_Nganjuk_timeseries.csv`):

```
        date       NO2
0  10/1/2023  0.000030
1  10/2/2023  0.000031
2  10/3/2023  0.000032
3  10/4/2023  0.000046
4  10/5/2023  0.000039
```

Total: **968 baris** data.

---

### e. Pengecekan Missing Value Data Harian pada CSV

Setelah data berbentuk CSV, dicek apakah Time Series harian sudah lengkap untuk rentang **2023-10-01 s/d 2026-06-01**:

```{code-cell}
import pandas as pd

df = pd.read_csv("../data/NO2_Nganjuk_timeseries.csv")
df['date'] = pd.to_datetime(df['date'])

# Buat rentang tanggal lengkap
start_date = "2023-10-01"
end_date   = "2026-06-01"
full_range = pd.date_range(start=start_date, end=end_date, freq='D')

# Cek tanggal yang hilang
missing_dates = full_range.difference(df['date'])

print(f"Jumlah hari missing: {len(missing_dates)}")
print("Daftar tanggal missing:")
print(missing_dates)
```

Apabila ditemukan missing value harian, atasi dengan interpolasi berbasis waktu:

```{code-cell}
df = df.sort_values('date')
full_range = pd.date_range(start="2023-10-01", end="2026-06-01", freq='D')

df = df.set_index('date').reindex(full_range)
df.index.name = 'date'

# Interpolasi linear berbasis waktu
df['NO2'] = df['NO2'].interpolate(method='time')

# Isi sisa NaN di ujung data
df['NO2'] = df['NO2'].bfill().ffill()

df.to_csv("../data/no2_timeseries_interpolated.csv")
```

dataset akhir setelah interpolasi adalah `no2_timeseries_interpolated.csv`


```{code-cell}
import pandas as pd

df = pd.read_csv("../data/no2_timeseries_interpolated.csv")
df['date'] = pd.to_datetime(df['date'])

# Buat rentang tanggal lengkap
start_date = "2023-10-01"
end_date   = "2026-06-01"
full_range = pd.date_range(start=start_date, end=end_date, freq='D')

# Cek tanggal yang hilang
missing_dates = full_range.difference(df['date'])

print(f"Jumlah hari missing: {len(missing_dates)}")
print("Daftar tanggal missing:")
print(missing_dates)
```

---

### f. Deteksi dan Penanganan Outlier (IQR)

Deteksi outlier menggunakan metode **Interquartile Range (IQR)**:

```{code-cell}
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv("../data/no2_timeseries_interpolated.csv")
df['date'] = pd.to_datetime(df['date'])

# Hitung IQR
Q1 = df['NO2'].quantile(0.25)
Q3 = df['NO2'].quantile(0.75)
IQR = Q3 - Q1

lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

# Filter outlier
outliers_iqr = df[(df['NO2'] < lower_bound) | (df['NO2'] > upper_bound)]

print("Jumlah Outlier (IQR):", len(outliers_iqr))
print(outliers_iqr[['date', 'NO2']].head())
```

Visualisasi outlier:

```{code-cell}
plt.figure(figsize=(15,5))
plt.plot(df['date'], df['NO2'], label="NO2", linewidth=1)

plt.scatter(outliers_iqr['date'], outliers_iqr['NO2'],
            color='red', marker='o', label="Outliers")

plt.axhline(upper_bound, color='orange', linestyle='dashed', label="Upper Bound (IQR)")
plt.axhline(lower_bound, color='blue',   linestyle='dashed', label="Lower Bound (IQR)")

plt.title("Deteksi Outlier Data NO2 (Metode IQR)")
plt.xlabel("Tanggal")
plt.ylabel("Kadar NO2")
plt.legend()
plt.tight_layout()
plt.xticks(
    ticks=[df['date'].iloc[0], df['date'].iloc[-1]],
    labels=[df['date'].iloc[0].strftime('%Y-%m-%d'),
            df['date'].iloc[-1].strftime('%Y-%m-%d')]
)
plt.show()
```

Setelah outlier terdeteksi, nilai outlier dihapus lalu diisi kembali menggunakan **Interpolasi Linear**:

```{code-cell}
# Tandai outlier menjadi NaN
df['NO2_cleaned'] = df['NO2'].mask(
    (df['NO2'] < lower_bound) | (df['NO2'] > upper_bound)
)

print("Jumlah nilai outlier:", df['NO2_cleaned'].isna().sum())

# Interpolasi linear
df['NO2_filled'] = df['NO2_cleaned'].interpolate(method='linear')
df['NO2_filled'] = df['NO2_filled'].bfill().ffill()

print("Jumlah missing setelah interpolasi:", df['NO2_filled'].isna().sum())
```

Visualisasi data setelah penanganan outlier:

```{code-cell}
plt.figure(figsize=(15,5))
plt.plot(df['date'], df['NO2_filled'], label="NO2 (Interpolated)", linewidth=1)
plt.xticks(
    ticks=[df['date'].iloc[0], df['date'].iloc[-1]],
    labels=[df['date'].iloc[0].strftime('%Y-%m-%d'),
            df['date'].iloc[-1].strftime('%Y-%m-%d')]
)
plt.title("Plot Data NO2 Setelah Outlier Removal & Interpolasi")
plt.xlabel("Tanggal")
plt.ylabel("Kadar NO2")
plt.legend()
plt.tight_layout()
plt.show()
```

---

## 3. Modeling menggunakan KNN Regression

Dengan data Time Series kadar NO₂ harian di daerah Nganjuk, tujuan modeling adalah **memprediksi kadar NO₂ satu hari ke depan** berdasarkan nilai hari-hari sebelumnya.

### a. Normalisasi Data

Karena menggunakan model KNN Regression yang sensitif terhadap skala data, normalisasi dilakukan menggunakan **MinMaxScaler (0–1)**:

```{code-cell}
from sklearn.preprocessing import MinMaxScaler
import pandas as pd

scaler = MinMaxScaler()
df['NO2_scaled'] = scaler.fit_transform(df[['NO2_filled']])
```

Contoh hasil normalisasi:

```
         date       NO2  NO2_scaled
0  2023-10-01  0.000030    0.305...
1  2023-10-02  0.000031    0.318...
2  2023-10-03  0.000032    0.325...
3  2023-10-04  0.000046    0.476...
4  2023-10-05  0.000038    0.390...
```

---

### b. Uji Korelasi Data

Data time series diubah menjadi format **supervised** terlebih dahulu dengan fitur lag (t-1 hingga t-30), lalu dihitung korelasinya terhadap label (t):

```{code-cell}
import pandas as pd

def create_supervised(data, n_lag=4):
    df_supervised = pd.DataFrame()

    for i in range(n_lag, 0, -1):
        df_supervised[f'NO2(t-{i})'] = data.shift(i)

    df_supervised['NO2(t)'] = data
    df_supervised.dropna(inplace=True)

    return df_supervised

supervised_df30 = create_supervised(df['NO2_scaled'], n_lag=30)

lag_cols     = supervised_df30.drop(columns="NO2(t)").columns
correlations = supervised_df30[lag_cols].corrwith(supervised_df30['NO2(t)'])

print(correlations)
```

Hasil korelasi (nilai > 0.5 dianggap signifikan):

Dari hasil ini, fitur **t-1 hingga t-11** memiliki korelasi di atas 0.5 terhadap label. Nilai korelasi tertinggi ada pada **t-1 (0.8240)**, menunjukkan bahwa nilai NO₂ kemarin adalah prediktor terkuat untuk nilai hari ini.

---

### c. Mengubah Data menjadi Supervised

Berdasarkan uji korelasi, dibuat tiga variasi data supervised untuk dibandingkan:

```{code-cell}
# 4 hari sebelumnya
supervised_df4  = create_supervised(df['NO2_scaled'], n_lag=4)

# 10 hari sebelumnya
supervised_df10 = create_supervised(df['NO2_scaled'], n_lag=10)

# 30 hari sebelumnya
supervised_df30 = create_supervised(df['NO2_scaled'], n_lag=30)
```

Contoh bentuk data dengan `n_lag=4` (727 baris, 5 kolom):

```
     NO2(t-4)  NO2(t-3)  NO2(t-2)  NO2(t-1)    NO2(t)
4    0.305...  0.318...  0.325...  0.476...  0.390...
5    0.318...  0.325...  0.476...  0.390...  0.421...
...
(727, 5)
```

---

### d. Modeling dan Evaluasi

Masing-masing data dilatih menggunakan **KNN Regression** dengan `n_neighbors=5` dan split data **80% train / 20% test**:

```{code-cell}
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

def MAPE(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    nonzero = y_true != 0
    return np.mean(np.abs((y_true[nonzero] - y_pred[nonzero]) / y_true[nonzero])) * 100

def train_knn(df_supervised, model_name=""):
    X = df_supervised.drop(columns=['NO2(t)']).values
    y = df_supervised['NO2(t)'].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=False
    )

    knn = KNeighborsRegressor(n_neighbors=5)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)

    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2   = r2_score(y_test, y_pred)
    mape = MAPE(y_test, y_pred)

    print(f"\n=== {model_name} ===")
    print(f"Train Size: {len(X_train)} — Test Size: {len(X_test)}")
    print(f"RMSE : {rmse:.6f}")
    print(f"R²   : {r2:.4f}")
    print(f"MAPE : {mape:.4f}%")

    return knn, y_test, y_pred

knn_4,  y_test_4,  y_pred_4  = train_knn(supervised_df4,  "KNN - 4 Hari Sebelumnya")
knn_10, y_test_10, y_pred_10 = train_knn(supervised_df10, "KNN - 10 Hari Sebelumnya")
knn_30, y_test_30, y_pred_30 = train_knn(supervised_df30, "KNN - 30 Hari Sebelumnya")
```

---

### e. Plotting

Visualisasi perbandingan nilai aktual vs prediksi untuk masing-masing model:

```{code-cell}
import matplotlib.pyplot as plt
import numpy as np

# 4 hari sebelumnya
plt.figure()
plt.plot(np.arange(len(y_test_4)), y_test_4,  label="Actual")
plt.plot(np.arange(len(y_pred_4)), y_pred_4,  label="Predicted")
plt.title("KNN Regression - 4 Hari Sebelumnya")
plt.xlabel("Sample Index")
plt.ylabel("NO2 Value (Normalized)")
plt.legend()
plt.show()

# 10 hari sebelumnya
plt.figure()
plt.plot(np.arange(len(y_test_10)), y_test_10, label="Actual")
plt.plot(np.arange(len(y_pred_10)), y_pred_10, label="Predicted")
plt.title("KNN Regression - 10 Hari Sebelumnya")
plt.xlabel("Sample Index")
plt.ylabel("NO2 Value (Normalized)")
plt.legend()
plt.show()

# 30 hari sebelumnya
plt.figure()
plt.plot(np.arange(len(y_test_30)), y_test_30, label="Actual")
plt.plot(np.arange(len(y_pred_30)), y_pred_30, label="Predicted")
plt.title("KNN Regression - 30 Hari Sebelumnya")
plt.xlabel("Sample Index")
plt.ylabel("NO2 Value (Normalized)")
plt.legend()
plt.show()
```

---
