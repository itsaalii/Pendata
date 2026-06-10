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
# Tugas — skforecast


Sumber: [skforecast.org/0.15.1/user_guides/explainability.html](https://skforecast.org/0.15.1/user_guides/explainability.html)

---

## Import Library

```{code-cell}
# Libraries
# ==============================================================================
import pandas as pd
import matplotlib.pyplot as plt
import shap
from sklearn.inspection import permutation_importance
from sklearn.inspection import PartialDependenceDisplay
from lightgbm import LGBMRegressor
from skforecast.datasets import fetch_dataset
from skforecast.recursive import ForecasterRecursive

shap.initjs()
```

`shap.initjs()` harus dipanggil di awal notebook untuk memuat JavaScript library yang dibutuhkan agar visualisasi interaktif SHAP (seperti `force_plot`) dapat ditampilkan dengan benar di lingkungan Jupyter Notebook.

---

## Load dan Siapkan Data

Data yang digunakan diperoleh dari paket R `tsibbledata`. Dataset berisi **52.608 record** dengan frekuensi setengah-jam dan 5 kolom:

- `Time` — tanggal dan waktu pencatatan
- `Date` — tanggal pencatatan
- `Demand` — permintaan listrik (MW)
- `Temperature` — suhu di Melbourne, ibu kota Victoria
- `Holiday` — indikator hari libur umum

```{code-cell}
# Download data
# ==============================================================================
data = fetch_dataset(name="vic_electricity")
data.head(3)
```

```{code-cell}
# Aggregation to daily frequency
# ==============================================================================
data = data.resample('D').agg({'Demand': 'sum', 'Temperature': 'mean'})
data.head(3)
```

Data asli berformat setengah-jam diagregasi ke frekuensi harian:
- Kolom `Demand` dijumlahkan (`sum`) untuk mendapatkan total permintaan harian
- Kolom `Temperature` dirata-ratakan (`mean`) untuk mendapatkan suhu rata-rata harian

```{code-cell}
# Split train-test
# ==============================================================================
data_train = data.loc[: '2014-12-21']
data_test  = data.loc['2014-12-22':]

print(f"Tanggal train : {data_train.index.min()} — {data_train.index.max()}")
print(f"Tanggal test  : {data_test.index.min()} — {data_test.index.max()}")
print(f"Shape train   : {data_train.shape}")
print(f"Shape test    : {data_test.shape}")
```

---

## Buat dan Latih Forecaster

Model forecasting dibuat untuk memprediksi permintaan energi menggunakan 7 nilai masa lalu (seminggu terakhir) dan suhu sebagai variabel eksogen.

```{code-cell}
# Create a recursive multi-step forecaster (ForecasterRecursive)
# ==============================================================================
forecaster = ForecasterRecursive(
    estimator = LGBMRegressor(random_state=123, verbose=-1),
    lags      = 7
)

forecaster.fit(
    y    = data_train['Demand'],
    exog = data_train['Temperature']
)

forecaster
```

Penjelasan parameter:

- `lags=7` — model menggunakan 7 nilai historis (t-1 s.d. t-7) sebagai fitur prediktor
- `exog=Temperature` — suhu harian digunakan sebagai variabel eksogen (fitur tambahan)
- `LGBMRegressor` — model gradient boosting berbasis pohon dari library LightGBM

---

## Feature Importance

Feature importance digunakan untuk mengidentifikasi fitur-fitur yang paling relevan terhadap prediksi, memahami perilaku model, dan memilih kumpulan fitur terbaik. Penting untuk diingat bahwa feature importance bukan ukuran kausalitas — pentingnya sebuah fitur tidak berarti fitur tersebut menyebabkan outcome.

### a. Model-Specific Feature Importance

Cara menghitung feature importance bergantung pada jenis model:
- **Decision tree-based** (Random Forest, Gradient Boosting): menggunakan *mean decrease impurity*
- **Linear model** (Ridge, Lasso): menggunakan koefisien atau koefisien yang dinormalisasi

Metode `get_feature_importances()` mengakses atribut `coef_` atau `feature_importances_` dari regressor internal.

```{code-cell}
# Feature importances
# ==============================================================================
feature_importances = forecaster.get_feature_importances()
feature_importances
```

```{code-cell}
# Plot feature importances
# ==============================================================================
fig, ax = plt.subplots(figsize=(5, 4))
feature_importances.sort_values('importance', ascending=True).plot(
    x      = 'feature',
    y      = 'importance',
    kind   = 'barh',
    ax     = ax,
    legend = False
)
ax.set_title('Feature Importance (LightGBM — mean decrease impurity)')
ax.set_xlabel('Importance')
ax.set_ylabel('Feature')
plt.tight_layout()
plt.show()
```

### b. Permutation Importance

Permutation importance adalah teknik inspeksi model yang mengukur kontribusi tiap fitur terhadap performa statistik model. Teknik ini mengacak nilai satu fitur, memutus hubungannya dengan target variable, lalu mengamati penurunan skor model. Teknik ini sangat berguna untuk model non-linear atau *opaque*.

Untuk menerapkan permutation importance pada skforecast, diperlukan **matriks training** yang sama dengan yang digunakan model untuk dilatih. Matriks ini diperoleh dengan metode `create_train_X_y()`.

```{code-cell}
# Training matrices used by the forecaster to fit the internal regressor
# ==============================================================================
X_train, y_train = forecaster.create_train_X_y(
    y    = data_train['Demand'],
    exog = data_train['Temperature']
)

display(X_train.head(3))  # Features (lag_1 ... lag_7, Temperature)
display(y_train.head(3))  # Target
```

```{code-cell}
# Permutation importance
# ==============================================================================
result = permutation_importance(
    estimator    = forecaster.estimator,
    X            = X_train,
    y            = y_train,
    n_repeats    = 10,
    random_state = 123,
    n_jobs       = -1
)

permutation_imp_df = pd.DataFrame({
    'feature'         : X_train.columns,
    'importance_mean' : result.importances_mean,
    'importance_std'  : result.importances_std
}).sort_values('importance_mean', ascending=False)

permutation_imp_df
```

```{code-cell}
# Plot permutation importance
# ==============================================================================
sorted_idx = result.importances_mean.argsort()

fig, ax = plt.subplots(figsize=(6, 5))
ax.boxplot(
    result.importances[sorted_idx].T,
    vert   = False,
    labels = X_train.columns[sorted_idx]
)
ax.set_title('Permutation Feature Importance')
ax.set_xlabel('Penurunan performa (skor model)')
ax.axvline(x=0, color='grey', linestyle='--')
plt.tight_layout()
plt.show()
```

---

## Partial Dependence Plot (PDP)

Partial Dependence Plot (PDP) adalah teknik inspeksi model yang menampilkan **rata-rata respons model** saat nilai satu fitur divariasikan di seluruh rentangnya, sementara fitur lainnya dipertahankan tetap (dimarginalkan). PDP melengkapi informasi dari feature importance: feature importance menunjukkan *seberapa penting* sebuah fitur, sedangkan PDP menunjukkan *bagaimana bentuk hubungannya* dengan target.

```{code-cell}
# Partial Dependence Plots
# ==============================================================================
features_to_plot = ['lag_1', 'lag_2', 'lag_3', 'lag_4',
                    'lag_5', 'lag_6', 'lag_7', 'Temperature']

fig, axes = plt.subplots(
    nrows   = 2,
    ncols   = 4,
    figsize = (14, 7),
    sharey  = False
)
axes = axes.flatten()

for i, feature in enumerate(features_to_plot):
    PartialDependenceDisplay.from_estimator(
        estimator  = forecaster.estimator,
        X          = X_train,
        features   = [feature],
        kind       = 'average',
        ax         = axes[i]
    )
    axes[i].set_title(f'PDP — {feature}')

plt.suptitle('Partial Dependence Plots', fontsize=14, y=1.02)
plt.tight_layout()
plt.show()
```

Interpretasi PDP:
- **Kurva naik** — nilai fitur yang lebih tinggi mendorong prediksi `Demand` ke atas
- **Kurva turun** — nilai fitur yang lebih tinggi mendorong prediksi ke bawah
- **Kurva datar** — fitur tidak terlalu berpengaruh pada rentang nilai tersebut
- **Kurva non-linear** — terdapat hubungan yang kompleks antara fitur dan target

---

## SHAP Values

SHAP (SHapley Additive exPlanations) adalah metode yang paling banyak digunakan untuk menjelaskan model machine learning. SHAP memberikan wawasan visual dan kuantitatif tentang bagaimana fitur dan nilainya mempengaruhi prediksi model.

SHAP melayani dua tujuan utama:

- **Global Interpretability** — mengidentifikasi fitur mana yang paling berpengaruh pada model selama training dengan merata-ratakan SHAP values di seluruh dataset
- **Local Interpretability** — menjelaskan prediksi individual dengan menunjukkan seberapa besar kontribusi tiap fitur terhadap output spesifik

Implementasi Python SHAP dibangun berdasarkan *explainers*. Setiap explainer hanya sesuai untuk kelas algoritma tertentu:
- `shap.TreeExplainer` — untuk model berbasis pohon (LightGBM, XGBoost, Random Forest)
- `shap.LinearExplainer` — untuk model linear
- `shap.KernelExplainer` — model-agnostic, lebih lambat

Untuk menghasilkan penjelasan SHAP dari model skforecast diperlukan dua komponen:
1. **Internal regressor** dari forecaster (`forecaster.regressor`)
2. **Matriks training** yang dibuat dari time series (`create_train_X_y()`)

### a. SHAP pada Data Training

```{code-cell}
# Training matrices used by the forecaster to fit the internal regressor
# ==============================================================================
X_train, y_train = forecaster.create_train_X_y(
    y    = data_train['Demand'],
    exog = data_train['Temperature']
)

display(X_train.head(3))
display(y_train.head(3))
```

```{code-cell}
# Create SHAP explainer
# ==============================================================================
explainer   = shap.TreeExplainer(forecaster.estimator)
shap_values = explainer.shap_values(X_train)

print(f"Shape SHAP values  : {shap_values.shape}")
print(f"Expected value     : {explainer.expected_value:.2f}")
print(f"Shape X_train      : {X_train.shape}")
```

```{code-cell}
# SHAP summary plot — bar (global feature importance)
# ==============================================================================
shap.summary_plot(
    shap_values,
    X_train,
    plot_type = 'bar',
    show      = True
)
```

SHAP summary plot (bar) menampilkan rata-rata absolut dari SHAP values untuk setiap fitur. Fitur dengan nilai tertinggi adalah yang paling berpengaruh terhadap output model secara keseluruhan.

```{code-cell}
# SHAP summary plot — dot (distribusi kontribusi per observasi)
# ==============================================================================
shap.summary_plot(
    shap_values,
    X_train,
    show = True
)
```

SHAP summary plot (dot): setiap titik mewakili satu observasi. Warna merah berarti nilai fitur tinggi, warna biru berarti nilai fitur rendah. Posisi titik pada sumbu X menunjukkan besar kontribusi SHAP — ke kanan berarti mendorong prediksi naik, ke kiri berarti mendorong prediksi turun.

```{code-cell}
# SHAP dependence plot untuk fitur Temperature
# ==============================================================================
fig, ax = plt.subplots(figsize=(7, 4))
shap.dependence_plot(
    "Temperature",
    shap_values,
    X_train,
    ax = ax
)
plt.title('SHAP Dependence Plot — Temperature')
plt.tight_layout()
plt.show()
```

SHAP dependence plot memperlihatkan hubungan antara nilai satu fitur (sumbu X) dengan nilai SHAP-nya (sumbu Y). Titik-titik diwarnai berdasarkan fitur lain yang paling berinteraksi, sehingga efek interaksi antar fitur juga terlihat.

```{code-cell}
# SHAP force plot — penjelasan untuk satu observasi (observasi pertama)
# ==============================================================================
shap.force_plot(
    base_value  = explainer.expected_value,
    shap_values = shap_values[0, :],
    features    = X_train.iloc[0, :]
)
```

```{code-cell}
# ==============================================================================
shap.force_plot(explainer.expected_value, shap_values[:200, :], X_train.iloc[:200, :])
```

SHAP force plot menampilkan penjelasan interaktif untuk satu prediksi individual:
- **Angka di tengah** — nilai prediksi untuk observasi tersebut
- **`base value`** — rata-rata prediksi model di seluruh training data (`expected_value`)
- **Panah merah** — fitur yang mendorong prediksi **naik** dari baseline
- **Panah biru** — fitur yang mendorong prediksi **turun** dari baseline
- **Lebar panah** — besarnya kontribusi fitur tersebut

### b. SHAP pada Data Prediksi

Selain menjelaskan model saat training, SHAP juga dapat digunakan untuk menjelaskan nilai yang di-forecast. Caranya adalah menggunakan matriks input yang dipakai secara internal oleh metode `predict()`.

```{code-cell}
# Predict
# ==============================================================================
predictions = forecaster.predict(
    steps = 10,
    exog  = data_test['Temperature']
)

predictions
```

```{code-cell}
# Create input matrix for predict method
# ==============================================================================
X_predict = forecaster.create_predict_X(
    steps = 10,
    exog  = data_test['Temperature']
)

X_predict
```

`create_predict_X()` mengembalikan matriks fitur yang persis sama dengan yang digunakan `forecaster.predict()` secara internal — berisi lag values dan variabel eksogen untuk setiap langkah prediksi.

```{code-cell}
# SHAP values untuk data prediksi
# ==============================================================================
shap_values_pred = explainer.shap_values(X_predict)

print(f"Shape SHAP values prediksi : {shap_values_pred.shape}")
```

```{code-cell}
# SHAP summary plot untuk data prediksi
# ==============================================================================
shap.summary_plot(
    shap_values_pred,
    X_predict,
    show = True
)
```

```{code-cell}
# SHAP force plot untuk langkah prediksi pertama
# ==============================================================================
shap.force_plot(
    base_value  = explainer.expected_value,
    shap_values = shap_values_pred[0, :],
    features    = X_predict.iloc[0, :]
)
```

Dengan SHAP pada data prediksi, kita dapat menjawab pertanyaan: **mengapa model memprediksi nilai X pada tanggal Y?** Setiap lag dan variabel eksogen yang digunakan forecaster untuk menghasilkan prediksi dapat dijelaskan kontribusinya secara individual.

---

## Kesimpulan
1. Analisa Prediksi

Model ini memprediksi total permintaan listrik harian (Demand) di Victoria, Australia dalam satuan megawatt (MW)

2. Bentuk Data Training

Setelah `resample('D')` dan `create_train_X_y()`, setiap baris data training berbentuk seperti ini:

| lag_1 | lag_2 | lag_3 | lag_4 | lag_5 | lag_6 | lag_7 | Tempreature | Demand |
| :----: | :----: | :----: | :----: | :----: | :----: | :----: | :---------: | :----: |
| 208000 | 210000 | 198000 | 205000 | 215000 | 200000 | 212000 |    22.1     | 195000 |
| 195000 | 208000 | 210000 | 198000 | 205000 | 215000 | 200000 |    18.5     | 221000 |

*input (fitur x)*

- `lag_1` sampai `lag_7`, yaitu demand 1 hingga 7 hari sebelumnya

- `tempreature` yaitu suhu rata-rata hari yang akan diprediksi


*Output (fitur Y)*

- *Demand*, yaitu total permintaan listrik yang ingin diprediksi

3. Apa itu lag?

Lag adalah nilai historis dari target variable itu sendiri yang dijadikan fitur input