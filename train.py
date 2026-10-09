"""Pipeline end-to-end prediksi biaya asuransi (regression).

Alur: load data -> EDA -> preprocessing -> training & tuning banyak model
      -> evaluasi -> pilih model terbaik -> simpan model + grafik + metrik.

Jalankan:  python train.py
"""
import json
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")  # simpan gambar ke file tanpa membuka jendela
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import sklearn
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, cross_val_score, train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor

from features import add_features

# =====================================================================
# KONFIGURASI
# =====================================================================
DATA_PATH = Path("data/insurance.csv")
OUTPUT_DIR = Path("outputs")
FIG_DIR = OUTPUT_DIR / "figures"
TARGET = "charges"
DROP_COLS = ["region", "sex"]  # korelasi sangat lemah terhadap charges (lihat heatmap)
TEST_SIZE = 0.2
RANDOM_STATE = 0
CV_FOLDS = 5
N_JOBS = -1

sns.set_theme(style="whitegrid")


# =====================================================================
# 1. LOAD DATA
# =====================================================================
def load_data() -> pd.DataFrame:
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"File {DATA_PATH} tidak ditemukan. Download 'insurance.csv' dari Kaggle "
            "(link ada di README) lalu taruh di folder data/."
        )
    df = pd.read_csv(DATA_PATH)
    required = {"age", "sex", "bmi", "children", "smoker", "region", TARGET}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Kolom berikut tidak ada di dataset: {sorted(missing)}")
    return df


# =====================================================================
# 2. EDA (grafik disimpan ke outputs/figures)
# =====================================================================
def make_eda_figures(df: pd.DataFrame) -> None:
    # Korelasi: semua kolom di-encode ke angka dulu (hanya untuk keperluan EDA)
    enc = df.copy()
    enc["sex"] = enc["sex"].map({"female": 0, "male": 1})
    enc["smoker"] = enc["smoker"].map({"no": 0, "yes": 1})
    enc["region"] = enc["region"].astype("category").cat.codes

    plt.figure(figsize=(8, 6))
    sns.heatmap(enc.corr(), annot=True, cmap="coolwarm", fmt=".2f")
    plt.title("Correlation Heatmap")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "correlation_heatmap.png", dpi=150)
    plt.close()

    # Insight utama: efek obesitas jauh lebih besar pada perokok
    tmp = df.copy()
    tmp["Kelompok BMI"] = np.where(tmp["bmi"] >= 30, "Obesitas (BMI >= 30)", "Non-obesitas")
    tmp["Perokok"] = tmp["smoker"].map({"yes": "Perokok", "no": "Bukan perokok"})
    plt.figure(figsize=(7, 5))
    ax = sns.barplot(data=tmp, x="Perokok", y=TARGET, hue="Kelompok BMI", errorbar=None)
    for container in ax.containers:
        ax.bar_label(container, fmt="%.0f", padding=2)
    plt.title("Rata-rata Biaya Asuransi: Status Merokok x Obesitas")
    plt.ylabel("Rata-rata charges")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "charges_smoker_obesity.png", dpi=150)
    plt.close()


# =====================================================================
# 3. PREPROCESSING
# =====================================================================
def prepare(df: pd.DataFrame):
    df = df.drop(columns=DROP_COLS)
    df["smoker"] = df["smoker"].map({"no": 0, "yes": 1})
    if df["smoker"].isna().any():
        raise ValueError("Kolom 'smoker' berisi nilai selain 'yes'/'no'.")
    return df.drop(columns=TARGET), df[TARGET]


# =====================================================================
# 4. DEFINISI MODEL & GRID TUNING
# =====================================================================
def build_models() -> dict:
    """Mengembalikan {nama: (estimator, grid_hyperparameter atau None)}."""
    scaled = lambda m: Pipeline([("scaler", StandardScaler()), ("model", m)])
    return {
        "Linear Regression": (LinearRegression(), None),
        "Ridge Regression": (Ridge(alpha=1.0), None),
        "Lasso Regression": (Lasso(alpha=1.0, max_iter=10000), None),
        "Linear Regression + Feature Engineering": (
            Pipeline([
                ("features", FunctionTransformer(add_features)),
                ("model", LinearRegression()),
            ]),
            None,
        ),
        "Decision Tree": (
            DecisionTreeRegressor(random_state=RANDOM_STATE),
            {"max_depth": [4, 5, 6], "min_samples_split": [2, 4, 6], "min_samples_leaf": [1, 2, 3]},
        ),
        "Random Forest": (
            RandomForestRegressor(random_state=RANDOM_STATE),
            {"n_estimators": [100, 200], "max_depth": [4, 6, 8], "min_samples_leaf": [1, 3]},
        ),
        "Gradient Boosting": (
            GradientBoostingRegressor(random_state=RANDOM_STATE),
            {"n_estimators": [100, 200], "max_depth": [2, 3, 4], "learning_rate": [0.05, 0.1]},
        ),
        "KNN": (scaled(KNeighborsRegressor()), {"model__n_neighbors": [3, 5, 7, 9, 11]}),
        "SVR": (
            scaled(SVR()),
            {"model__C": [1000, 10000, 50000], "model__gamma": ["scale", 0.1], "model__kernel": ["rbf"]},
        ),
    }


# =====================================================================
# 5. EVALUASI
# =====================================================================
def evaluate(name, model, params, X_train, X_test, y_train, y_test) -> dict:
    pred = model.predict(X_test)
    mse = mean_squared_error(y_test, pred)
    cv_r2 = cross_val_score(model, X_train, y_train, cv=CV_FOLDS, scoring="r2", n_jobs=N_JOBS)
    return {
        "Model": name,
        "CV R2 (mean)": cv_r2.mean(),
        "CV R2 (std)": cv_r2.std(),
        "Test R2": r2_score(y_test, pred),
        "Test RMSE": np.sqrt(mse),
        "Test MAE": mean_absolute_error(y_test, pred),
        "Best Params": json.dumps(params),
    }


# =====================================================================
# 6. GRAFIK HASIL
# =====================================================================
def plot_model_comparison(results: pd.DataFrame) -> None:
    data = results.sort_values("CV R2 (mean)")
    y = np.arange(len(data))
    plt.figure(figsize=(9, 5.5))
    plt.barh(y - 0.2, data["CV R2 (mean)"], height=0.4, label="CV R2 (train, 5-fold)")
    plt.barh(y + 0.2, data["Test R2"], height=0.4, label="Test R2")
    plt.yticks(y, data["Model"])
    plt.xlim(0.7, 0.95)
    plt.xlabel("R2 Score")
    plt.title("Perbandingan Model")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "model_comparison.png", dpi=150)
    plt.close()


def plot_actual_vs_predicted(name, model, X_test, y_test) -> None:
    pred = model.predict(X_test)
    plt.figure(figsize=(6, 6))
    plt.scatter(y_test, pred, alpha=0.5)
    lims = [min(y_test.min(), pred.min()), max(y_test.max(), pred.max())]
    plt.plot(lims, lims, "r--", label="Prediksi sempurna")
    plt.xlabel("Actual charges")
    plt.ylabel("Predicted charges")
    plt.title(f"Actual vs Predicted - {name}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIG_DIR / "actual_vs_predicted.png", dpi=150)
    plt.close()


def plot_permutation_importance(name, model, X_test, y_test) -> pd.DataFrame:
    """Permutation importance: model-agnostic, berlaku untuk model apa pun."""
    r = permutation_importance(
        model, X_test, y_test, scoring="r2", n_repeats=20, random_state=RANDOM_STATE
    )
    imp = (
        pd.DataFrame({"Feature": X_test.columns, "Importance": r.importances_mean, "Std": r.importances_std})
        .sort_values("Importance", ascending=False)
    )
    plt.figure(figsize=(7, 4))
    sns.barplot(data=imp, x="Importance", y="Feature", color="#4C72B0")
    plt.title(f"Permutation Importance - {name}")
    plt.xlabel("Penurunan R2 saat fitur diacak")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "feature_importance.png", dpi=150)
    plt.close()
    return imp


# =====================================================================
# 7. MAIN
# =====================================================================
def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    df = load_data()
    print(f"Dataset: {df.shape[0]} baris, {df.shape[1]} kolom | missing value: {int(df.isna().sum().sum())}")
    make_eda_figures(df)

    X, y = prepare(df)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )
    print(f"Fitur: {list(X.columns)} | train={len(X_train)}, test={len(X_test)}\n")

    rows, fitted = [], {}
    for name, (estimator, grid) in build_models().items():
        print(f"Training: {name}")
        if grid:
            gs = GridSearchCV(estimator, grid, cv=CV_FOLDS, scoring="neg_mean_squared_error", n_jobs=N_JOBS)
            gs.fit(X_train, y_train)
            model, params = gs.best_estimator_, gs.best_params_
        else:
            model, params = estimator.fit(X_train, y_train), {}
        fitted[name] = model
        rows.append(evaluate(name, model, params, X_train, X_test, y_train, y_test))

    # Model terbaik dipilih berdasarkan CV R2 pada data TRAIN (bukan data test),
    # supaya data test tetap menjadi ukuran performa yang jujur.
    results = pd.DataFrame(rows).sort_values("CV R2 (mean)", ascending=False).reset_index(drop=True)
    best_name = results.loc[0, "Model"]
    best_model = fitted[best_name]

    results.to_csv(OUTPUT_DIR / "model_comparison.csv", index=False)
    plot_model_comparison(results)
    plot_actual_vs_predicted(best_name, best_model, X_test, y_test)
    imp = plot_permutation_importance(best_name, best_model, X_test, y_test)

    joblib.dump(best_model, OUTPUT_DIR / "best_model.joblib")
    meta = {
        "best_model": best_name,
        "features": list(X.columns),
        "smoker_encoding": {"no": 0, "yes": 1},
        "test_metrics": {
            "r2": float(results.loc[0, "Test R2"]),
            "rmse": float(results.loc[0, "Test RMSE"]),
            "mae": float(results.loc[0, "Test MAE"]),
        },
        "sklearn_version": sklearn.__version__,
    }
    (OUTPUT_DIR / "model_metadata.json").write_text(json.dumps(meta, indent=2))

    pd.set_option("display.width", 220)
    pd.set_option("display.max_colwidth", 70)
    print("\n===== HASIL (diurutkan berdasarkan CV R2) =====")
    print(results.drop(columns="Best Params").round(4).to_string(index=False))
    print(f"\nModel terbaik (berdasarkan CV R2): {best_name}")
    print("\nPermutation importance:")
    print(imp.round(4).to_string(index=False))
    print(f"\nModel & grafik tersimpan di folder '{OUTPUT_DIR}/'")


if __name__ == "__main__":
    main()
