"""Feature engineering untuk prediksi biaya asuransi.

Dipisah ke modul sendiri supaya fungsi yang sama dipakai saat training
(train.py) dan saat prediksi (predict.py), serta bisa di-pickle bersama model.
"""
import pandas as pd

OBESITY_BMI_THRESHOLD = 30  # BMI >= 30 dikategorikan obesitas (standar WHO)


def add_features(X: pd.DataFrame) -> pd.DataFrame:
    """Menambah fitur turunan untuk model linear.

    - obese          : 1 jika BMI >= 30
    - smoker_x_obese : interaksi perokok x obesitas (efek BMI jauh lebih besar pada perokok)
    - age2           : usia kuadrat (menangkap kenaikan biaya yang non-linear terhadap usia)
    """
    X = X.copy()
    X["obese"] = (X["bmi"] >= OBESITY_BMI_THRESHOLD).astype(int)
    X["smoker_x_obese"] = X["smoker"] * X["obese"]
    X["age2"] = X["age"] ** 2
    return X
