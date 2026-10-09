"""Prediksi biaya asuransi untuk satu orang memakai model terlatih.

Contoh:
    python predict.py --age 35 --bmi 31.5 --children 2 --smoker yes

Jalankan `python train.py` dulu supaya outputs/best_model.joblib tersedia.
"""
import argparse
import json
from pathlib import Path

import joblib
import pandas as pd

import features  # noqa: F401  (diperlukan agar model hasil training bisa di-load)

MODEL_PATH = Path("outputs/best_model.joblib")
META_PATH = Path("outputs/model_metadata.json")


def main() -> None:
    parser = argparse.ArgumentParser(description="Prediksi biaya asuransi kesehatan")
    parser.add_argument("--age", type=int, required=True, help="Usia (tahun)")
    parser.add_argument("--bmi", type=float, required=True, help="Body Mass Index")
    parser.add_argument("--children", type=int, required=True, help="Jumlah anak/tanggungan")
    parser.add_argument("--smoker", choices=["yes", "no"], required=True, help="Perokok atau bukan")
    args = parser.parse_args()

    if not MODEL_PATH.exists():
        raise SystemExit("Model belum ada. Jalankan dulu: python train.py")

    model = joblib.load(MODEL_PATH)
    meta = json.loads(META_PATH.read_text())

    row = pd.DataFrame(
        [{
            "age": args.age,
            "bmi": args.bmi,
            "children": args.children,
            "smoker": meta["smoker_encoding"][args.smoker],
        }]
    )[meta["features"]]

    prediction = float(model.predict(row)[0])
    print(f"Model            : {meta['best_model']}")
    print(f"Estimasi charges : {prediction:,.2f}")
    print(f"(MAE model di data uji sekitar {meta['test_metrics']['mae']:,.0f}, "
          "jadi anggap ini perkiraan kasar, bukan angka pasti.)")


if __name__ == "__main__":
    main()
