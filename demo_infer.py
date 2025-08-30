#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Demo: so sánh inference plaintext vs Homomorphic Encryption (Paillier)
- Load pipeline (LDA) đã train sẵn
- Lấy K mẫu từ test_features.npz
- Đo thời gian dự đoán plaintext
- Mã hoá vector (Paillier) và thực hiện dự đoán tuyến tính tương đương trong miền mã hoá:
    y = w·x + b  (chỉ dùng phần tuyến tính của LDA)
- Giải mã điểm số & so sánh với plaintext (sai số ~0)
- In thời gian / mẫu cho HE

Yêu cầu:
- Đã có các artifacts sau (ví dụ time-window baseline):
    featuresdir = inputs/SE001_timewin64
    modeldir    = runs/timewin64_cv
  hoặc dùng xDAWN:
    featuresdir = inputs/SE001_xdawn
    modeldir    = runs/xdawn_cv

- Thư viện: python-paillier (phe). Nếu thiếu: pip install "phe==1.5.0"
"""

import argparse
import time
import numpy as np
from pathlib import Path
import joblib

try:
    import phe as paillier
except Exception as e:
    paillier = None

def load_data_and_model(featuresdir: Path, modeldir: Path):
    te = np.load(featuresdir/"test_features.npz")
    Xte, yte = te["X"], te["y"]
    pipe = joblib.load(modeldir/"model_best.joblib")
    # Lấy scaler và LDA bên trong Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    scaler = None; lda = None
    try:
        # pipe = Pipeline([('scaler', ...), ('clf', ...)])
        scaler = pipe.named_steps.get("scaler", None)
        lda = pipe.named_steps.get("clf", None)
    except Exception:
        pass
    if scaler is None or lda is None:
        raise RuntimeError("Không tìm thấy scaler/clf trong Pipeline. Script này chỉ hỗ trợ Pipeline(['scaler','clf']).")
    if not isinstance(lda, LinearDiscriminantAnalysis):
        raise RuntimeError("Demo HE này chỉ hỗ trợ LDA (tuyến tính).")
    return Xte, yte, pipe, scaler, lda

def lda_linear_form(lda, X_std):
    """
    Trả về điểm tuyến tính g(x) ≈ w·x + b để so sánh plaintext vs HE.
    Với LDA 2 lớp, decision_function tương đương tuyến tính trên không gian chuẩn hoá.
    Ở đây ta lấy coef_ và intercept_ của 'clf' trong Pipeline sau khi fit.
    """
    # sklearn LDA có attributes coef_, intercept_ khi solver='svd' + predict_proba
    w = lda.coef_.reshape(-1)
    b = float(lda.intercept_.reshape(-1)[0])
    scores = X_std @ w + b
    return scores, w, b

def he_linear_scores(X_std, w, b, keybits=2048):
    """
    Tính y = w·x + b với Paillier HE.
    HE hỗ trợ cộng & nhân với số thường: Enc(sum_i w_i * x_i) = sum_i (Enc(x_i) * w_i)
    Quy ước: mã hoá x (dữ liệu), giữ w,b ở dạng thường (server side).
    """
    if paillier is None:
        raise RuntimeError("Thiếu thư viện 'phe'. Cài: pip install 'phe==1.5.0'")

    pub, priv = paillier.generate_paillier_keypair(n_length=keybits)

    he_times = []
    scores_dec = []

    for i in range(X_std.shape[0]):
        x = X_std[i]
        t0 = time.perf_counter()
        # mã hoá từng feature
        enc_x = [pub.encrypt(float(v)) for v in x]
        # w·x (w là số thường): sum_i (enc_x_i * w_i)
        enc_dot = None
        for xi_enc, wi in zip(enc_x, w):
            term = xi_enc * float(wi)
            enc_dot = term if enc_dot is None else enc_dot + term
        # + b
        enc_score = enc_dot + float(b)
        # giải mã
        score = priv.decrypt(enc_score)
        t1 = time.perf_counter()
        he_times.append(t1 - t0)
        scores_dec.append(score)
    return np.asarray(scores_dec), np.mean(he_times), np.std(he_times)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--featuresdir", type=str, default="inputs/SE001_timewin64",
                    help="Thư mục chứa test_features.npz")
    ap.add_argument("--modeldir", type=str, default="runs/timewin64_cv",
                    help="Thư mục chứa model_best.joblib")
    ap.add_argument("--K", type=int, default=8, help="Số mẫu test để demo")
    ap.add_argument("--keybits", type=int, default=2048, help="Độ dài khoá Paillier")
    args = ap.parse_args()

    featuresdir = Path(args.featuresdir); modeldir = Path(args.modeldir)
    Xte, yte, pipe, scaler, lda = load_data_and_model(featuresdir, modeldir)

    # Lấy K mẫu đầu để demo cho nhanh
    K = min(args.K, len(Xte))
    X = Xte[:K].copy(); y = yte[:K].copy()

    # Plaintext: dùng pipeline predict_proba
    t0 = time.perf_counter()
    proba_plain = pipe.predict_proba(X)[:,1]
    t1 = time.perf_counter()
    plain_time = (t1 - t0)/K if K > 0 else 0.0

    # Lấy vector hoá chuẩn hoá và hệ số LDA tuyến tính
    X_std = scaler.transform(X)
    scores_pt, w, b = lda_linear_form(lda, X_std)

    # HE tuyến tính: y = w·x + b trong miền mã hoá
    scores_he, he_mean, he_std = he_linear_scores(X_std, w, b, keybits=args.keybits)

    # Chuẩn hoá về “giống” decision_function
    # (Vì predict_proba là sigmoid/softmax sau tuyến tính; ở đây ta minh hoạ mức tuyến tính)
    # Kiểm tra chênh lệch
    mae = float(np.mean(np.abs(scores_pt - scores_he)))

    print(f"[DEMO] featuresdir={featuresdir}  modeldir={modeldir}  K={K}  keybits={args.keybits}")
    print(f"[PLAINTEXT] time/sample ≈ {plain_time:.6f} s (pipeline predict_proba)")
    print(f"[HE-LINEAR] time/sample ≈ {he_mean:.6f} ± {he_std:.6f} s")
    print(f"[CHECK] |scores_plain - scores_HE| MAE = {mae:.6e}")

    # Gợi ý kết luận ngắn
    if mae < 1e-6:
        print("[OK] Điểm tuyến tính khớp gần như tuyệt đối giữa plaintext và HE (sai số ~0).")
    else:
        print("[WARN] Có sai khác nhỏ giữa plaintext và HE, nhưng thường rất bé do sai số số học.")

if __name__ == "__main__":
    main()
