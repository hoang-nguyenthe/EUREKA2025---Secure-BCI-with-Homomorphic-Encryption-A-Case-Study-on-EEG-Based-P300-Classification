import json, time, numpy as np
from pathlib import Path
from joblib import load
from argparse import ArgumentParser
from math import log
from phe import paillier
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, roc_auc_score

def logit(p):
    if p <= 0: return float("-inf")
    if p >= 1: return float("inf")
    return log(p / (1 - p))

ap = ArgumentParser()
ap.add_argument("--featuresdir", default="artifacts")
ap.add_argument("--modeldir",    default="artifacts")
ap.add_argument("--n", type=int, default=128)
ap.add_argument("--keybits", type=int, default=2048)
ap.add_argument("--S", type=int, default=10000)
ap.add_argument("--seed", type=int, default=0)
args = ap.parse_args()

FD = Path(args.featuresdir)
MD = Path(args.modeldir)

# Lặp qua tất cả các thư mục A_xx (A_01 đến A_19)
subject_dirs = [f"A_{str(i).zfill(2)}" for i in range(1, 20)]  # A_01 đến A_19

# Lặp qua từng thư mục A_xx
for subject in subject_dirs:
    # Đọc tệp features cho mỗi thư mục A_xx
    features_file = FD / f"{subject}_test_features.npz"
    if not features_file.exists():
        print(f"Không tìm thấy tệp {features_file}, bỏ qua {subject}.")
        continue

    Xte = np.load(features_file)["X"]
    yte = np.load(features_file)["y"]

    # Tải mô hình và scaler
    model_obj = load(MD / f"{subject}_model_best.joblib")
    scaler = None
    clf = model_obj
    if isinstance(model_obj, Pipeline):
        scaler = model_obj.named_steps.get("scaler", None)
        clf    = model_obj.named_steps.get("clf", None)
        if clf is None:
            raise ValueError(f"Mô hình trong {subject} không tìm thấy step 'clf' trong Pipeline.")
    if scaler is None:
        try:
            scaler = load(MD / f"{subject}_scaler_best.joblib")
        except:
            class _NoScaler:
                def transform(self, X): return X
            scaler = _NoScaler()

    # Đọc ngưỡng tối ưu và chuyển sang logit
    cfg = json.loads((MD / f"{subject}_best_config.json").read_text(encoding="utf-8"))
    thr_p = float(cfg.get("thr_opt", 0.5))
    thr_s = logit(thr_p)

    # Trọng số tuyến tính
    if not hasattr(clf, "coef_") or not hasattr(clf, "intercept_"):
        raise ValueError(f"Mô hình trong {subject} không phải LR/LDA tuyến tính (không có coef_/intercept_).")
    w = clf.coef_[0].astype(np.float64)
    b = float(clf.intercept_[0])

    # Scale features
    Xte_s = scaler.transform(Xte)

    # Chọn ngẫu nhiên N mẫu để suy luận HE
    rng = np.random.default_rng(args.seed)
    idx = rng.choice(len(Xte_s), size=min(args.n, len(Xte_s)), replace=False)
    Xt = Xte_s[idx]; yt = yte[idx]

    # ---- Plaintext tham chiếu ----
    s_plain = Xt @ w + b
    p_plain = 1 / (1 + np.exp(-s_plain))
    yhat_plain = (s_plain >= thr_s).astype(int)
    acc_plain = accuracy_score(yt, yhat_plain)
    try: auc_plain = roc_auc_score(yt, p_plain)
    except: auc_plain = float("nan")
    print(f"[PLAINTEXT] N={len(Xt)}  AUC={auc_plain:.4f}  Acc@opt={acc_plain:.4f}  Subject={subject}")

    # ---- Paillier HE ----
    pub, prv = paillier.generate_paillier_keypair(n_length=args.keybits)
    S = args.S
    w_q = np.round(w * S).astype(np.int64)
    b_q = int(round(b * (S**2)))

    def enc_vec(x): return [pub.encrypt(int(v)) for v in np.round(x * S).astype(np.int64)]
    def enc_lin(enc_x):
        acc = pub.encrypt(0)
        for c, wi in zip(enc_x, w_q):
            acc = acc + (c * int(wi))
        return acc + pub.encrypt(int(b_q))

    t0 = time.time()
    preds = []
    probs = []
    for i in range(len(Xt)):
        enc_x = enc_vec(Xt[i])
        enc_s = enc_lin(enc_x)
        s = prv.decrypt(enc_s) / (S**2)
        preds.append(1 if s >= thr_s else 0)
        probs.append(1 / (1 + np.exp(-s)))
    t1 = time.time()

    acc_he = accuracy_score(yt, preds)
    try: auc_he = roc_auc_score(yt, np.array(probs))
    except: auc_he = float("nan")
    print(f"[HE-Paillier] N={len(Xt)}  AUC={auc_he:.4f}  Acc@opt={acc_he:.4f}  time={t1 - t0:.2f}s  ~{(t1 - t0) / len(Xt):.3f}s/sample  Subject={subject}")
