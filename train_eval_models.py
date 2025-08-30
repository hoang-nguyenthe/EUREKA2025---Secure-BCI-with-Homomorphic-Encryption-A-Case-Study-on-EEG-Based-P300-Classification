
import json, numpy as np, pandas as pd
from pathlib import Path
from argparse import ArgumentParser
from joblib import dump
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             roc_auc_score, roc_curve, balanced_accuracy_score, classification_report)
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

def build_estimators(linear_only=False):
    ests = []
    lr = LogisticRegression(max_iter=2000, class_weight="balanced")
    lr_params = {"clf__C": np.logspace(-3,2,6), "clf__solver":["liblinear","lbfgs"]}
    ests.append((lr, lr_params, "lr"))
    if not linear_only:
        svc = SVC(kernel="rbf", probability=True, class_weight="balanced")
        svc_params = {"clf__C": np.logspace(-3,2,6), "clf__gamma":["scale","auto"]}
        ests.append((svc, svc_params, "svc"))
        knn = KNeighborsClassifier()
        knn_params={"clf__n_neighbors":[3,5,7,9,11],"clf__weights":["uniform","distance"]}
        ests.append((knn, knn_params, "knn"))
        dt = DecisionTreeClassifier(class_weight="balanced")
        dt_params={"clf__max_depth":[3,5,7,9,None]}
        ests.append((dt, dt_params, "dt"))
    lda = LinearDiscriminantAnalysis()
    lda_params = {}  # không có HP
    ests.append((lda, lda_params, "lda"))
    return ests

def main():
    ap = ArgumentParser()
    ap.add_argument("--indir",  default="artifacts", help="nơi chứa train_features.npz & test_features.npz")
    ap.add_argument("--outdir", default="artifacts", help="nơi ghi model/scaler/best_config.json & csv")
    ap.add_argument("--linear-only", action="store_true", help="Chỉ LR/LDA (phù hợp HE)")
    ap.add_argument("--cv", type=int, default=5)
    args = ap.parse_args()

    IN  = Path(args.indir)
    OUT = Path(args.outdir); OUT.mkdir(parents=True, exist_ok=True)

    tr = np.load(IN/"train_features.npz"); Xtr, ytr = tr["X"], tr["y"]
    te = np.load(IN/"test_features.npz");  Xte, yte = te["X"], te["y"]

    cv = StratifiedKFold(n_splits=args.cv, shuffle=True, random_state=42)
    rows=[]
    best_auc=-1; best_est=None; best_name=None; best_grid=None

    for clf, params, name in build_estimators(linear_only=args.linear_only):
        pipe = Pipeline([("scaler", StandardScaler()), ("clf", clf)])
        scoring={"accuracy":"accuracy","precision":"precision","recall":"recall","f1":"f1","roc_auc":"roc_auc"}
        grid = GridSearchCV(pipe, param_grid=(params or {}), scoring=scoring, refit="roc_auc",
                            cv=cv, n_jobs=-1, verbose=0)
        grid.fit(Xtr, ytr)
        y_score_tr = (grid.best_estimator_.predict_proba(Xtr)[:,1]
                      if hasattr(grid.best_estimator_,"predict_proba")
                      else grid.best_estimator_.decision_function(Xtr))
        record = {
            "model":name, "cv_best_roc_auc":grid.best_score_,
            "train_auc": roc_auc_score(ytr, y_score_tr),
            "best_params": grid.best_params_
        }
        rows.append(record)
        if grid.best_score_ > best_auc:
            best_auc = grid.best_score_; best_est = grid.best_estimator_; best_name = name; best_grid = grid

    pd.DataFrame(rows).to_csv(OUT/"results_cv.csv", index=False)
    print("=== CV SUMMARY ===")
    print(pd.DataFrame(rows).sort_values("cv_best_roc_auc", ascending=False))

    # Chọn ngưỡng trên validation (tách từ Train) cho model thắng
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    tr_idx, val_idx = next(sss.split(Xtr, ytr))
    # Dự báo trên VAL với best_est
    best_est.fit(Xtr[tr_idx], ytr[tr_idx])  # fit trên train-subsample
    if hasattr(best_est, "predict_proba"):
        p_val = best_est.predict_proba(Xtr[val_idx])[:,1]
    else:
        # decision_function -> scale về [0,1] bằng min-max đơn giản cho ROC
        s_val = best_est.decision_function(Xtr[val_idx]); s = (s_val - s_val.min())/(s_val.max()-s_val.min()+1e-9)
        p_val = s
    fpr,tpr,thr = roc_curve(ytr[val_idx], p_val)
    thr_opt = float(thr[(tpr - fpr).argmax()])

    # Fit lại trên toàn bộ TRAIN
    best_est.fit(Xtr, ytr)

    # Đánh giá TEST
    if hasattr(best_est, "predict_proba"):
        p_te = best_est.predict_proba(Xte)[:,1]
    else:
        s_te = best_est.decision_function(Xte); s = (s_te - s_te.min())/(s_te.max()-s_te.min()+1e-9)
        p_te = s
    y05 = (p_te>=0.5).astype(int)
    yT  = (p_te>=thr_opt).astype(int)
    def pack(yhat):
        return dict(acc=accuracy_score(yte, yhat),
                    bal_acc=balanced_accuracy_score(yte, yhat),
                    f1_pos=f1_score(yte, yhat, pos_label=1))
    auc = roc_auc_score(yte, p_te)
    print(f"\n[WINNER]: {best_name} | CV AUC={best_auc:.4f}")
    print(f"[TEST] AUC={auc:.4f}  thr_opt={thr_opt:.4f}")
    print("[TEST] @0.5:", pack(y05))
    print("[TEST] @opt:", pack(yT))
    print("\nClassification report (@opt):\n", classification_report(yte, yT, digits=3))

    # Lưu model/scaler/ cấu hình
    # Lấy scaler & clf bên trong Pipeline để dùng HE (nếu là tuyến tính)
    scaler = best_est.named_steps["scaler"]; clf = best_est.named_steps["clf"]
    dump(best_est, OUT/"model_best.joblib")             # pipeline đầy đủ
    dump(scaler,  OUT/"scaler_best.joblib")            # scaler riêng
    cfg = {"winner": best_name, "thr_opt": thr_opt}
    (OUT/"best_config.json").write_text(json.dumps(cfg, indent=2), encoding="utf-8")
    print("Saved:", OUT/"model_best.joblib", OUT/"scaler_best.joblib", OUT/"best_config.json")

    # Ghi chú HE-ready
    he_ready = isinstance(clf, (LogisticRegression, LinearDiscriminantAnalysis))
    if he_ready:
        print("HE-ready: YES (tuyến tính). Có thể dùng he_infer_paillier_artifacts.py.")
    else:
        print("HE-ready: NO (mô hình không tuyến tính).")

if __name__ == "__main__":
    main()
