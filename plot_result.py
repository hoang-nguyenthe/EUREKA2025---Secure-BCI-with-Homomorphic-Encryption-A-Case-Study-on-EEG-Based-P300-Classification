import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import confusion_matrix, classification_report
from joblib import load
import numpy as np  # Đảm bảo rằng NumPy được import

# Đọc tệp kết quả từ CSV
CSV = Path("artifacts/summary_all.csv")
OUT_DIR = Path("artifacts")
OUT_DIR.mkdir(parents=True, exist_ok=True)

def pick(df, pattern):
    """Tìm và trả về tên cột chứa pattern (case-insensitive)"""
    pattern = pattern.lower()
    for c in df.columns:
        if pattern in c.lower():
            return c
    raise KeyError(f"Không tìm thấy cột chứa '{pattern}' trong {list(df.columns)}")

def plot_confusion_matrix(cm, labels, title="Confusion Matrix"):
    """Vẽ và hiển thị confusion matrix"""
    plt.figure(figsize=(6, 6))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(title)
    plt.colorbar()
    tick_marks = range(len(labels))
    plt.xticks(tick_marks, labels)
    plt.yticks(tick_marks, labels)

    # Đặt nhãn cho ma trận nhầm lẫn
    thresh = cm.max() / 2.
    for i in range(len(labels)):
        for j in range(len(labels)):
            plt.text(j, i, format(cm[i, j], 'd'),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()

def main():
    if not CSV.exists():
        raise FileNotFoundError(f"Không thấy {CSV}. Hãy chạy ./run_he_suite.sh hoặc aggregate trước.")

    df = pd.read_csv(CSV)

    # Chuẩn hóa tên cột: lower + thay khoảng trắng/ký tự đặc biệt -> _
    norm_cols = []
    for c in df.columns:
        cc = c.strip().lower()
        for ch in [' ', '/', '@', '-', '(', ')']:
            cc = cc.replace(ch, '_')
        while '__' in cc:
            cc = cc.replace('__', '_')
        norm_cols.append(cc.strip('_'))
    df.columns = norm_cols

    # Bắt các cột cần thiết theo mẫu (case-insensitive, linh hoạt)
    col_model   = pick(df, 'model')
    col_n       = pick(df, 'n')
    col_keybits = pick(df, 'keybits')
    col_auc_pl  = pick(df, 'auc_plain')
    col_auc_he  = pick(df, 'auc_he')
    col_acc_pl  = pick(df, 'acc_plain')
    col_acc_he  = pick(df, 'acc_he')
    # thời gian có thể là time_per_sample_s hoặc time/sample
    try:
        col_tps = pick(df, 'time_per_sample')
    except KeyError:
        col_tps = pick(df, 'time')

    # Sắp xếp cho đẹp
    df = df.sort_values([col_keybits, col_n]).reset_index(drop=True)

    # Lặp qua tất cả các thư mục A_xx (A_01 đến A_19)
    subject_dirs = [f"A_{str(i).zfill(2)}" for i in range(1, 20)]  # A_01 đến A_19

    for subject in subject_dirs:
        # Đọc tệp features cho mỗi thư mục A_xx
        features_file = OUT_DIR / f"{subject}_test_features.npz"
        if not features_file.exists():
            print(f"Không tìm thấy tệp {features_file}, bỏ qua {subject}.")
            continue

        # Đọc dữ liệu và mô hình cho từng subject
        model_file = OUT_DIR / f"{subject}_model_best.joblib"
        model = load(model_file)
        data = np.load(features_file)
        Xte, yte = data["X"], data["y"]

        # Dự đoán trên tập kiểm tra
        y_pred = model.predict(Xte)
        cm = confusion_matrix(yte, y_pred)

        # Vẽ và lưu confusion matrix
        plot_confusion_matrix(cm, labels=["Non-target", "Target"], title=f"Confusion Matrix for {subject}")
        plt.savefig(OUT_DIR / f"{subject}_confusion_matrix.png", dpi=150)
        
        # In ra classification report và confusion matrix
        print(f"\nConfusion Matrix for {subject}:")
        print(cm)
        print(f"Classification Report for {subject}:")
        print(classification_report(yte, y_pred))

        # Lưu classification report vào tệp
        with open(OUT_DIR / f"{subject}_classification_report.txt", "w") as f:
            f.write(f"Confusion Matrix for {subject}:\n")
            f.write(str(cm))
            f.write(f"\n\nClassification Report for {subject}:\n")
            f.write(classification_report(yte, y_pred))

        # Lưu các biểu đồ tổng hợp
        plt.figure()
        for kb, g in df.groupby(col_keybits):
            plt.plot(g[col_n], g[col_auc_pl], 'o-', label=f'Plain AUC (kb={kb})')
            plt.plot(g[col_n], g[col_auc_he], 's--', label=f'HE AUC (kb={kb})')
        plt.xlabel('Số đặc trưng N')
        plt.ylabel('AUC')
        plt.title(f'AUC for {subject} (Plain vs HE)')
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(OUT_DIR / f"{subject}_auc_vs_n_by_keybits.png", dpi=150)

        # Tính toán thời gian và lưu biểu đồ
        plt.figure()
        for kb, g in df.groupby(col_keybits):
            plt.plot(g[col_n], g[col_tps], 'o-', label=f'kb={kb}')
        plt.xlabel('Số đặc trưng N')
        plt.ylabel('Thời gian / mẫu (s)')
        plt.title(f'Time per sample for {subject}')
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(OUT_DIR / f"{subject}_time_per_sample_by_n_and_keybits.png", dpi=150)

if __name__ == "__main__":
    main()
