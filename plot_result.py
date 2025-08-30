import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

CSV = Path("artifacts/summary_all.csv")
OUT_DIR = Path("artifacts")
OUT_DIR.mkdir(parents=True, exist_ok=True)

def pick(df, pattern):
    pattern = pattern.lower()
    for c in df.columns:
        if pattern in c.lower():
            return c
    raise KeyError(f"Không tìm thấy cột chứa '{pattern}' trong {list(df.columns)}")

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

    # 1) Plot AUC theo N (mỗi đường là một key size)
    plt.figure()
    for kb, g in df.groupby(col_keybits):
        plt.plot(g[col_n], g[col_auc_pl], 'o-', label=f'Plain AUC (kb={kb})')
        plt.plot(g[col_n], g[col_auc_he], 's--', label=f'HE AUC (kb={kb})')
    plt.xlabel('Số đặc trưng N')
    plt.ylabel('AUC')
    plt.title('AUC: Plain vs HE (theo N, nhóm theo keybits)')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUT_DIR / 'auc_vs_n_by_keybits.png', dpi=150)

    # 2) Plot ACC@opt theo N
    plt.figure()
    for kb, g in df.groupby(col_keybits):
        plt.plot(g[col_n], g[col_acc_pl], 'o-', label=f'Plain ACC@opt (kb={kb})')
        plt.plot(g[col_n], g[col_acc_he], 's--', label=f'HE ACC@opt (kb={kb})')
    plt.xlabel('Số đặc trưng N')
    plt.ylabel('Accuracy @ threshold tối ưu')
    plt.title('ACC@opt: Plain vs HE (theo N, nhóm theo keybits)')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUT_DIR / 'acc_vs_n_by_keybits.png', dpi=150)

    # 3) Plot thời gian/inference theo key size (lấy N cố định tốt nhất nếu có)
    plt.figure()
    for kb, g in df.groupby(col_keybits):
        plt.plot(g[col_n], g[col_tps], 'o-', label=f'kb={kb}')
    plt.xlabel('Số đặc trưng N')
    plt.ylabel('Thời gian / mẫu (s)')
    plt.title('Chi phí suy luận HE theo N và keybits')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUT_DIR / 'time_per_sample_by_n_and_keybits.png', dpi=150)

    # 4) Xuất 1 summary markdown ngắn
    md = []
    md.append("# Biểu đồ tổng hợp kết quả\n")
    md.append(f"- Nguồn dữ liệu: `{CSV}`\n")
    md.append("- Hình ảnh đã lưu:\n")
    md.append("  - `artifacts/auc_vs_n_by_keybits.png`\n")
    md.append("  - `artifacts/acc_vs_n_by_keybits.png`\n")
    md.append("  - `artifacts/time_per_sample_by_n_and_keybits.png`\n")
    (OUT_DIR / "summary_plots.md").write_text("\n".join(md), encoding="utf-8")

    print("# Ảnh và summary_plots.md nằm trong artifacts/")
    print((OUT_DIR / "summary_plots.md").read_text(encoding="utf-8"))

if __name__ == "__main__":
    main()

