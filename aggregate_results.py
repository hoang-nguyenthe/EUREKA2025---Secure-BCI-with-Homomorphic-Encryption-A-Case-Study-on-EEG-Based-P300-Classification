import csv, json, os
from pathlib import Path

ART = Path("artifacts")
CSV = ART / "results_log_artifacts.csv"
OUT_CSV = ART / "summary_all.csv"
OUT_MD  = ART / "summary_all.md"

if not CSV.exists():
    raise SystemExit(f"Not found: {CSV}. Hãy chạy run_he_suite.sh trước.")

# Đọc tất cả bản ghi
rows = []
with open(CSV, newline="") as f:
    reader = csv.DictReader(f)
    for r in reader:
        rows.append(r)

# Chuẩn hóa kiểu dữ liệu cần sort
def ffloat(x, default=None):
    try:
        return float(x)
    except:
        return default

# Sắp xếp: theo model, rồi theo features (N), rồi keybits tăng dần
rows.sort(key=lambda r: (r.get("model",""), ffloat(r.get("N","")), ffloat(r.get("keybits","")) or 0))

# Ghi CSV tổng hợp
cols = ["model","N","keybits","S","acc_plain@opt","auc_plain","acc_he@opt","auc_he","time_total_s","time_per_sample_s"]
with open(OUT_CSV, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=cols)
    w.writeheader()
    for r in rows:
        w.writerow({k: r.get(k,"") for k in cols})

# Ghi Markdown bảng gọn đẹp
def fmt(v, nd=4):
    try:
        return f"{float(v):.{nd}f}"
    except:
        return str(v)

with open(OUT_MD, "w") as f:
    f.write("# Tổng hợp kết quả HE vs plaintext\n\n")
    f.write("| model | N | keybits | AUC_plain | ACC_plain@opt | AUC_HE | ACC_HE@opt | time/sample (s) |\n")
    f.write("|---|---:|---:|---:|---:|---:|---:|---:|\n")
    for r in rows:
        f.write("| {model} | {N} | {keybits} | {auc_p} | {acc_p} | {auc_h} | {acc_h} | {tps} |\n".format(
            model=r.get("model",""),
            N=r.get("N",""),
            keybits=r.get("keybits",""),
            auc_p=fmt(r.get("auc_plain","")),
            acc_p=fmt(r.get("acc_plain@opt","")),
            auc_h=fmt(r.get("auc_he","")),
            acc_h=fmt(r.get("acc_he@opt","")),
            tps=fmt(r.get("time_per_sample_s",""), nd=3),
        ))

print(f"[OK] Wrote {OUT_CSV} and {OUT_MD}")

