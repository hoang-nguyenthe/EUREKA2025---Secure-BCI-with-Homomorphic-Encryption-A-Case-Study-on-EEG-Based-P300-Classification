#!/usr/bin/env bash
set -euo pipefail

# === cấu hình bộ chạy mặc định ===
# xDAWN (bạn đã có)
XDAWN_FEAT="inputs/SE001_xdawn"
XDAWN_MODEL="runs/xdawn_cv"

# time-window (tùy chọn: chỉ chạy nếu tồn tại)
TIMEWIN_FEAT="inputs/SE001_timewin64"
TIMEWIN_MODEL="runs/timewin64_cv"

# số feature dùng trong HE (đang dùng 128 cho nhất quán)
N_HE=128
# các độ dài khoá Paillier để so sánh
KEYBITS=("1024" "2048" "3072")

run_one () {
  local FEAT="$1"
  local MODEL="$2"
  local N="$3"
  local KB="$4"

  echo "==> HE infer: FEAT=${FEAT} | MODEL=${MODEL} | N=${N} | KEYBITS=${KB}"
  python he_infer_paillier_artifacts.py \
    --featuresdir "${FEAT}" \
    --modeldir   "${MODEL}" \
    --n "${N}" --keybits "${KB}"

  # log vào artifacts/results_log_artifacts.csv
  python log_he_artifacts.py
  echo
}

echo "[RUN] xDAWN trước"
for kb in "${KEYBITS[@]}"; do
  run_one "${XDAWN_FEAT}" "${XDAWN_MODEL}" "${N_HE}" "${kb}"
done

# nếu có baseline time-window thì chạy thêm
if [ -d "${TIMEWIN_FEAT}" ] && [ -d "${TIMEWIN_MODEL}" ]; then
  echo "[RUN] time-window baseline (phát hiện tự động vì thấy thư mục tồn tại)"
  for kb in "${KEYBITS[@]}"; do
    run_one "${TIMEWIN_FEAT}" "${TIMEWIN_MODEL}" "${N_HE}" "${kb}"
  done
else
  echo "[SKIP] time-window baseline: không thấy ${TIMEWIN_FEAT} hoặc ${TIMEWIN_MODEL}"
fi

echo "[DONE] Đã ghi kết quả vào artifacts/results_log_artifacts.csv"
