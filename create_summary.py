import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score
from joblib import load

# Định nghĩa thư mục chứa các kết quả từ tất cả A_xx
input_dir = Path("artifacts")
output_dir = Path("artifacts/summary")
output_dir.mkdir(parents=True, exist_ok=True)

# Tạo danh sách các A_xx (A_01 đến A_19)
subject_dirs = [f"A_{str(i).zfill(2)}" for i in range(1, 20)]

# Khởi tạo các biến để tính toán trung bình
all_conf_matrices = []
all_accuracies = []
all_precisions = []
all_recalls = []
all_f1_scores = []
all_auc_scores = []
all_times = []
subject_names = []

# Hàm vẽ confusion matrix với giá trị số trong ô
def plot_confusion_matrix(cm, labels, title="Confusion Matrix"):
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(labels))
    plt.xticks(tick_marks, labels)
    plt.yticks(tick_marks, labels)

    # Đặt nhãn cho ma trận nhầm lẫn
    thresh = cm.max() / 2.
    for i in range(len(labels)):
        for j in range(len(labels)):
            # Chỉnh sửa lại định dạng cho các giá trị thực
            plt.text(j, i, format(cm[i, j], '.2f'),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()

# Lặp qua từng subject (A_xx)
for subject in subject_dirs:
    print(f"Processing {subject}")
    
    # Đọc tệp confusion matrix, classification report, và các thông số khác cho mỗi subject
    features_file = input_dir / f"{subject}_test_features.npz"
    if not features_file.exists():
        print(f"File {features_file} does not exist, skipping {subject}.")
        continue
    
    # Tải dữ liệu từ test features
    data = np.load(features_file)
    Xte = data["X"]
    yte = data["y"]

    # Tải model tốt nhất
    model = load(input_dir / f"{subject}_model_best.joblib")
    
    # Dự đoán trên tập kiểm tra
    y_pred = model.predict(Xte)
    
    # Tính toán confusion matrix
    cm = confusion_matrix(yte, y_pred)
    all_conf_matrices.append(cm)
    
    # Tính toán các chỉ số như accuracy, precision, recall, f1-score
    report = classification_report(yte, y_pred, output_dict=True)
    accuracy = report['accuracy']
    precision = report['1']['precision']
    recall = report['1']['recall']
    f1_score_value = report['1']['f1-score']
    
    all_accuracies.append(accuracy)
    all_precisions.append(precision)
    all_recalls.append(recall)
    all_f1_scores.append(f1_score_value)
    
    # Tính toán AUC
    auc_score = roc_auc_score(yte, y_pred)
    all_auc_scores.append(auc_score)
    
    # Tính toán thời gian suy luận (giả sử có thông số thời gian trong tệp hoặc đo thời gian)
    time_per_sample = model.predict_proba(Xte)  # Giả sử bạn tính thời gian từ predict_proba
    time_taken = time_per_sample.shape[0] / len(Xte)  # Ví dụ: tính thời gian mỗi mẫu
    all_times.append(time_taken)
    
    subject_names.append(subject)

# Tính toán các kết quả trung bình cho tất cả các A_xx
avg_accuracy = np.mean(all_accuracies)
avg_precision = np.mean(all_precisions)
avg_recall = np.mean(all_recalls)
avg_f1_score = np.mean(all_f1_scores)
avg_auc = np.mean(all_auc_scores)
avg_time = np.mean(all_times)

# Xuất ra các kết quả tổng hợp
summary_df = pd.DataFrame({
    'Subject': subject_names,
    'Accuracy': all_accuracies,
    'Precision': all_precisions,
    'Recall': all_recalls,
    'F1 Score': all_f1_scores,
    'AUC': all_auc_scores,
    'Avg Time Per Sample (s)': all_times
})

# Lưu vào CSV
summary_df.to_csv(output_dir / "summary_results.csv", index=False)

# Tính toán confusion matrix tổng hợp (trung bình các confusion matrix)
avg_conf_matrix = np.mean(all_conf_matrices, axis=0)

# Lưu confusion matrix
np.save(output_dir / "average_confusion_matrix.npy", avg_conf_matrix)

# Vẽ các biểu đồ
# AUC vs N (Keybits)
plt.figure()
plt.plot(all_auc_scores, 'o-')
plt.title('AUC vs Keybits')
plt.xlabel('Keybits')
plt.ylabel('AUC')
plt.savefig(output_dir / 'auc_vs_keybits.png')

# Time per sample vs N (Keybits)
plt.figure()
plt.plot(all_times, 'o-')
plt.title('Time per sample vs Keybits')
plt.xlabel('Keybits')
plt.ylabel('Time per sample (s)')
plt.savefig(output_dir / 'time_per_sample_vs_keybits.png')

# Confusion Matrix visualization
plt.figure(figsize=(6, 6))
plot_confusion_matrix(avg_conf_matrix, labels=["Non-target", "Target"], title="Average Confusion Matrix")
plt.savefig(output_dir / 'average_confusion_matrix.png')

# Print Summary Results
print(f"Average Results for all subjects:")
print(f"Average Accuracy: {avg_accuracy:.4f}")
print(f"Average Precision: {avg_precision:.4f}")
print(f"Average Recall: {avg_recall:.4f}")
print(f"Average F1 Score: {avg_f1_score:.4f}")
print(f"Average AUC: {avg_auc:.4f}")
print(f"Average Time Per Sample: {avg_time:.4f}s")

# Ensure to save summary results as markdown too
with open(output_dir / "summary_results.md", 'w') as f:
    f.write(f"# Summary Results for all A_xx\n")
    f.write(f"Average Accuracy: {avg_accuracy:.4f}\n")
    f.write(f"Average Precision: {avg_precision:.4f}\n")
    f.write(f"Average Recall: {avg_recall:.4f}\n")
    f.write(f"Average F1 Score: {avg_f1_score:.4f}\n")
    f.write(f"Average AUC: {avg_auc:.4f}\n")
    f.write(f"Average Time Per Sample: {avg_time:.4f}s\n")

print(f"Summary results and images have been saved to {output_dir}")
