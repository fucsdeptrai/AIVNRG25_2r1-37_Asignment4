import pandas as pd
import matplotlib.pyplot as plt

# Đọc file kết quả
df = pd.read_csv('results/exp_results.csv')

# Thống kê trung bình của mỗi chỉ số theo updater
grouped = df.groupby('updater').mean().reset_index()
print(grouped[['updater', 'mse_mean', 'cos_mean', 'update_time', 'mem_norm_change']])

# Vẽ boxplot so sánh MSE giữa hai updater
plt.figure(figsize=(6,4))
df.boxplot(column='mse_mean', by='updater')
plt.title('MSE by Updater')
plt.suptitle('')
plt.ylabel('Mean Squared Error')
plt.savefig('mse_comparison.png')

# Vẽ boxplot so sánh cosine similarity
plt.figure(figsize=(6,4))
df.boxplot(column='cos_mean', by='updater')
plt.title('Cosine Similarity by Updater')
plt.suptitle('')
plt.ylabel('Mean Cosine Similarity')
plt.savefig('cos_comparison.png')

# Vẽ boxplot so sánh thời gian cập nhật
plt.figure(figsize=(6,4))
df.boxplot(column='update_time', by='updater')
plt.title('Update Time by Updater')
plt.suptitle('')
plt.ylabel('Update Time (s)')
plt.savefig('time_comparison.png')

# Vẽ boxplot so sánh thay đổi norm bộ nhớ
plt.figure(figsize=(6,4))
df.boxplot(column='mem_norm_change', by='updater')
plt.title('Memory Norm Change by Updater')
plt.suptitle('')
plt.ylabel('Memory Norm (Frobenius)')
plt.savefig('norm_change_comparison.png')
