import re
import csv

# 日志文件路径
log_file_path = "C:/Users/Administrator/Desktop/fsdownload/log.txt"

# CSV文件路径
csv_file_path = "score{i}.csv"

# 匹配指标的正则表达式
pearson_regex = r"pearson=([\d.\-]+)"
rmse_regex = r"rmse=([\d.\-]+)"
tau_regex = r"tau=([\d.\-]+)"
spearman_regex = r"rho=([\d.\-]+)"

# 读取日志文件
with open(log_file_path, "r", encoding='utf-8') as f:
    log_content = f.read().splitlines()
data = [[] for i in range(4)]
idx = 0
for line in log_content:
    # 提取指标值
    pearson_match = re.search(pearson_regex, line)
    rmse_match = re.search(rmse_regex, line)
    tau_match = re.search(tau_regex, line)
    spearman_match = re.search(spearman_regex, line)
    if not pearson_match:
        continue
    pearson = float(pearson_match.group(1))
    rmse = float(rmse_match.group(1))
    tau = float(tau_match.group(1))
    rho = float(spearman_match.group(1))
    data[idx].append([pearson, rmse, tau, rho])
    idx += 1
    idx %= 4
# 将指标保存为CSV文件
for i in range(4):
    with open(csv_file_path.format(i=i), "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["pearson", "rmse", "tau", "rho"])
        for j in data[i]:
            writer.writerow(j)
