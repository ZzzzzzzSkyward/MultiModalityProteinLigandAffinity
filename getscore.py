import re
import csv

# 日志文件路径
log_file_path = "z:/log.txt"

# CSV文件路径
csv_file_path = "score.csv"

# 匹配指标的正则表达式
pearson_regex = r"pearson=\[([\d.\-]+),([\d.\-]+)\]"
rmse_regex = r"rmse=([\d.\-]+)"
tau_regex = r"tau=\[([\d.\-]+),([\d.\-]+)\]"
spearman_regex = r"rho=SpearmanrResult\(correlation=([\d.\-]+), pvalue=([\d.\-]+)\)"

# 读取日志文件
with open(log_file_path, "r") as f:
    log_content = f.read().splitlines()
data = []
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
    data.append([pearson, rmse, tau, rho])
# 将指标保存为CSV文件
with open(csv_file_path, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["pearson", "rmse", "tau", "rho"])
    for i in data:
        writer.writerow(i)
