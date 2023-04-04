import os
import numpy as np

# 定义文件夹路径
folder_path = "z:/"

# 遍历文件夹中的所有.npy文件
for filename in os.listdir(folder_path):
    if filename.endswith(".npy"):
        # 加载.npy文件
        file_path = os.path.join(folder_path, filename)
        arr = np.load(file_path)

        # 输出shape和前10个元素
        print(f"File: {filename}, Shape: {arr.shape}")