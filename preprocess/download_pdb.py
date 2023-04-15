import os
import urllib.request
from process import get_id

pdb_dir = "z:/refined-set/"
save_dir = "z:/download/"
if not os.path.exists(save_dir):
    os.mkdir(save_dir)


def download_pdb(name, pdb_id):
    # 构造PDB文件下载链接
    url = f"https://files.rcsb.org/download/{pdb_id}.pdb"

    # 构造本地保存路径
    save_path = os.path.join(save_dir, f"{name}.pdb")
    if os.path.exists(save_path):
        return
    # 下载PDB文件并保存到本地
    try:
        print(f"downloading {pdb_id}")
        urllib.request.urlretrieve(url, save_path)
    except Exception as e:
        print(f"error {pdb_id}")

# 指定PDB文件所在的文件夹


# 获取所有PDB ID
pdb_ids = [name for name in os.listdir(
    pdb_dir) if os.path.isdir(pdb_dir + "/" + name)]

# 循环下载每个PDB文件并保存到对应的文件夹中
if __name__ == "__main__":
    for pdb_id in pdb_ids:
        download_pdb(pdb_id, get_id(pdb_id, True))
