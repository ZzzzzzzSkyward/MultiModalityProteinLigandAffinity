'''
从PDBBind2020和rcsb下载的pdb文件仍然无法被pdb2pqr读取
使用pdbfixer修复这些文件
https://github.com/openmm/pdbfixer
https://htmlpreview.github.io/?https://github.com/openmm/pdbfixer/blob/master/Manual.html
'''
import os
import glob
import multiprocessing as mp
from subprocess import call
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")


def process_file(pdb_file):
    """修复单个 PDB 文件，并重命名为 *_output.pdb"""
    output_file = pdb_file.replace('.pdb', '_output.pdb')
    if os.path.exists(output_file):
        return
    # print(pdb_file, output_file)
    call(['pdbfixer', pdb_file, f'--output={output_file}'])


def main():
    """主函数"""
    # 设置进程数
    # num_processes = 4

    # 获取当前工作目录
    current_dir = os.getcwd()  # 该文件需要在refined-set/里运行

    # 查找所有第一级子文件夹中扩展名为 .pdb 的文件，不包括 *_output.pdb 文件
    pdb_files = glob.glob(os.path.join(current_dir, '*', '*.pdb'))
    pdb_files = [f for f in pdb_files if not f.endswith('_output.pdb')]

    # 使用多进程并行修复所有 pdb 文件，并显示进度条
    # processes=num_processes
    with mp.Pool() as pool:
        for _ in tqdm(pool.imap_unordered(process_file, pdb_files),
                      total=len(pdb_files)):
            pass


if __name__ == '__main__':
    main()
