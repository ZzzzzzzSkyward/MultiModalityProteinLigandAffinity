# 该文件用于处理PDBBind2020数据集
import pandas as pd
import numpy as np
from Bio.SeqUtils import seq3, seq1
from Bio import SeqIO
from Bio.PDB import PDBParser
from Bio.PDB.Polypeptide import aa1, aa3
from rdkit import Chem


# 读取PDB文件


def pdb_parse(pdb_file_path):
    pdb_records = list(SeqIO.parse(pdb_file_path, "pdb-seqres"))
    # 提取氨基酸序列
    sequence = []
    for record in pdb_records:
        for residue in record.seq:
            if residue != "-":
                sequence.append(seq1(residue))

    return sequence

# 提取1d氨基酸序列


def extract_seq(file_list):
    sequences = []
    for file in file_list:
        sequences.append("".join(pdb_parse(file)))
    return sequences

# one-hot编码


def one_hot_encoding(sequence):
    aa_list = [
        'G',
        'A',
        'V',
        'L',
        'I',
        'F',
        'W',
        'Y',
        'D',
        'E',
        'N',
        'Q',
        'H',
        'K',
        'R',
        'C',
        'M',
        'S',
        'T',
        'P']
    aa_dict = {aa_list[i]: i for i in range(len(aa_list))}
    n_aa = len(aa_list)
    n_seq = len(sequence)
    one_hot = np.zeros((n_seq, n_aa), dtype=int)
    for i, aa in enumerate(sequence):
        if aa in aa_dict:
            one_hot[i, aa_dict[aa]] = 1
    return one_hot


# 示例使用
if __name__ == '__main__':
    pdb_file_path = "D:/pdb/refined-set/1a28/1a28_protein.pdb"
    print(pdb_parse(pdb_file_path))


# 选择-logKd/Ki作为标签，因为只有这个数据是全的


def extract_protein_compound_label(file_path="INDEX_refined_data.2020"):
    """
    从PDBbind 2020数据集中提取负对数Kd/Ki值
    @param file_path PDBbind 2020数据集中的index/INDEX_xx_data.2020文件路径
    @return 包含负对数Kd/Ki的数据框
    """
    # 读取PDBbind 2020数据集中的index/INDEX_xx_data.2020文件
    df = pd.read_csv(
        file_path,
        delim_whitespace=True,
        header=None,
        skiprows=4,
        usecols=[
            0,
            1,
            2,
            3])

    # 重命名列名
    df.columns = ['pdbid', 'resolution', 'release_year', '-logKd/Ki']

    # 筛选出包含-logKd/Ki值的数据
    data = df.loc[df['-logKd/Ki'].notna()]

    return data
