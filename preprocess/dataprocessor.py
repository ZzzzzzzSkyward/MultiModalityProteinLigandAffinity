# 该文件用于处理PDBBind2020数据集
from rdkit.Chem import AllChem
from ast import dump
from site import ENABLE_USER_SITE
import tqdm
import hashlib
from imghdr import tests
import shutil
import random
from rdkit import Chem
from Bio.PDB.Polypeptide import aa1, aa3
from Bio.PDB import PDBParser
from Bio import SeqIO
from Bio.SeqUtils import seq3, seq1
import numpy as np
import pandas as pd
import os
import sys
# 参数定义
PDBBindDir = "D:/pdb/"
data_path = PDBBindDir + "refined-set/"
output_path = "z:/pdb/"
pairs_length = len([name for name in os.listdir(
    data_path) if os.path.isdir(os.path.join(data_path, name))])
# 本次任务处理的集合
num_size = 100
percent_train = 0.1

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

# 从目录文件读取


def extract_seq_from_file(file_path):
    seqs = None
    with open(file_path, 'r', encoding='utf-8') as f:
        file_list = f.read().splitlines()
        seqs = extract_seq(file_list)
    with open(file_path + "_sequence", 'w', encoding='utf-8') as f:
        f.write("\n".join(seqs))

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
if __name__ == 'main__':
    pdb_file_path = "D:/pdb/refined-set/1a28/1a28_protein.pdb"
    print(pdb_parse(pdb_file_path))


# 选择-logKd/Ki作为标签，因为只有这个数据是全的


def extract_protein_compound_label(
        file_path="INDEX_refined_data.2020", splitset=None, filename=None):
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
    data = df.loc[df['-logKd/Ki'].notna() & df['pdbid'].isin(splitset)]
    if filename:
        dump_set(filename + "_logk", data)
    return data

# 生成SMILES文本


def generate_smiles(file_path):
    """
    从mol2/sdf文件中读取分子并生成对应的SMILES字符串

    :param file_path: mol2/sdf文件路径
    :return: SMILES字符串
    """
    # 读取分子
    mol = Chem.MolFromMolFile(file_path)

    # 生成SMILES
    smiles = Chem.MolToSmiles(mol)

    return smiles


def generate_compound_1d(splitset, filename=None):
    smiles_list = []
    for folder_name in splitset:
        # 构建mol2文件路径
        mol2_path = os.path.join(data_path, folder_name, f"{folder_name}.mol2")

        # 读取分子并生成SMILES
        smiles_list.append(generate_smiles(mol2_path))
    if filename:
        dump_set(filename + "_smiles", smiles_list)
    return smiles_list


def generate_adjacency_matrices(file_set, file_path, output_file):
    """
    从mol2/sdf文件中读取分子并生成对应的邻接矩阵，并将所有邻接矩阵导出到一个大文件中

    :param file_set: 包含分子文件名的集合
    :param file_path: 包含mol2/sdf文件的文件夹路径
    :param output_file: 导出的numpy文件路径
    """
    # 处理每个文件
    adjacency_matrices = []
    for file_name in file_set:
        # 构建文件路径
        file_path_full = os.path.join(file_path, file_name)

        # 读取分子
        mol = Chem.MolFromMolFile(file_path_full)

        # 生成邻接矩阵
        adj_mat = AllChem.GetAdjacencyMatrix(mol, useBO=True)

        # 添加邻接矩阵到列表中
        adjacency_matrices.append(adj_mat)

    # 将所有邻接矩阵导出到一个文件中
    adjacency_matrices_np = np.array(adjacency_matrices)
    np.save(output_file, adjacency_matrices_np)
# 注意，通过计算哈希值来确定蛋白质或配体是否相同，只能用于格式统一的数据集


def get_pdb_hash(file_path):
    """计算pdb文件的哈希值"""
    with open(data_path + file_path, "rb") as f:
        content = f.read()
        definition = []
        for line in content.splitlines():
            if line.startswith(b"ATOM") or line.startswith(
                    b"HETATM") or line.startswith(b"CONECT"):
                definition.append(line)
        return hash(b"".join(definition))


def get_mol2_hash(file_path):
    """计算mol2文件的哈希值"""
    start = 0
    with open(data_path + file_path, "rb") as f:
        content = f.read().splitlines()
        for i, line in enumerate(content):
            if line.startswith(b"@<TRIPOS>ATOM"):
                start = i
                break
        definition = b"".join(content[start:])

        return hash(definition)


hash_record = {}


def get_id(dirname, is_protein=False):
    filename = f"{dirname}/{dirname}_protein.pdb" if is_protein else f"{dirname}/{dirname}_ligand.mol2"
    if filename in hash_record:
        return hash_record[filename]
    hash_record[filename] = get_pdb_hash(
        filename) if is_protein else get_mol2_hash(filename)
    return hash_record[filename]


def dump_set(file_path, data):
    with open(file_path, "w", encoding='utf-8') as f:
        for item in data:
            f.write(item + "\n")


def split_dataset():
    global num_size
    # 获取所有蛋白质-化合物配体的文件夹名称
    pairs = [
        name for name in os.listdir(data_path) if os.path.isdir(
            os.path.join(
                data_path,
                name))]

    # 随机选择num_size个文件夹，并将其分配到训练集或测试集
    selected = random.sample(pairs, num_size)
    num_train = int(num_size * percent_train)

    # 创建训练集和测试集文件夹
    #train_path = os.path.join(output_path, "train")
    #test_path = os.path.join(output_path, "test")
    #os.makedirs(train_path, exist_ok=True)
    #os.makedirs(test_path, exist_ok=True)

    # 将选择的文件夹分配到训练集或测试集，并检查测试集中的分子和蛋白质是否满足要求
    trains = set()
    tests = set()
    test_proteins = set()
    test_ligands = set()
    known_proteins = set()
    known_ligands = set()
    progress = tqdm.tqdm(range(num_size))
    for i, pair in enumerate(selected):
        progress.update(1)
        if i < num_train:
            #dest_path = os.path.join(train_path, pair)
            #src_path = os.path.join(data_path, pair)
            #shutil.copytree(src_path, dest_path)
            protein_name = get_id(pair, is_protein=True)
            ligand_name = get_id(pair, is_protein=False)
            known_proteins.add(protein_name)
            known_ligands.add(ligand_name)
            trains.add(pair)
        else:
            protein_name = get_id(pair, is_protein=True)
            ligand_name = get_id(pair, is_protein=False)
            if protein_name not in known_proteins or ligand_name not in known_ligands:
                #dest_path = os.path.join(test_path, pair)
                #src_path = os.path.join(data_path, pair)
                #shutil.copytree(src_path, dest_path)
                tests.add(pair)
    # 进一步细分测试集
    test_protein_only = set()
    test_ligand_only = set()
    test_both_none = set()
    test_both_present = set()
    for pair in tests:
        protein_name = get_id(pair, is_protein=True)
        ligand_name = get_id(pair, is_protein=False)
        if protein_name not in known_proteins and ligand_name in known_ligands:
            test_protein_only.add(pair)
        elif protein_name in known_proteins and ligand_name not in known_ligands:
            test_ligand_only.add(pair)
        elif protein_name not in known_proteins and ligand_name not in known_ligands:
            test_both_none.add(pair)
        else:
            test_both_present.add(pair)
    dump_set("test_protein_only", test_protein_only)
    dump_set("test_ligand_only", test_ligand_only)
    dump_set("test_both_none", test_both_none)
    dump_set("train", trains)
    dump_set("all", trains | tests)
    print(
        f"选取{num_size}对,{num_train}|{num_size - num_train}={num_train/num_size}")
    print("集合|蛋白质|化合物")
    print(
        f"训练集|{len(trains)}")
    print(
        f"测试集|{len(test_proteins)}|{len(test_ligands)}")
    print(
        f"蛋白质测试集|{len(test_protein_only)}|")
    print(
        f"化合物测试集| |{len(test_ligand_only)}")
    print(
        f"配对测试集|{len(test_both_none)}")
    print(
        f"非测试集|{len(test_both_present)}")


if __name__ == "__main__":
    split_dataset()
