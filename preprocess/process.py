# 该文件用于处理PDBBind2020数据集
import zipfile
from functools import cache
import itertools
import json
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
def_path = PDBBindDir + "index/"
output_path = "z:/"
pairs_length = len([name for name in os.listdir(
    data_path) if os.path.isdir(os.path.join(data_path, name))])
# 本次任务处理的集合
num_size = 128
percent_train = 4 / 32
protein_seq_maxlength = 1024

# 读取PDB文件

# 定义三字母到一字母氨基酸缩写的映射字典
aa_dict = {
    'ALA': 'A',
    'CYS': 'C',
    'ASP': 'D',
    'GLU': 'E',
    'PHE': 'F',
    'GLY': 'G',
    'HIS': 'H',
    'ILE': 'I',
    'LYS': 'K',
    'LEU': 'L',
    'MET': 'M',
    'ASN': 'N',
    'PRO': 'P',
    'GLN': 'Q',
    'ARG': 'R',
    'SER': 'S',
    'THR': 'T',
    'VAL': 'V',
    'TRP': 'W',
    'TYR': 'Y'}
aa_dict_number = {
    'ALA': 1,
    'VAL': 2,
    'LEU': 3,
    'ILE': 4,
    'MET': 5,
    'PHE': 6,
    'TRP': 7,
    'PRO': 8,
    'TYR': 9,
    'CYS': 10,
    'HIS': 11,
    'LYS': 12,
    'ARG': 13,
    'GLN': 14,
    'ASN': 15,
    'GLU': 16,
    'ASP': 17,
    'SER': 18,
    'THR': 19,
    'GLY': 20}


def pdb_parse(filename):
    seq = []
    # 创建PDB解析器对象
    parser = PDBParser()

    # 读取PDB文件
    structure = parser.get_structure('protein', filename)

    # 遍历每个模型
    for model in structure:
        # 遍历每个链
        for chain in model:
            print(chain.id)
            # 提取氨基酸序列
            # sequence = [aa_dict.get(residue.get_resname(), '')  # 扔掉不是氨基酸的东西
            sequence = [aa_dict_number.get(residue.get_resname())  # 扔掉不是氨基酸的东西
                        for residue in chain if aa_dict_number.get(residue.get_resname())]
            #sequence_str = ''.join(sequence)
            # seq.append(sequence_str)
            if len(sequence) > 0:
                seq.append(sequence)
            #print('Chain {}: {}'.format(chain.id, sequence_str))

    return seq
# 提取1d氨基酸序列


def extract_seq(file_list):
    sequences = []
    for file in file_list:
        # sequences.append("".join(pdb_parse(file)))
        chains = pdb_parse(file)
        flat_array = list(itertools.chain(*chains))
        sequences.append(flat_array)
    return sequences

# 目前还是采用裁剪的方法，因为想不出来还有什么可行的方案


def pad_array(arr, dtype=np.int32):
    max_len = max([len(row) for row in arr])
    print("Pad array: max_len: {}".format(max_len))
    padded_arr = np.zeros(
        # (len(arr), min(protein_seq_maxlength, max_len)), dtype=dtype)
        (len(arr), protein_seq_maxlength), dtype=dtype)
    for i, row in enumerate(arr):
        padded_arr[i, :min(protein_seq_maxlength, len(row))
                   ] = row[:min(protein_seq_maxlength, len(row))]
    return padded_arr
# 从目录文件读取


def extract_seq_from_file(splitset):
    file_list = [data_path + i + "/" + i +
                 "_protein.pdb" for i in read_filelist(splitset)]
    seqs = extract_seq(file_list)
    if not seqs:
        return
    # with open(output_path + splitset + "_sequence", 'w', encoding='utf-8') as f:
        # f.write("\n".join(seqs))
    # print(seqs)
    seqs = pad_array(seqs)
    dump_npy(splitset + "_sequence", seqs)

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
        splitset, file_path=def_path + "INDEX_refined_data.2020"):
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
    pdbs = read_filelist(splitset)
    data = df.loc[df['-logKd/Ki'].notna() & df['pdbid'].isin(pdbs)].values
    data = data[:, 3]
    data = data.astype(np.float32)
    dump_npy(splitset + "_logk", data)
    return data


# 生成SMILES文本
SMILES_CHARS = [' ', '#', '(', ')', '+', '-', '.', '/', ':', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
                '=', '@', 'B', 'C', 'F', 'H', 'I', 'N', 'O', 'P', 'S', '[', '\\', ']', 'a', 'c', 'l', 'n', 'o', 'r', 's']
smiles_dict = {v: i for i, v in enumerate(SMILES_CHARS)}


def generate_smiles(file_path):
    """
    从mol2/sdf文件中读取分子并生成对应的SMILES字符串

    :param file_path: mol2/sdf文件路径
    :return: SMILES字符串
    """
    # 读取分子
    mol = Chem.MolFromMolFile(file_path, sanitize=False)

    # 生成SMILES
    smiles = Chem.MolToSmiles(mol)

    return smiles


def generate_compound_1d(splitset):
    smiles_list = []
    ligands = read_filelist(splitset)
    for folder_name in ligands:
        # 构建mol2文件路径
        mol2_path = data_path + folder_name + "/" + folder_name + "_ligand.sdf"
        # 读取分子并生成SMILES
        one_smiles = generate_smiles(mol2_path)
        for i in one_smiles:
            assert i in smiles_dict, f"'{i}' not in smiles dict"
        one_smiles = [smiles_dict[i] for i in one_smiles]
        smiles_list.append(one_smiles)
    #dump_set(splitset + "_smiles", smiles_list)
    dump_npy(splitset + "_smiles", pad_array(smiles_list))
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
    dump_npy(output_file, adjacency_matrices_np)
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
hashfile = data_path + "hash_record.json"


def read_hash_record():
    global hash_record
    if os.path.exists(hashfile):
        with open(hashfile, "r", encoding="utf-8") as f:
            hash_record = json.load(f)


def write_hash_record():
    with open(hashfile, "w", encoding="utf-8") as f:
        json.dump(hash_record, f)


read_hash_record()


def get_id(dirname, is_protein=False):
    filename = f"{dirname}/{dirname}_protein.pdb" if is_protein else f"{dirname}/{dirname}_ligand.mol2"
    if filename in hash_record:
        return hash_record[filename]
    hash_record[filename] = get_pdb_hash(
        filename) if is_protein else get_mol2_hash(filename)
    return hash_record[filename]


def dump_set(file_path, data):
    with open(output_path + file_path, "w", encoding='utf-8') as f:
        for item in data:
            f.write(item + "\n")


def dump_npy(file_path, data):
    with open(output_path + file_path, "wb") as f:
        np.save(f, data)


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
    write_hash_record()
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


cached_files = {}


def read_filelist(splitset):
    if splitset in cached_files:
        return cached_files[splitset]
    with open(output_path + splitset, "r", encoding='utf-8') as f:
        files = f.read().splitlines()
    cached_files[splitset] = files
    return files


def zip_train_test_files(path=output_path, compress=False):
    """
    将所有以 train 和 test 开头的无后缀文件打包成 zip 文件
    """
    # 获取所有以 train 和 test 开头的文件路径
    file_paths = [path + f for f in os.listdir(path)
                  if os.path.isfile(path + f)
                  and (f.startswith('train') or f.startswith('test'))
                  and '.' not in f]

    # 创建 zip 文件并添加文件
    with zipfile.ZipFile(path + 'data.zip', 'w', zipfile.ZIP_LZMA if compress else zipfile.ZIP_DEFLATED) as zipf:
        for file_path in file_paths:
            zipf.write(file_path, arcname=os.path.basename(file_path))


all = ["train", "test_protein_only", "test_ligand_only", "test_both_none"]
if __name__ == "__main__":
    #split_dataset()
    # for i in ["test_both_none"]:
    # for i in []:
    # for i in ["train"]:
    for i in ["train","test_both_none"]:
    #for i in all:
        extract_seq_from_file(i)
        generate_compound_1d(i)
        extract_protein_compound_label(i)
    # zip_train_test_files()