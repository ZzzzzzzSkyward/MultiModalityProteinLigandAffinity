# 该文件用于处理PDBBind2020数据集
from math import log
from sklearn.decomposition import PCA
import scipy.sparse as sp
import zipfile
import itertools
import tqdm
import json
import random
import shutil
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem
from Bio.PDB import PDBParser
import numpy as np
import pandas as pd
import os
import warnings
# 去掉rdkit的警告
warnings.filterwarnings("ignore")
# 参数定义
PDBBindDir = "D:/pdb/"
data_path = PDBBindDir + "refined-set/"
toppath = "O:/refined-set/"
def_path = PDBBindDir + "index/"
output_path = "z:/"
pairs_length = len([
    name for name in os.listdir(data_path)
    if os.path.isdir(os.path.join(data_path, name))
])
# 本次任务处理的集合
num_size = 5319
percent_train = 1
protein_seq_maxlength = 2048
ligand_seq_maxlength = 1024

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
    'TYR': 'Y'
}
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
    'GLY': 20
}


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
            # print(chain.id)
            # 提取氨基酸序列
            # sequence = [aa_dict.get(residue.get_resname(), '')  # 扔掉不是氨基酸的东西
            sequence = [
                aa_dict_number.get(residue.get_resname())  # 扔掉不是氨基酸的东西
                for residue in chain
                if aa_dict_number.get(residue.get_resname())
            ]
            # sequence_str = ''.join(sequence)
            # seq.append(sequence_str)
            if len(sequence) > 0:
                # sequence.append(0)#whether to add 0?
                seq.append(sequence)
            # print('Chain {}: {}'.format(chain.id, sequence_str))

    return seq


# 提取1d氨基酸序列


def extract_seq(file_list):
    sequences = []
    progress = tqdm.tqdm(total=len(file_list))
    for file in file_list:
        # sequences.append("".join(pdb_parse(file)))
        chains = pdb_parse(file)
        flat_array = list(itertools.chain(*chains))
        sequences.append(flat_array)
        progress.update(1)
    progress.close()
    return sequences


# 目前还是采用裁剪的方法，因为想不出来还有什么可行的方案


def pad_array(arr, dtype=np.int32):
    max_len = max([len(row) for row in arr])
    target_len = ligand_seq_maxlength
    print("Pad array: max_len: {}".format(max_len))
    padded_arr = np.zeros(
        # (len(arr), min(protein_seq_maxlength, max_len)), dtype=dtype)
        (len(arr), target_len),
        dtype=dtype)
    for i, row in enumerate(arr):
        padded_arr[i, :min(target_len, len(row))
                   ] = row[:min(target_len, len(row))]
    return padded_arr


# 从目录文件读取


def extract_seq_from_file(splitset):
    file_list = [
        data_path + i + "/" + i + "_protein.pdb"
        for i in read_filelist(splitset)
    ]
    seqs = extract_seq(file_list)
    if not seqs:
        return
    # with open(output_path + splitset + "_sequence", 'w', encoding='utf-8') as f:
    # f.write("\n".join(seqs))
    # print(seqs)
    seqs = pad_array(seqs, np.int8)
    dump_npy(splitset + "_sequence", seqs)


# one-hot编码


def one_hot_encoding(sequence):
    aa_list = [
        'G', 'A', 'V', 'L', 'I', 'F', 'W', 'Y', 'D', 'E', 'N', 'Q', 'H', 'K',
        'R', 'C', 'M', 'S', 'T', 'P'
    ]
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

# 选择logKd作为标签，因为只有这个数据是全的

unit_string = {
    'm': "e-3",
    'u': "e-6",
    'n': "e-9",
    'p': "e-12",
    'f': "e-15"
}


def convert_unit(s):
    s = s.replace(
        'M',
        '')
    for i, unit in unit_string.items():
        s = s.replace(
            i,
            unit)
    return s


def extract_protein_compound_label(splitset,
                                   file_path=def_path +
                                   "INDEX_refined_data.2020"):
    """
    从PDBbind 2020数据集中提取负对数Kd/Ki值
    @param file_path PDBbind 2020数据集中的index/INDEX_xx_data.2020文件路径
    @return 包含负对数Kd/Ki的数据框
    """
    # 读取PDBbind 2020数据集中的index/INDEX_xx_data.2020文件
    df = pd.read_csv(file_path,
                     delim_whitespace=True,
                     header=None,
                     skiprows=4,
                     usecols=[0, 3, 4])

    # 重命名列名
    df.columns = ['pdbid', '-logKd/Ki', 'Kd/Ki']

    # 筛选出包含-logKd/Ki值的数据
    pdbs = read_filelist(splitset)
    kd_data = []
    # read 'Kd/Ki' from df, search by 'pdb'
    # 就当做两者是通用的罢
    kd_or_ki=""
    for pdb in pdbs:
        try:
            kd_or_ki = df.loc[df['pdbid'] == pdb, 'Kd/Ki'].values
            kd_or_ki = kd_or_ki[0]
            if 'Kd' in kd_or_ki:
                kd = float(convert_unit(kd_or_ki.split('=')[1]))
                kd = kd_or_ki
            else:
                ki = float(convert_unit(kd_or_ki.split('=')[1]))
                kd = ki
            kd_data.append(-log(kd))  # 改成以e为底（没啥用）
        except Exception as e:
            print(pdb, e, kd_or_ki)
    dump_npy(splitset + "_logk", np.array(kd_data, np.float32))
    return kd_data


# 生成SMILES文本
SMILES_CHARS = [
    ' ', '#', '(', ')', '+', '-', '.', '/', ':', '0', '1', '2', '3', '4', '5',
    '6', '7', '8', '9', '=', '@', '%', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z',
    '[', '\\', ']', 'a', 'b', 'c', 'e', 'i', 'l', 'n', 'o', 'r', 's', 't', 'u'
]
# _ added for 1.5 bond, $=branch, %=ring,eliminate ] for short, replace [ with |
SELFIES_CHARS = [
    ' ', '#', '(', ')', '+', '-', '.', '/', ':', '0', '1', '2', '3', '4', '5',
    '6', '7', '8', '9', '=', '_', '$', '%', '@', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z',
    '|', '\\', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z'
]
assert len(SMILES_CHARS) < 128, "SMILES_CHARS should be less than 256"
assert len(SELFIES_CHARS) < 128, "SELFIES_CHARS should be less than 256"
smiles_dict = {v: i for i, v in enumerate(SMILES_CHARS)}


def generate_smiles(file_path):
    """
    从mol2/sdf文件中读取分子并生成对应的SMILES字符串

    :param file_path: mol2/sdf文件路径
    :return: SMILES字符串
    """
    try:
        # replace Du in file with H
        with open(file_path, 'r') as f:
            lines = f.readlines()
        with open(file_path, 'w') as f:
            for line in lines:
                if line.find('Du') > 0:
                    line = line.replace('Du', 'H')
                f.write(line)
        # 读取分子
        mol = Chem.MolFromMolFile(file_path, sanitize=False)
        # 生成SMILES
        smiles = Chem.MolToSmiles(mol)
    except Exception as e:
        print(file_path)
        # raise e
        return None

    return smiles


def generate_compound_1d(splitset, selfies=False):
    smiles_list = []
    ligands = read_filelist(splitset)
    for folder_name in ligands:
        # 构建mol2文件路径
        mol2_path = data_path + folder_name + "/" + folder_name + "_ligand.sdf"
        # 读取分子并生成SMILES
        one_smiles = generate_smiles(mol2_path)
        if one_smiles is None:
            continue
        for i in one_smiles:
            assert i in smiles_dict, f"'{i}' not in smiles dict, sentence is {one_smiles},{smiles_dict}"
        if not selfies:
            one_smiles = [smiles_dict[i] for i in one_smiles]
        smiles_list.append(one_smiles)
    # dump_set(splitset + "_smiles", smiles_list)
    if selfies:
        import compound_selfies as selfies
        data = selfies.encode(smiles_list, SELFIES_CHARS)
        dump_npy(splitset + "_selfies", pad_array(data, np.int8))
    else:
        dump_npy(splitset + "_smiles", pad_array(smiles_list, np.int8))
    return smiles_list


# 注意，通过计算哈希值来确定蛋白质或配体是否相同，只能用于格式统一的数据集
# 不靠谱，已弃用


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
hashfile = PDBBindDir + "ligands_full.json"


def read_hash_record():
    global hash_record
    if os.path.exists(hashfile):
        with open(hashfile, "r", encoding="utf-8") as f:
            hash_record = json.load(f)
    else:
        print("no hash record file")


def write_hash_record():
    with open(hashfile, "w", encoding="utf-8") as f:
        json.dump(hash_record, f)


read_hash_record()


def get_id(dirname, is_protein=False):
    # filename = f"{dirname}/{dirname}_protein.pdb" if is_protein else
    # f"{dirname}/{dirname}_ligand.mol2"
    if dirname in hash_record:
        return hash_record[dirname][0 if is_protein else 1]
    else:
        print("error: no record for", dirname)
        raise Exception
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


def dump_sparse(file_path, data):
    data = sp.csc_matrix(data)
    """存储稀疏矩阵到文件"""
    with open(output_path + file_path, 'wb') as f:
        # 存储矩阵的shape和非0元素个数
        shape = data.shape
        nnz = data.nnz
        f.write(shape[0].to_bytes(4, 'big'))
        f.write(shape[1].to_bytes(4, 'big'))
        f.write(nnz.to_bytes(4, 'big'))

        # 存储数据、行索引和列索引
        if isinstance(data, sp.csc_matrix) or isinstance(data, sp.csr_matrix):
            data = data.data
            indices = data.indices
            indptr = data.indptr

            data.tofile(f)
            indices.tofile(f)
            indptr.tofile(f)

        elif isinstance(data, sp.coo_matrix):
            row = data.row
            col = data.col
            data = data.data

            row.tofile(f)
            col.tofile(f)
            data.tofile(f)


def split_dataset():
    global num_size
    # 获取所有蛋白质-化合物配体的文件夹名称
    pairs = [
        name for name in os.listdir(data_path)
        if os.path.isdir(os.path.join(data_path, name)) and name != "index"
        and name != "readme"
    ]

    # 随机选择num_size个文件夹，并将其分配到训练集或测试集
    global num_size
    num_size = min(num_size, len(pairs))
    selected = random.sample(pairs, num_size)
    num_train = int(num_size * percent_train)

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
            # dest_path = os.path.join(train_path, pair)
            # src_path = os.path.join(data_path, pair)
            # shutil.copytree(src_path, dest_path)
            protein_name = get_id(pair, is_protein=True)
            ligand_name = get_id(pair, is_protein=False)
            if random.random(
            ) < 0.9 or protein_name not in known_proteins and ligand_name not in known_ligands:
                known_proteins.add(protein_name)
                known_ligands.add(ligand_name)
                trains.add(pair)
            else:
                tests.add(pair)
        else:
            # protein_name = get_id(pair, is_protein=True)
            # ligand_name = get_id(pair, is_protein=False)
            # if protein_name not in known_proteins or ligand_name not in known_ligands:
            # dest_path = os.path.join(test_path, pair)
            # src_path = os.path.join(data_path, pair)
            # shutil.copytree(src_path, dest_path)
            tests.add(pair)
    num_train = len(trains)
    # write_hash_record()
    # 进一步细分测试集
    test_protein_only = set()
    test_ligand_only = set()
    test_both_none = set()
    test_both_present = set()
    for pair in tests:
        protein_name = get_id(pair, is_protein=True)
        ligand_name = get_id(pair, is_protein=False)
        test_proteins.add(protein_name)
        test_ligands.add(ligand_name)
        if protein_name not in known_proteins and ligand_name not in known_ligands:
            test_both_none.add(pair)
        elif ligand_name not in known_ligands:
            test_ligand_only.add(pair)
        elif protein_name not in known_proteins:
            test_protein_only.add(pair)
        else:
            test_both_present.add(pair)
    if len(test_ligand_only) == 0:
        # return 0
        pass
    dump_set("test_protein_only", test_protein_only)
    dump_set("test_ligand_only", test_ligand_only)
    dump_set("test_both_none", test_both_none)
    dump_set("test_both_present", test_both_present)
    dump_set("train", trains)
    dump_set("all", trains | tests)
    print(
        f"选取{num_size}对,{num_train}|{num_size - num_train}={num_train/num_size}"
    )
    print("集合|蛋白质|化合物")
    print(f"训练集|{len(trains)}")
    print(f"测试集|{len(test_proteins)}|{len(test_ligands)}")
    print(f"蛋白质测试集|{len(test_protein_only)}|")
    print(f"化合物测试集| |{len(test_ligand_only)}")
    print(f"没有测试集|{len(test_both_none)}")
    print(f"出现测试集|{len(test_both_present)}")


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
    file_paths = [
        path + f for f in os.listdir(path)
        if os.path.isfile(path + f) and (f.startswith('train') or f.startswith(
            'test') or f == "all") and '.' not in f
    ]

    # 创建 zip 文件并添加文件
    with zipfile.ZipFile(
            path + 'data.zip', 'w',
            zipfile.ZIP_LZMA if compress else zipfile.ZIP_DEFLATED) as zipf:
        for file_path in file_paths:
            zipf.write(file_path, arcname=os.path.basename(file_path))


def split_zernike(splitset):
    # 加载npy格式的矩阵
    matrix = np.load(output_path + 'data_matrix.npy')

    # 读取trainid.txt文件中的训练矩阵子集
    pdbids = read_filelist(splitset)

    # 读取pdbid.txt文件中的PDB ID列表
    with open(PDBBindDir + 'pdbid.txt', 'r', encoding='utf-8') as f:
        pdb_ids = [line.strip() for line in f]

    # 确定训练矩阵子集的列索引c
    indices = []
    for dirname in pdbids:
        pdbid = get_id(dirname, True)
        indices.append(pdb_ids.index(pdbid))

    # 提取trainid.txt中的训练矩阵子集
    split_matrix = matrix[indices, :]

    dump_npy(splitset + "_zernike", split_matrix)


def analyze_data():
    train_file, test_protein_only_file, test_ligand_only_file, test_both_present_file, test_both_none_file = output_path + "train", output_path + \
        "test_protein_only", output_path + "test_ligand_only", output_path + \
        "test_both_present", output_path + "test_both_none"
    # 打开五个文件并读取其中的pdbid
    with open(train_file, 'r') as f:
        train_data = set(line.strip() for line in f)
    with open(test_protein_only_file, 'r') as f:
        test_protein_only_data = set(line.strip() for line in f)
    with open(test_ligand_only_file, 'r') as f:
        test_ligand_only_data = set(line.strip() for line in f)
    with open(test_both_present_file, 'r') as f:
        test_both_present_data = set(line.strip() for line in f)
    with open(test_both_none_file, 'r') as f:
        test_both_none_data = set(line.strip() for line in f)

    # 分别计算train和test集合中的蛋白质、配体数目和蛋白质-配体对数目
    train_proteins = set()
    train_ligands = set()
    train_pairs = set()
    for pair in train_data:
        protein, ligand = get_id(pair, True), get_id(pair, False)
        train_proteins.add(protein)
        train_ligands.add(ligand)
        train_pairs.add(pair)

    test_proteins = set()
    test_ligands = set()
    test_pairs = set()
    val_ligands = set()
    val_proteins = set()
    for pair in test_both_present_data:
        protein, ligand = get_id(pair, True), get_id(pair, False)
        val_ligands.add(ligand)
        val_proteins.add(protein)
    print("Val Set:")
    print(f'{len(val_proteins)},{len(val_ligands)}')
    for pair in test_protein_only_data.union(
            test_ligand_only_data,
            test_both_none_data):
        protein, ligand = get_id(pair, True), get_id(pair, False)
        test_proteins.add(protein)
        test_ligands.add(ligand)
        if pair in train_pairs:
            continue
        test_pairs.add(pair)

    # 输出分析结果
    print("Train dataset:")
    print(f"Number of protein-ligand pairs: {len(train_pairs)}")
    print(f"Number of unique proteins: {len(train_proteins)}")
    print(f"Number of unique ligands: {len(train_ligands)}")

    print("\nTest dataset:")
    print(f"Number of protein-ligand pairs: {len(test_pairs)}")
    print(f"Number of unique proteins: {len(test_proteins)}")
    print(f"Number of unique ligands: {len(test_ligands)}")

    # 计算test集合中不存在于train集合中的蛋白质、配体数目和蛋白质-配体对数目
    test_proteins_not_in_train = test_proteins - train_proteins
    test_ligands_not_in_train = test_ligands - train_ligands
    test_pairs_not_in_train = test_pairs - train_pairs

    print("\nTest dataset - Not in train dataset:")
    print(
        f"Number of unique proteins not in train dataset: {len(test_proteins_not_in_train)}")
    print(
        f"Number of unique ligands not in train dataset: {len(test_ligands_not_in_train)}")
    print(
        f"Number of protein-ligand pairs not in train dataset: {len(test_pairs_not_in_train)}")


def merge_matrix(splitset, pdbid_list, pattern, name):
    output_file = splitset + name
    mats = []
    merged_mat = None
    shape = None
    for pdbid in pdbid_list:
        file_name = toppath + pdbid + "/" + pattern.format(pdbid)
        try:
            mat = np.load(file_name)
        except BaseException:
            continue
        shape = mat.shape
        merged_mat = mat
    for pdbid in pdbid_list:
        file_name = toppath + pdbid + "/" + pattern.format(pdbid)
        try:
            mat = np.load(file_name)
        except BaseException:
            mat = np.ones(shape) * -1
        mats.append(mat)
    merged_mat = np.stack(mats)
    dump_npy(output_file, merged_mat)


def generate_compound_2d(splitset, pdbid_list):
    from compound_gcn import batch_generate
    features, matrices = batch_generate(pdbid_list)
    dump_npy(splitset + "_2dfeature", features)
    dump_npy(splitset + "_matrix_sparse", matrices)


def reverse_sequence(splitset):
    file = output_path + splitset + "_sequence"
    data = np.load(file)
    data = np.flip(data, axis=1)
    dump_npy(splitset + "_sequence", data)


def reduce_morgan_matrix(morgan_matrix, components=None):
    """
    使用 PCA 算法对 Morgan 指纹矩阵进行降维，并提示最佳降维维度或者展示 PCA 维度的信息量

    Args:
        morgan_matrix: Morgan 指纹矩阵，大小为 (n_samples, n_features)

    Returns:
        降维后的矩阵，大小为 (n_samples, n_components)
    """
    # 应用 PCA 算法进行降维
    pca = PCA()
    reduced_data = pca.fit_transform(morgan_matrix)

    # 计算方差解释率
    variance_explained = pca.explained_variance_ratio_

    # 提示最佳降维维度或者展示 PCA 维度的信息量
    if not components:
        print("PCA 维度\t信息量")
        for i, explained_ratio in enumerate(np.cumsum(variance_explained)):
            print(f"{i + 1}\t{explained_ratio:.0%}")
        best_n_components = np.argmax(np.cumsum(variance_explained) >= 0.95) + 1
        print(f"\n建议选择的最佳降维维度为：{best_n_components}")

    # 选择最佳降维维度进行降维
    pca = PCA(n_components=components or best_n_components)
    reduced_data = pca.fit_transform(morgan_matrix)

    return reduced_data


def get_compound_fingerprint(splitset, pdbid_list):
    ret = []
    for pdbid in pdbid_list:
        mol2file = data_path + pdbid + "/" + pdbid + "_ligand.mol2"
        sdffile = data_path + pdbid + "/" + pdbid + "_ligand.sdf"
        mol = Chem.MolFromMol2File(mol2file)
        if not mol:
            uppl = Chem.SDMolSupplier(sdffile)
            mol = uppl[0]
            assert mol, pdbid
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, 128)
        # num = int(DataStructs.BitVectToText(fp), base=2)
        arr = np.zeros((128,))
        DataStructs.ConvertToNumpyArray(fp, arr)
        ret.append(arr)
    ret = np.array(ret, dtype=np.float32)
    # ret = reduce_morgan_matrix(ret,128)
    dump_npy(splitset + "_fingerprint", ret)


if __name__ == "__main__":
    all = [
        "test_protein_only", "test_ligand_only", "test_both_none",
        "test_both_present",
        "train",
    ]
    # split_dataset()
    # for i in ["test_both_none"]:
    for i in all:
        # for i in ["train_full", "test_protein_only_full"]:
        pdbid_list = read_filelist(i)
        # for i in []:
        # extract_seq_from_file(i)
        # reverse_sequence(i)
        for pattern, name in [
                ("{}_protein_feature_complex_alpha_1DCNN.npy", "_complex_alpha"),
                ("{}_protein_feature_complex_electrostatics_1DCNN.npy",
                 "_complex_electric"),
                ("{}_protein_feature_complex_interaction_1DCNN.npy", "_complex_interact"),
                ("{}_ligand_feature_alpha_handcrafted.npy", "_ligand_alpha"),
                ("{}_ligand_feature_ligand_level1_handcrafted.npy", "_ligand_lv1")
        ]:
            pass
            # merge_matrix(i, pattern, name)
        # generate_compound_1d(i, True)
        # get_compound_fingerprint(i, pdbid_list)
        extract_protein_compound_label(i)
        # generate_compound_2d(i,pdbid_list)
        # split_zernike(i)
    # zip_train_test_files()
    # analyze_data()
