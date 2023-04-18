'''

'''
import deepchem as dc
import numpy as np
from rdkit import Chem
import os
import warnings
# 去掉rdkit的警告
warnings.filterwarnings("ignore")

datadir = "D:/pdb/refined-set/"
outputdir = "z:/"
maxlength = 64
PeriodicTable = Chem.GetPeriodicTable()


def read_sdf_file(filename):
    """
    从SDF文件中读取分子数据
    """
    suppl = Chem.SDMolSupplier(filename)
    mols = [mol for mol in suppl if mol is not None]
    return mols


def get_adjacent_matrix(mol, diag):
    adj_matrix = Chem.rdmolops.GetAdjacencyMatrix(mol, useBO=True)
    for i in range(len(diag)):
        adj_matrix[i, i] = diag[i]
    return np.array(adj_matrix)


def get_charge_matrix(mol, max_atoms):
    charge_matrix_cal = dc.feat.CoulombMatrix(maxlength)
    charge_matrix = charge_matrix_cal(mol)[0]
    return pad_or_truncate_2d(charge_matrix, max_atoms)


def get_distance_matrix(mol):
    dist_matrix = Chem.rdmolops.GetDistanceMatrix(mol)
    return np.array(dist_matrix)


max_seq_len = 0


def pad_or_truncate_3d(matrix, max_len, dtype=np.float32):
    """
    对矩阵进行填充或裁剪以满足定长要求
    Args:
        matrix: 输入矩阵,大小为[batch, dimension, seq_len, seq_len]
        max_len: 目标序列长度
    Returns:
        定长的矩阵,大小为[batch, dimension, max_len, max_len]
    """
    batch = len(matrix)
    dim = len(matrix[0])
    shape = [batch, dim, *matrix[0][0].shape]
    for i in range(2, len(shape)):
        shape[i] = max_len
    ret = np.zeros(shape, dtype=dtype)
    global max_seq_len
    for i, mat in enumerate(matrix):
        mat = np.array(mat)
        _, *size = mat.shape
        seq_len = size[0]
        # 如果序列长度大于max_len,则裁剪
        max_seq_len = max(max_seq_len, seq_len)
        if seq_len != max_len:
            l = min(max_len, seq_len)
            #print(ret.shape, mat.shape, i, l)
            ret[i, :, :l, :l] = mat[:, :l, :l]  # assume 2d graph
        # 如果序列长度正好等于max_len,则直接返回
        else:
            ret[i] = mat

    return ret


def pad_or_truncate_2d(matrix, max_len, dtype=np.float32):
    """
    对矩阵进行填充或裁剪以满足定长要求
    Args:
        matrix: 输入矩阵,大小为[seq_len, seq_len]
        max_len: 目标序列长度
    Returns:
        定长的矩阵,大小为[max_len, max_len]
    """
    shape = [max_len, max_len]
    ret = np.zeros(shape, dtype=dtype)
    global max_seq_len
    mat = np.array(matrix)
    _, *size = mat.shape
    seq_len = size[0]
    # 如果序列长度大于max_len,则裁剪
    max_seq_len = max(max_seq_len, seq_len)
    if seq_len != max_len:
        l = min(max_len, seq_len)
        #print(ret.shape, mat.shape, i, l)
        ret[:l, :l] = mat[:l, :l]  # assume 2d graph
    # 如果序列长度正好等于max_len,则直接返回
    else:
        ret = mat

    return ret


def pad_or_truncate_1d(matrix, max_len, dtype=np.float32):
    """
    对向量进行填充或裁剪以满足定长要求
    Args:
        matrix: 输入矩阵,大小为[batch, seq_len]
        max_len: 目标序列长度
    Returns:
        定长的矩阵,大小为[batch, max_len]
    """
    batch = len(matrix)
    shape = [batch, *matrix[0].shape]
    shape[1] = max_len
    ret = np.zeros(shape, dtype=dtype)
    global max_seq_len
    for i, mat in enumerate(matrix):
        seq_len = len(mat)
        # 如果序列长度大于max_len,则裁剪
        max_seq_len = max(max_seq_len, seq_len)
        if seq_len != max_len:
            l = min(max_len, len(mat[0]))
            # print(i,max_len,ret.shape,mat.shape,l)
            ret[i, :l] = mat[:l]
        # 如果序列长度正好等于max_len,则直接返回
        else:
            ret[i] = mat

    return ret


def normalize(matrix):
    D = np.diag(np.sum(matrix, axis=1))
    try:
        D_half_inv = np.diag(D[D != 0]**(-1 / 2), 0)   # 计算D的对角线元素的平方根的倒数
        A_hat = D_half_inv @ matrix @ D_half_inv
    except BaseException:
        D_half_inv = np.diag(D[D != 0]**(-1 / 2), 0)   # 忽略D中不可逆的元素
        A_hat = D_half_inv @ matrix @ D_half_inv

    return A_hat


def get_matrices(mol):
    num_features = 3
    atomnum = 0
    # features: atom number,charge
    bonds = []
    features = np.zeros((mol.GetNumAtoms(), num_features), np.int32)
    for i, atom in enumerate(mol.GetAtoms()):
        # use rdkit to get the atom's weight
        features[i, 0] = atomnum = atom.GetAtomicNum()
        # get charge
        features[i, 1] = PeriodicTable.GetDefaultValence(
            atomnum) - atom.GetFormalCharge()
        # get bond number, diag include H
        bonds.append(atom.GetTotalNumHs())
        # get hybridization
        features[i, 2] = atom.GetHybridization()
    # get the bonding matrix
    adj_matrix = get_adjacent_matrix(mol, bonds)
    # get charge matrix
    # this one counts H, so it is not aligned
    #charge_matrix = get_charge_matrix(mol, atomnum)
    # get distance matrix
    distance_matrix = get_distance_matrix(mol)
    #adj_matrix = normalize(adj_matrix)
    return features, distance_matrix, adj_matrix, charge_matrix  # , charge_matrix


def generate_gcn_input(sdf_file):
    """
    从SDF文件中读取分子数据，并将其预处理为GCN所需的输入格式
    """
    mols = read_sdf_file(sdf_file)
    mol = mols[0] if len(mols) > 0 else None
    if mol is not None:
        return get_matrices(
            mol)
    return None, None, None, None


if __name__ == '__main__':
    # 示例代码
    testsdf = "D:/pdb/refined-set/1a1e/1a1e_ligand.sdf"
    feature, *matrix = generate_gcn_input(testsdf)
    if feature is None:
        pass
    else:
        print(feature)
        print(*matrix)
        outputpath = "z:/"
        np.savetxt(outputpath + "feature.txt", feature)
        for i in range(len(matrix)):
            np.savetxt(
                outputpath +
                "mat" +
                str(i) +
                ".txt",
                matrix[i],
                fmt="%02d")


def batch_generate(splitset):
    features = []
    matrices = []
    for i in splitset:
        if not os.path.isdir(datadir + i):
            continue
        for j in os.listdir(datadir + i):
            if j.endswith(".sdf"):
                feature, *matrice = generate_gcn_input(
                    datadir + i + "/" + j)
                if feature is not None:
                    features.append(feature)
                    matrices.append(matrice)
                else:
                    print(j)
    features = pad_or_truncate_1d(features, maxlength, np.int32)
    matrices = pad_or_truncate_3d(matrices, maxlength, np.int32)
    print(features.shape, matrices.shape)
    print("max length: ", max_seq_len)
    return features, matrices
