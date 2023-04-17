'''
又要裁剪，不合理，弃用
'''
import numpy as np
from rdkit import Chem


def read_sdf_file(filename):
    """
    从SDF文件中读取分子数据
    """
    suppl = Chem.SDMolSupplier(filename)
    mols = [mol for mol in suppl if mol is not None]
    return mols


PeriodicTable = Chem.GetPeriodicTable()

maxlength = 512


def preprocess_molecule(mol, max_atoms=maxlength):
    """
    对单个分子进行预处理操作，包括节点特征和邻接矩阵的计算等
    """
    # 计算节点特征：元素类型、电荷、质量等
    nodes_feat = []
    for atom in mol.GetAtoms():
        feat = [
            # 原子序数
            float(atom.GetAtomicNum()),
            # 电荷
            float(atom.GetFormalCharge()),
            # 质量
            float(PeriodicTable.GetAtomicWeight(atom.GetAtomicNum()))
        ]
        nodes_feat.append(feat)
    #nodes_feat.sort(key=lambda x: x[0])
    # 构建邻接矩阵
    adj_matrix = Chem.rdmolops.GetAdjacencyMatrix(mol)

    # 填充邻接矩阵为固定维度
    n_atoms = len(nodes_feat)
    global maxlength
    maxlength = max(maxlength, n_atoms)
    if n_atoms < max_atoms:
        pad_size = max_atoms - n_atoms
        nodes_feat += [[0] * len(nodes_feat[0])] * pad_size
        adj_matrix = np.pad(
            adj_matrix, ((0, pad_size), (0, pad_size)), 'constant')
    else:
        nodes_feat = nodes_feat[:max_atoms]
        adj_matrix = adj_matrix[:max_atoms, :max_atoms]

    # 将邻接矩阵转化为度矩阵D和标准化的邻接矩阵A_hat
    D = np.diag(np.sum(adj_matrix, axis=1))
    A_hat = np.linalg.inv(D) @ adj_matrix

    return np.array(nodes_feat), np.array(A_hat)


def generate_gcn_input(sdf_file, max_atoms=30):
    """
    从SDF文件中读取分子数据，并将其预处理为GCN所需的输入格式
    """
    mols = read_sdf_file(sdf_file)
    mol = mols[0] if len(mols) > 0 else None
    if mol is not None:
        nodes_feat, adj_matrix = preprocess_molecule(
            mol, max_atoms=max_atoms)
        # 将节点特征和邻接矩阵打包成元组，作为GCN的输入
        return np.array(nodes_feat), np.array(adj_matrix)
    return None, None


if __name__ == '__main__':
    # 示例代码
    testsdf = "D:/pdb/refined-set/1a1e/1a1e_ligand.sdf"
    nodes_feats, adj_matrices = generate_gcn_input(testsdf)
    print(nodes_feats.shape)  # (n_samples, max_atoms, n_features)
    print(adj_matrices.shape)
    print(nodes_feats)
    # print(adj_matrices)
