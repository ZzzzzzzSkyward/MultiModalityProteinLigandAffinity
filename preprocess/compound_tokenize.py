import torchtext
import rdkit
from rdkit import Chem

# 将SMILES字符串转换为分子对象
smiles = 'CCO'
molecule = Chem.MolFromSmiles(smiles)

# 自定义Tokenize函数


def tokenize_molecule(molecule):
    tokens = []
    for atom in molecule.GetAtoms():
        tokens.append(atom.GetSymbol())
        for neighbor in atom.GetNeighbors():
            # 将连接原子的键类型添加到Token序列中
            bond = molecule.GetBondBetweenAtoms(
                atom.GetIdx(), neighbor.GetIdx())
            bond_type = str(bond.GetBondType())
            tokens.append(bond_type)
    return tokens


# 定义Field类
#molecule_field = torchtext.data.Field(sequential=True, tokenize=tokenize_molecule, use_vocab=True)
# 使用自定义的Tokenize函数将分子对象转换为一维Token序列
tokens = tokenize_molecule(molecule)
print(tokens)
