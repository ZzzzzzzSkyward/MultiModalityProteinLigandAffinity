from compound_selfies import *
import os
import numpy as np

SELFIES_CHARS = [
    ' ', '#', '(', ')', '+', '-', '.', '/', ':', '0', '1', '2', '3', '4', '5',
    '6', '7', '8', '9', '=', '_', '$', '%', '@', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z',
    '|', '\\', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z'
]
a=len(SELFIES_CHARS)
print(a)
ligand_seq_maxlength = 32


def pad_array(arr, dtype=np.int8):
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


def generate_ligand_selfies(smiles_file_path, file_name):
    """
    生成ligand的selfies
    :param smiles_file_path: smiles文件路径
    :param file_name: 保存的文件名
    :return:
    """
    # 读取smiles文件
    with open(smiles_file_path, 'r') as f:
        smiles = f.readlines()
    # 生成selfies
    selfies = encode(smiles, SELFIES_CHARS)
    selfies=pad_array(selfies)
    # 保存selfies
    np.save(file_name, selfies)


if __name__ == '__main__':
    # smiles文件路径
    smiles_dir = "z:/gdb13.cno"
    # 保存的文件名=smiles文件名+后缀
    for i in os.listdir(smiles_dir):
        if i.endswith('.smi'):
            file_name = smiles_dir + '/' + i + '_selfies'
            smiles_file_path = smiles_dir + '/' + i
            generate_ligand_selfies(smiles_file_path, file_name)
            print(i + ' is done')
