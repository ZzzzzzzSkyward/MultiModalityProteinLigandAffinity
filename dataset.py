from header import *
from logg import *

all_dataset_files = {
    # the amino sequence of proteins
    "seq": "{cate}_sequence",
    # the smiles of compounds
    'smiles': "{cate}_smiles",
    # selfies
    'selfies': "{cate}_selfies",
    # the information of compounds, in adjacent matrix
    'adjacent': "{cate}_matrix",
    '4': "4_selfies",
    '5': "5_selfies",
    '6': "6_selfies",
    '7': "7_selfies",
    '8': "8_selfies",
    # protein contact
    'contact': "{cate}_contact",
    # label of contact
    'interact': "{cate}_interact",
    # label of compound-protein interaction
    'logk': "{cate}_logk",
    'zernike': "{cate}_zernike",
    'fingerprint': "{cate}_fingerprint",
    'matrix': "{cate}_matrix_sparse",
    '2dfeature': "{cate}_2dfeature",
    'alpha': "{cate}_ligand_alpha",
    'lv1': "{cate}_ligand_lv1",
    'complexinteract': '{cate}_complex_interact'
}


def load_matrix(filename):
    """如果文件名包含sparse,加载为稠密矩阵,否则使用np.load正常加载
    注意稀疏矩阵格式int32
    """
    if 'sparse' in filename:
        with open(filename, 'rb') as f:
            shape = (np.fromfile(f, dtype='int32', count=1)[0],
                     np.fromfile(f, dtype='int32', count=1)[1])

            nnz = np.fromfile(f, dtype='int32', count=1)[0]
            data = np.fromfile(f, dtype='int32', count=nnz)
            mat = np.reshape(data, shape)
    else:
        mat = np.load(filename)
    return mat


def _load(filename):
    log("loadfile", filename)
    mat = load_matrix(filename)
    if mat.dtype == np.int8:
        mat = torch.from_numpy(mat).to(torch.long)
    else:
        mat = torch.from_numpy(mat).to(torch.float)
    return mat


def _load_multi(dir, cate, selected):
    data_dict = {}
    for file in all_dataset_files:
        if file not in selected:
            continue
        tag = file
        file = all_dataset_files[file].format(cate=cate)
        filepath = zzz._todir(dir) + file
        if os.path.exists(filepath):
            data = _load(filepath)
            data_dict[tag] = data
        else:
            warn("Could not load " + filepath)

    return data_dict


def _dataset_get_data(self, index):
    return tuple(self.data[index])


def _dataset_select(self, names=[]):
    datas = self.load_data(names)
    data = []
    onekey = list(datas.keys())[0]
    for j in range(len(datas[onekey])):
        one = []
        for i in names:
            one.append(datas[i][j])
        data.append(one)
    self.data = data
    return data


def dataset(class_name, filename="", dir="", subdir=None):
    log("Add dataset", class_name, filename, dir, subdir)
    base_class = torch.utils.data.Dataset
    directory = zzz._todir(dir)
    if subdir:
        directory = zzz._todir(dir) + subdir
    # Define a new class that inherits from the provided base class
    class_attributes = {
        'length': -1,
        '__init__':
        lambda self: setattr(self, "filename", filename),
        'load_data':
        lambda self, selected: _load_multi(directory, filename, selected),
        'choose':
        _dataset_select,
        '__len__':
        lambda self: self.length if self.length > 0 else len(self.data),
        '__getitem__':
        _dataset_get_data
    }
    new_class = type(class_name, (base_class, ), class_attributes)
    return new_class


def dataloader(_dataset, batch_size=32, shuffle=True):
    # print("loader", "batch_size", batch_size, "shuffle", shuffle)
    loader = torch.utils.data.DataLoader(dataset=_dataset,
                                         batch_size=batch_size,
                                         shuffle=shuffle)
    return loader
