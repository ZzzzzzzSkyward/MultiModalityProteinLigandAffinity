from header import *
from logg import warn
all_dataset_files = {
    # the amino sequence of proteins
    "seq": "{cate}_sequence",
    # the smiles of compounds
    'smiles': "{cate}_smiles",
    # the information of compounds, in adjacent matrix
    'adjacent': "{cate}_adjacent",
    # protein contact
    'contact': "{cate}_contact",
    # label of contact
    'interact': "{cate}_interact",
    # label of compound-protein interaction
    'logk': "{cate}_logk"
}


def _load_multi(dir, cate, selected):
    data_dict = {}
    for file in all_dataset_files:
        if file not in selected:
            continue
        tag = file
        file = all_dataset_files[file].format(cate=cate)
        filepath = zzz._todir(dir) + file
        if os.path.exists(filepath):
            data = np.load(filepath)
            data_dict[tag] = data
        else:
            warn("Could not load " + filepath)

    return data_dict


def _dataset_get_data(self, index):
    return self.data[index]


def dataset_select(self, names=[]):
    datas = self.load_data(names)
    data = []
    onekey=list(datas.keys())[0]
    for j in range(len(datas[onekey])):
        one = []
        for i in names:
            one.append(datas[i][j])
        data.append(one)
    self.data = data
    return data


def dataset(class_name, filename="", dir="", subdir=None):
    base_class = torch.utils.data.Dataset
    directory = zzz._todir(dir)
    if subdir:
        directory = zzz._todir(dir) + subdir
    # Define a new class that inherits from the provided base class
    class_attributes = {
        '__init__': lambda self: setattr(self, "filename", filename),
        'load_data': lambda self, selected: _load_multi(directory, filename, selected),
        'choose': dataset_select,
        '__len__': lambda self: len(self.data),
        '__getitem__': _dataset_get_data
    }
    new_class = type(class_name, (base_class,), class_attributes)
    return new_class


def dataloader(_dataset, batch_size=32, shuffle=True):
    loader = torch.utils.data.DataLoader(
        dataset=_dataset,
        batch_size=batch_size,
        shuffle=shuffle)
    return loader
