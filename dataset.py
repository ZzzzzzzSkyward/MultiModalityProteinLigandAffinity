from header import *
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


def load_multi(dir, cate, selected):
    # Load each .npy file and append to a list
    data_dict = {}
    for file in all_dataset_files:
        if file not in selected:
            continue
        file = all_dataset_files[file].format(cate)
        filepath = dir + "/" + file
        if os.path.exists(filepath):
            data = np.load(filepath)
            data_dict[file] = data
        else:
            warn("Could not load " + file)

    return data_dict


def dataset_get_data(self, index):
    return self.data[index]


def dataset_select(self, names):
    datas = self.load_data(names)
    data = []
    for i in names:
        data.append(datas[i])
    self.data = data


def dataset(class_name, filename="", dir="", subdir=None):
    base_class = torch.utils.data.Dataset
    directory = dir
    if subdir:
        directory = dir + "/" + subdir
    # Define a new class that inherits from the provided base class
    class_attributes = {
        '__init__': lambda self: setattr(self, "filename", filename),
        'load_data': lambda self, selected: load_multi(directory, filename, selected),
        'select': dataset_select,
        '__len__': lambda self: len(self.data[0][0]),
        '__getitem__': dataset_get_data
    }
    new_class = type(class_name, (base_class,), class_attributes)
    return new_class


def dataloader(_dataset, batch_size=32):
    loader = torch.utils.data.DataLoader(
        dataset=_dataset,
        batch_size=batch_size,
        shuffle='train' in _dataset.__name__)
    return loader
