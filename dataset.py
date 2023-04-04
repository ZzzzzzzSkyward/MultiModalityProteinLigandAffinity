from header import *
all_dataset_files = [
    # the amino sequence of proteins, and other information
    "protein_{cate}.npy",
    # the information of compounds , in matrix
    "compound_{cate}_verify.npy",
    # the information of compounds, in adjacent matrix
    "compound_{cate}_adjacent.npy",
    # protein contact
    "protein_{cate}_contacts.npy",
    # "protein_{cate}_contacts_true.npy",#???#maybe ground truth contact
    "protein_{cate}_interact.npy",
    # label of compound-protein interaction
    "label_interact_{cate}.npy",  # value∈{0,1}
    # label of protein interaction
    "label_protein_{cate}.npy"
]


def load_multi(dir, cate):
    # Load each .npy file and append to a list
    data_list = []
    for file in all_dataset_files:
        file = file.format(cate)
        filepath = os.path.join(dir, file)
        data = np.load(filepath)
        data_list.append(data)

    # Concatenate the list of arrays into a single array
    merged_data = np.concatenate(data_list, axis=0)

    return merged_data


# Example usage:
# MyCustomDataset = create_new_dataset_class('MyCustomDataset', base_class=torch.utils.data.Dataset,
#                                           filename='my_data.npy')
def dataset(class_name, filename="", dir=""):
    base_class = torch.utils.data.Dataset
    # Define a new class that inherits from the provided base class
    class_attributes = {
        '__init__': lambda self: self.load_data(filename),
        'load_data': lambda self, file: setattr(self, 'data', load_multi(dir, file)),
        '__len__': lambda self: len(self.data),
        '__getitem__': lambda self, index: self.data[index]
    }
    new_class = type(class_name, (base_class,), class_attributes)
    return new_class
