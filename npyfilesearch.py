import zipfile
import os


def find_npy_files(directory):
    npy_files = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith('.py'):
                with open(os.path.join(root, file), 'r', encoding='utf-8') as f:
                    contents = f.read()
                    if "np.load(" in contents:
                        lines = contents.split('\n')
                        for line in lines:
                            if "np.load(" in line:
                                start_index = line.find('np.load(') + 8
                                end_index = line.find(')', start_index)
                                filename = line[start_index:end_index].strip(
                                    "'\"")
                                npy_file_path = filename
                                npy_files.append(npy_file_path)
    return npy_files


def erase_dup():
    with open('npyfiles.txt', 'r') as f:
        files = list(set(f.read().splitlines()))
    with open('npyfiles.txt', 'w') as f:
        f.write("\n".join(files))

def compare():
    prefix="data_processed/"
    # Open zip file
    with zipfile.ZipFile('e:/data.zip', 'r') as z:
        # Get a list of file names in the zip archive
        zip_file_names = [i[len(prefix):] for i in z.namelist()]

    # Load file names from previous extraction
    with open('npyfiles.txt', 'r') as f:
        extracted_file_names = set(f.read().splitlines())

    # Find files that are in the zip but not in the previous extraction
    new_files = set(zip_file_names) - extracted_file_names

    # Print the new file names
    print("\n".join(new_files))

if __name__ == '__main__':
    # f=find_npy_files("./")
    # with open("npyfiles.txt","w",encoding='utf-8') as file:
    # file.write("\n".join(f))
    compare()
