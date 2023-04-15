'''
This script is used to save the topology files to a zip file.
Because my working dir is in a ramdisk so I have to save data.
'''
import zipfile
import os
import tqdm
import time
import shutil

path = "D:/pdb/"
temppath = "z:/refined-set/"
storepath = 'o:/refined-set/'
file_paths = os.listdir(temppath)

idx = 12
temp = 4


def fn():
    progress = tqdm.tqdm(total=len(file_paths))
    count = 0
    with zipfile.ZipFile(path + f'topology_temp{temp}.zip', 'w',
                         zipfile.ZIP_DEFLATED) as zipf:
        for file_path in file_paths:
            progress.update(1)
            # zipf.write(temppath + file_path)
            unwanted = [
                'pdb', 'mol2', 'sdf', 'log', 'csv', 'pts', 'PH', 'bds', 'out'
            ]
            flag = 0
            doneflag = 0
            # check the file protein_feature_complex_interaction_1DCNN.npy
            # exists
            for file in os.listdir(temppath + file_path):
                if any(file.endswith(ext) for ext in unwanted):
                    continue
                if file.endswith(
                        'protein_feature_complex_alpha_1DCNN.npy'):
                    doneflag = 1
                zipf.write(temppath + file_path + '/' + file)
                flag = 1
            count += flag
            # delete this folder if done
            if doneflag:
                pass
                # shutil.rmtree(temppath + file_path)
    progress.close()
    print(time.strftime("%m-%d %H:%M:%S", time.localtime()))
    print("total count:", count)


def fn2():
    progress = tqdm.tqdm(total=len(file_paths))
    count = 0
    for file_path in file_paths:
        progress.update(1)
        # zipf.write(temppath + file_path)
        unwanted = [
            'pdb', 'mol2', 'sdf', 'log', 'csv', 'pts', 'ph', 'bds', 'out'
        ]
        flag = 0
        doneflag = 0
        # check the file protein_feature_complex_interaction_1DCNN.npy exists
        for file in os.listdir(temppath + file_path):
            if file.endswith(
                    'protein_feature_complex_interaction_1DCNN.npy'):
                doneflag = 1
                break
        if doneflag:
            for file in os.listdir(temppath + file_path):
                if any(file.endswith(ext) for ext in unwanted):
                    continue
                fpath = temppath + file_path + '/' + file
                if os.path.exists(fpath):
                    # try make dir
                    if not os.path.exists(storepath + file_path + '/'):
                        os.makedirs(storepath + file_path + '/')
                    # move to storepath
                    shutil.move(fpath, storepath + file_path + '/')
            flag = 1
            # delete this folder if done
            shutil.rmtree(temppath + file_path)
        count += flag
    progress.close()
    print(time.strftime("%m-%d %H:%M:%S", time.localtime()))
    print("total count done:", count)


def loop():
    global idx
    j = 0
    while j > -1:
        j += 1
        try:
            fn()
        except BaseException:
            pass
        time.sleep(60 * 10)
        if j % 6 == 0:
            try:
                fn2()
            except BaseException:
                pass
            idx += 1


if __name__ == "__main__":
     #loop()
    # time.sleep(60 * 60*2)
    fn2()
