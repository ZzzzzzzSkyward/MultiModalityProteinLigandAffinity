import zipfile
import os
import tqdm
import time
import shutil

path = "D:/pdb/"
temppath = "z:/refined-set/"
file_paths = os.listdir(temppath)


def fn():
    progress = tqdm.tqdm(total=len(file_paths))
    count = 0
    with zipfile.ZipFile(path + 'topology_temp3.zip', 'w',
                         zipfile.ZIP_DEFLATED) as zipf:
        for file_path in file_paths:
            progress.update(1)
            #zipf.write(temppath + file_path)
            unwanted = [
                'pdb', 'mol2', 'sdf', 'log', 'csv', 'pts', 'ph', 'bds', 'out'
            ]
            flag = 0
            doneflag = 0
            #check the file protein_feature_complex_interaction_1DCNN.npy exists
            for file in os.listdir(temppath + file_path):
                if any(file.endswith(ext) for ext in unwanted):
                    continue
                if file.endswith(
                        'protein_feature_complex_interaction_1DCNN.npy'):
                    doneflag = 1
                zipf.write(temppath + file_path + '/' + file)
                flag = 1
            count += flag
            #delete this folder if done
            if doneflag:
                pass
                #shutil.rmtree(temppath + file_path)
    progress.close()
    print(time.strftime("%m-%d %H:%M:%S", time.localtime()))
    print("total count:", count)


while 1:
    fn()
    time.sleep(60 * 10)
