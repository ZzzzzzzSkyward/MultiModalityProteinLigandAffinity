# Dependency---------------------------------------------------------------------------------------
# Javaplex                                                                                        |
# download matlab-examples from https://github.com/appliedtopology/javaplex/releases/tag/4.3.1    |
# unzip, rename the folder as javeplex and put it in the same folder where this script is         |
# that is, put javaplex.jar in javaplex/                                                          |
# -------------------------------------------------------------------------------------------------|
# R-TDA                                                                                           |
# install TDA package in R and properly set the R library path                                    |
# -------------------------------------------------------------------------------------------------|
# Python packages                                                                                 |
# Numpy, Pickle                                                                                   |
# --------------------------------------------------------------------------------------------------

import threading
from tqdm import tqdm
from timeout import run
from multiprocessing import Pool
import numpy as np
import os
import TopBio.ReadFile.ReadMOL2 as ReadMOL2
import TopBio.ReadFile.ReadPDB as ReadPDB
import TopBio.ReadFile.ReadPQR as ReadPQR
import TopBio.PersistentHomology.PHSmallMolecule as PHSmallMolecule
import TopBio.PersistentHomology.PHComplex as PHComplex
import TopBio.Feature.LigandFeature as LigandFeature
import TopBio.Feature.ComplexFeature as ComplexFeature

# 去掉os.system输出
DEVNULL = open(os.devnull, 'w')
os.oldsys = os.system


def runsys(command):
    # print(command)
    if command.find('matlab') >= 0:
        run(command, 60)
    else:
        os.oldsys(f'{command} > ' + os.devnull)


os.system = runsys

# 防止报错


def trial(func):
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            print(e)
            pass
    return wrapper


class tryc:
    def __init__(self, func):
        self.func = func

    def __call__(self, *args, **kwargs):
        try:
            return self.func(*args, **kwargs)
        except Exception as e:
            # print(e)
            pass


forcestop = False
skipfailed = False
Clean = False
base = 'z:/refined-set/'


def gen_pqr(dirname):
    working_dir = base + dirname
    if not os.path.exists(working_dir):
        return
    protein_name = dirname + '_protein'
    download_protein_name = dirname + "_output"
    if os.path.exists(working_dir + '/' + protein_name + '.pqr'):
        return
    print(dirname)
    os.system("pdb2pqr --ff=AMBER --keep-chain " + working_dir + '/' +
              protein_name + '.pdb ' + working_dir + '/' + protein_name +
              '.pqr')


@ tryc
def fn(dirname):
    global skipfailed
    global forcestop
    if forcestop:
        return
    working_dir = base + dirname
    if not os.path.exists(working_dir):
        return
    protein_name = dirname + '_protein'
    # print(dirname)
    ligand_name = dirname + '_ligand'
    # check if 1a8i_protein_feature_complex_alpha_1DCNN is there
    if os.path.exists(working_dir + '/' + protein_name +
                      "_feature_complex_alpha_1DCNN.npy"):
        print("done")
        return
    # check if pqr file exists
    if not os.path.exists(working_dir + '/' + protein_name + '.pqr'):
        # if there is a .log file, skip, because it is done before and failed
        if skipfailed and os.path.exists(
                working_dir + '/' + protein_name + '.log'):  # and not os.path.exists(working_dir + '/' + 'tmp.out'):
            return
        # generate one using pdb2pqr
        gen_pqr(dirname)
    # check if pqr file exists
    if not os.path.exists(working_dir + '/' + protein_name + '.pqr'):
        print('pqr file not found for', dirname)
        return
    # check if .pkl file exists
    if not os.path.exists(
            working_dir + '/' + protein_name + '_alpha.pkl'):
        a = ReadMOL2.SmallMolecule(ligand_name, working_dir)
        PHSmallMolecule.Level1_Rips(a, ligand_name, working_dir)
        PHSmallMolecule.Alpha(a, ligand_name, working_dir)
        ReadPDB.get_pdb_structure(a, 50.0, protein_name, working_dir)
        ReadPQR.get_pqr_structure(ligand_name, protein_name, working_dir)
        PHComplex.Interaction_Rips(50.0, protein_name, working_dir)
        PHComplex.Electrostatics_Rips(16.0, protein_name, working_dir)
        PHComplex.Alpha(50.0, protein_name, working_dir)
    if not os.path.exists(working_dir + '/' + protein_name + '_alpha.pkl'):
        print("error: no pkl", dirname)
        return
    '''
    # TopBP-ML
    LigandFeature.GenerateFeature_alpha(ligand_name, working_dir)
    LigandFeature.GenerateFeature_level1(ligand_name, working_dir)
    ComplexFeature.GenerateFeature_interaction_ML(protein_name, working_dir,
                                                  'all')
    ComplexFeature.GenerateFeature_electrostatics_ML(protein_name, working_dir)
    ComplexFeature.GenerateFeature_alpha_ML(protein_name, working_dir,
                                            'carbon')
    ComplexFeature.GenerateFeature_alpha_ML(protein_name, working_dir, 'heavy')
        # TopBP-DL
        ComplexFeature.GenerateFeature_interaction_1DCNN(protein_name, working_dir)
        ComplexFeature.GenerateFeature_electrostatics_1DCNN(
            protein_name, working_dir)
        ComplexFeature.GenerateFeature_alpha_2DCNN(protein_name, working_dir)

        # TopVS-ML
        LigandFeature.GenerateFeature_alpha(ligand_name, working_dir)
        LigandFeature.GenerateFeature_level1(ligand_name, working_dir)
        ComplexFeature.GenerateFeature_interaction_ML(protein_name, working_dir,
                                                    'pair')
        ComplexFeature.GenerateFeature_electrostatics_ML(protein_name, working_dir)
        ComplexFeature.GenerateFeature_alpha_ML(protein_name, working_dir,
                                                'carbon')
        ComplexFeature.GenerateFeature_alpha_ML(protein_name, working_dir, 'heavy')
    '''
    # TopVS-DL
    LigandFeature.GenerateFeature_alpha(ligand_name, working_dir)
    LigandFeature.GenerateFeature_level1(ligand_name, working_dir)
    ComplexFeature.GenerateFeature_interaction_1DCNN(protein_name, working_dir)
    ComplexFeature.GenerateFeature_electrostatics_1DCNN(
        protein_name, working_dir)
    ComplexFeature.GenerateFeature_alpha_1DCNN(protein_name, working_dir)
    '''
        # All features
        LigandFeature.GenerateFeature_alpha(ligand_name, working_dir)
        LigandFeature.GenerateFeature_level1(ligand_name, working_dir)
        ComplexFeature.GenerateFeature_interaction_ML(protein_name, working_dir,
                                                    'all')
        ComplexFeature.GenerateFeature_interaction_1DCNN(protein_name, working_dir)
        ComplexFeature.GenerateFeature_electrostatics_ML(protein_name, working_dir)
        ComplexFeature.GenerateFeature_electrostatics_1DCNN(
            protein_name, working_dir)
        ComplexFeature.GenerateFeature_alpha_ML(protein_name, working_dir,
                                                'carbon')
        ComplexFeature.GenerateFeature_alpha_ML(protein_name, working_dir, 'heavy')
        ComplexFeature.GenerateFeature_alpha_1DCNN(protein_name, working_dir)
        ComplexFeature.GenerateFeature_alpha_2DCNN(protein_name, working_dir)
    '''
    if Clean:
        for i in ['PH', 'csv', 'pts', 'bds']:
            for j in os.listdir(working_dir):
                if j.endswith('.' + i):
                    os.remove(working_dir + '/' + j)
        if os.path.exists(working_dir + '/tmp.out'):
            os.remove(working_dir + '/tmp.out')


def loop():
    arr = os.listdir(base)
    # shuffle
    # np.random.shuffle(arr)
    pbar = tqdm(total=len(arr))
    for i in arr:
        fn(i)
        try:
            pbar.update(1)
        except Exception as e:
            print(i)
            print(e)


def tryfn(z):
    return fn(z)


def loop_parallel():
    arr = os.listdir(base)
    # shuffle
    np.random.shuffle(arr)
    # get 10 of them
    # arr = arr[:10]
    # filter out the ones that starts with 1
    # arr = [i for i in arr if i[0] == '1']

    # define the number of processes to use
    # processes=num_processes
    # create a process pool with the specified number of processes
    pool = Pool(processes=4)  # too many processes can cause out of memory
    pbar = tqdm(total=len(arr))
    # map the function fn to each element in the shuffled array
    for result in pool.imap_unordered(tryfn, arr):
        pbar.update(1)

    # close the process pool
    pool.close()
    pool.join()


def run_thread_loop():
    # run loop_multi from thread
    t = threading.Thread(target=loop_parallel)
    t.start()
    # leave it at background
    # set forcestop=True to exit #???useless


if __name__ == '__main__':
    loop_parallel()
    # loop()
