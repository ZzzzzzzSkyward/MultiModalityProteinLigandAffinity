'''
使用分子动力学软件Gromacs对蛋白质进行预处理
但是由于计算量过大而放弃
'''
import os
import subprocess
from multiprocessing import Pool
from tqdm import tqdm

gmxcommand = 'gmx pdb2gmx -f {}.pdb -o {}.gro -water tip3p -ff amber03 -ignh'
editconfcommand = 'gmx editconf -f {}.gro -o {}_box.gro -c -d 1.0 -bt cubic'
solvantcommand = 'gmx solvate -cp {}_box.gro -cs spc216.gro -o {}_solv.gro -p {}.top'
combinecmd = 'gmx grompp -f conf.mdp -c {}_solv.gro -p {}.top -o {}.tpr -maxwarn 10'
thiscmd = gmxcommand
allcmd = [gmxcommand, editconfcommand, solvantcommand, combinecmd]

ignore_error = True
parallel = True
fix_pdb = True


def process_folders():
    folders = [folder for folder in os.listdir('.') if os.path.isdir(folder)]
    with Pool() as pool:
        for _ in tqdm(pool.imap_unordered(processf, folders),
                      total=len(folders)):
            pass


def process_folders_one():
    folders = [folder for folder in os.listdir('.') if os.path.isdir(folder)]
    for i, f in tqdm(enumerate(folders), total=len(folders)):
        processf(f)


def processf(folder):
    # prefer fixed
    pdb_file = f'{folder}/{folder}_protein_output.pdb'
    if not os.path.exists(pdb_file):
        pdb_file = f'{folder}/{folder}_protein.pdb'
    gro_file = f'{folder}/{folder}_protein.gro'
    if not os.path.exists(pdb_file) or os.path.exists(gro_file):
        return
    cmd = thiscmd.replace('{}', folder)
    runcmd(cmd)
    if not os.path.exists(gro_file):
        print(folder, "failed")
    cmd = allcmd[1].replace('{}', folder)
    runcmd(cmd)
    cmd = allcmd[2].replace('{}', folder)
    runcmd(cmd)
    cmd = allcmd[3].replace('{}', folder)
    runcmd(cmd)


def runcmd(cmd):
    subprocess.run(
        cmd,
        shell=True,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE if ignore_error else None)


if __name__ == '__main__':
    if parallel:
        process_folders()
    else:
        process_folders_one()
