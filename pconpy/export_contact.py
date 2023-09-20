'''
pconpy
export distance map and contact map for given pdb file
you must install dssp first, such as 'conda install -c salilab dssp'
run 'mkdssp' to check if dssp is installed successfully
'''
import os
import multiprocessing
import argparse
parser = argparse.ArgumentParser()
parser.add_argument(
    '-fixchain',
    action='store_true',
    help='fix chain id')
parser.add_argument(
    '-parallel',
    action='store_true',
    help='use parallel processing')
args = parser.parse_args()
print(args)
# 设置PDB文件夹路径和输出文件夹路径
pdb_folder = './'
output_folder = 'contact_map'


def export(pdb_folder, folder, filename, outputname, output_folder):
    pdb_path = os.path.join(pdb_folder, folder, filename)
    output_path = os.path.join(output_folder,
                               outputname[:-4] + '.txt')
    if os.path.exists(output_path):
        return
    # 生成接触图
    os.system(
        f'python ../pconpy/pconpy/pconpy.py dmap --plaintext --pdb {pdb_path} --chains A -o {output_path}')


def export_detect_chain(pdb_folder, folder, filename,
                        outputname, output_folder):
    # try to read a chain id from the pdb file
    pdb_path = os.path.join(pdb_folder, folder, filename)
    output_path = os.path.join(output_folder, outputname[:-4] + '.txt')
    if os.path.exists(output_path):
        return
    # 生成接触图
    os.system(
        f'python ../pconpy/pconpy/pconpy.py dmap --plaintext --pdb {pdb_path} -o {output_path}')


def run_single(folder, fn=export):
    flag = False
    for filename in os.listdir(folder):
        if filename.endswith('_output.pdb'):
            fn(
                pdb_folder,
                folder,
                filename,
                filename.replace(
                    "_protein_output",
                    ""),
                output_folder)
            flag = True
    if not flag:
        for filename in os.listdir(folder):
            if filename.endswith('.pdb'):
                fn(
                    pdb_folder,
                    folder,
                    filename,
                    filename.replace(
                        "_protein",
                        ""),
                    output_folder)
                flag = True


def export_contact_single(fn=export):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    # 遍历PDB文件夹下的所有文件
    for folder in os.listdir(pdb_folder):
        if not os.path.isdir(folder):
            continue
        run_single(folder,fn)


def export_contact_parallel(fn=export):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    # 遍历PDB文件夹下的所有文件
    pool = multiprocessing.Pool()
    for folder in os.listdir(pdb_folder):
        if not os.path.isdir(folder):
            continue
        pool.apply_async(run_single, (folder, fn))
    pool.close()
    pool.join()


if __name__ == '__main__':
    if args.parallel:
        export_contact_parallel(args.fixchain and export_detect_chain or export)
    else:
        export_contact_single(args.fixchain and export_detect_chain or export)
