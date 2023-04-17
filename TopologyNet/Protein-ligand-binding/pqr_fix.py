'''
pdb2pqr有个bug
HETATM22787  H1  HOH   438       7.662  78.883  71.116  0.4170 0.0000
数字连到一起了
但是发现并不影响，因为根本不用这个字段
'''

import os

root_dir = 'z:/refined-set/'


def fix_pqr_format(pqr_file):
    with open(pqr_file, 'r') as f:
        lines = f.readlines()

    for i, line in enumerate(lines):
        if line.startswith('HETATM'):
            if len(line.split()[0]) > 6:
                lines[i] = line[:6] + ' ' + line[6:]

    with open(pqr_file, 'w') as f:
        f.write(''.join(lines))


def process_pqr_files(root_dir):
    for dirpath, dirnames, filenames in os.walk(root_dir):
        for filename in filenames:
            if filename.endswith('.pqr'):
                pqr_file = os.path.join(dirpath, filename)
                fix_pqr_format(pqr_file)


if __name__ == '__main__':
    process_pqr_files(root_dir)
