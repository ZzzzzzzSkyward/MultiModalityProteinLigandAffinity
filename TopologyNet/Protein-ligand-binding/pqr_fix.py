import os


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
    root_dir = 'z:/refined-set/'
    process_pqr_files(root_dir)
