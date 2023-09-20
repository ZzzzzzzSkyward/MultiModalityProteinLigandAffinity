'''
只能用这个丐版
使用pconpy了
'''
import sys
import math
import argparse
import re
from itertools import combinations


def dist(p1, p2):
    dx = p1[0] - p2[0]
    dy = p1[1] - p2[1]
    dz = p1[2] - p2[2]
    return math.sqrt(dx**2 + dy**2 + dz**2)


def read_atoms(file, chain=".", model=1):
    pattern = re.compile(chain)

    current_model = model
    atoms = []
    for line in file:
        line = line.strip()
        if line.startswith("ATOM") and current_model == model:
            type = line[12:16].strip()
            chain = line[21:22]
            if type == "CA" and re.match(pattern, chain):
                x = float(line[30:38].strip())
                y = float(line[38:46].strip())
                z = float(line[46:54].strip())
                atoms.append((x, y, z))
        elif line.startswith("MODEL"):
            current_model = int(line[10:14].strip())
    return atoms


def compute_contacts(atoms, threshold):
    contacts = []
    for i in range(len(atoms)-2):
        for j in range(i+2, len(atoms)):
            if dist(atoms[i], atoms[j]) < threshold:
                contacts.append((i+1, j+1))
    return contacts


def write_output(contacts, file):
    for c in contacts:
        file.write("\t".join(map(str, c))+"\n")


def pdb_to_cm(file, threshold, chain=".", model=1):
    atoms = read_atoms(file, chain, model)
    return compute_contacts(atoms, threshold)