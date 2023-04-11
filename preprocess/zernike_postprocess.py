import os
import numpy as np

# Load the Zernike moment files
workingdir = "z:/pdb/"
pdb_dir = "D:/pdb/"
data_dir = "z:/"
charge_file = workingdir + "{pdb}.moment"
geo_file = workingdir + "{pdb}.geo"

# Load the list of PDB IDs
with open(pdb_dir + "pdbid.txt", "r", encoding='utf-8') as f:
    pdb_list = [line.strip() for line in f]
# Define the dimensions of the data matrix
num_pdb = len(pdb_list)
#incase the process is not finished and you want to see
#for i, pdb_id in enumerate(pdb_list):
#    if os.path.exists(charge_file.format(pdb=pdb_id)) and os.path.exists(
#            geo_file.format(pdb=pdb_id)):
pdb_id = pdb_list[0]
charge_dim = np.genfromtxt(charge_file.format(pdb=pdb_id),
                           delimiter="\n").shape[0]
geo_dim = np.genfromtxt(geo_file.format(pdb=pdb_id), delimiter="\n").shape[0]
matrix_dim = charge_dim + geo_dim

# Create an empty matrix to store the data
data_matrix = np.zeros((num_pdb, matrix_dim))
# Loop through the PDB IDs in the list and store the corresponding data in the matrix
for i, pdb_id in enumerate(pdb_list):
    if os.path.exists(charge_file.format(pdb=pdb_id)) and os.path.exists(
            geo_file.format(pdb=pdb_id)):
        charge_data = np.genfromtxt(charge_file.format(pdb=pdb_id),
                                    delimiter="\n")
        geo_data = np.genfromtxt(geo_file.format(pdb=pdb_id), delimiter="\n")
        vector = np.concatenate((charge_data, geo_data))
        data_matrix[i] = vector
    else:
        print("Could not load data for " + pdb_id)
        #raise FileNotFoundError

# Save the data matrix to a file
#np.savetxt("data_matrix.txt", data_matrix, delimiter="\t")
np.save(data_dir + "data_matrix.npy", data_matrix)
