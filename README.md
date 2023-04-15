# A Multi-Modal Framework for Protein-Ligand Affinity and Contact Prediction

## Motivation

Recent protein-ligand affinity and contact prediction tasks mainly depends on amino acid sequence, 2d graph and chemical fingerprint data to extract the features of proteins and ligands. As an attempt, cross modality models are developed to combine the strengths of different views of molecules, which includes 1d+2d, 2d+3d pattern that either utilizes all modalities or treat some as priviledged resource. Considering the scarcity of high dimensional data, methods such as pretraining and unsupervised learning are also brought into this field. As another attempt, it is possible that multiple views of molecule containing varied features may as well enhance model's understanding of the protein, the ligand, and their interaction.

## Results

[Todo]

## Data

All protein-ligand pairs used in the experiment are from PDBBind 2020 refined dataset. You can download it from [PDBBind](http://pdbbind.org.cn/). By the time the experiment carries out, PDBBind 2021 dataset is still not available. You can also use other sources as long as `pdb`, `sdf` and `mol2` files are available.

All scripts to process the data are put in `preprocess`, `biozernike` and `TopologyNet`. The experiment is designed to use 8 types of data: amino acid sequence of protein, SMILES of ligand, contact map of protein, adjacent matrix of ligand, topology of both protein and ligand, zernike descriptor of protein,.

For experiment, the dataset are split into train set, validate set(both present), and test set(protein only, ligand only and both none) after a shuffle. The ligands' identity are fully determined by its name in the index file.

| Dataset\Category | Length | Protein | Ligand | Unique protein | Unique ligand |
| ---------------- | ------ | ------- | ------ | -------------- | ------------- |
| Train            | 4532   | 3174    | 3460   |                |               |
| Validate         | 147    | 119     | 120    |                |               |
| Unique Protein   | 176    | 570     | 564    | 344            | 453           |
| Unique ligand    | 291    |         |        |                |               |
| Both unique      | 170    |         |        |                |               |
| sum              | 5317   | 3863    | 4144   |                |               |

[Todo]

## Experiments

The provided scripts are using pytorch 1.11 and cuda 11.3.  All required python libraries can be found in `header.py.`

[Todo]

## Citation

[Todo]
