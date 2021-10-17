This file details the dataset generation: collecting bioassays, initial bioassay selection and pruning, and creating transferable bioassay pairs.

## Collecting bioassay data
We collected PubChem bioassay data from NCBI Entrez with the following criteria:
- Substance Type: Chemical
- Screening Stage: Confirmatory
- Target: Single
- Target Type: Protein Target


## Initial bioassay selection and pruning

### Processing raw bioassays
Use `process_raw.py` to combine assays if they correspond to the same target and to select compounds which are specified as 'Inhibitors' in Phenotype column. If there are duplicate entries SMILES with same labels, keep one of such entries. If there are duplicate entries SMILES with different labels, discard all entries of that compound from the bioassay.

```bash
python process_raw.py --assay_dir raw_assay_dir --cid_smiles_path mapped_cid_smiles_file --choosen_pfam_path choosen_pfams_file --output_dir processed_assay_dir --target_aids_list_path target_aids_file
```
Below, we describe the format of input files and how to create them:

`assay_dir` specifies the path to the directory containing all the downloaded bioassays from PubChem as explained above.

`cid_smiles_path` specifies the path to the file containing the list of mapped CIDs and their corresponding canonical SMILES. The first column in this file must contain the CID and the 2nd column must contain the canonical SMILES; two columns are tab-separated.

- First get all the CIDs used in the bioassays in `cid_list.txt`.
- Download and extract PubChem mapped `CID-SMILES` file from [here](https://ftp.ncbi.nlm.nih.gov/pubchem/Compound/Extras/CID-SMILES.gz) and map CIDs to SMILES. Store the mapping for each compound in `cid_list.txt` into `mapped_unique_cid_smiles.txt`. Ensure SMILES are canonical using `standardize_smiles.py` and stores the CID and canoncial SMILES in tab-separated format in `standardized_cid_smiles.txt`.

```bash
python standardize_smiles.py mapped_unique_cid_smiles.txt standardized_cid_smiles.txt
```

`choosen_pfam_path` specifies the path to the file containing the list of protein families one is interested in -- one protein family name in one line. We used UniProt to get protein family information for our targets, and used the top-10 protein families with the most number of corresponding protein targets from `Human` organism.

`output_dir` specifies the path to the directory where the processed assays will be stored.

`target_aids_list_path` specifies the path to the file containing the list of target and associated AIDs in the dataset. The first column contains the target accession ID; the second column contains space-separated list of PubChem assay identifiers (AIDs); and the third column contains the protein family name. The three columns are separated by tabs.
For example, a line in this file can be: 

```setup
EAW86722<\t>2251 463251 485339 504868 540365 588564 588566 588583<\t>G-protein coupled receptor 1 family
```

After the above processing, each generated CSV file contains a list of inhibitor compounds with respective 0/1 labels. Each file must have a header `smiles,target`. We denote each generated file with corresponding target accession ID. Each file corresponds to a combined assay with all the information from all the assays for the respective target.

### Selection of bioassays
- Use only those processed assays which has more than 50 active compounds.
- Find any protein families that has only 1 associated assay with more than 50 active compounds. Remove all those assays since they can't be used for pairing.


## Transferable bioassay pairing

### Pairing assays

- Construct bioassay pairs such that in each pair, the protein targets of the two bioassays are from the same protein family.

```bash
python process_assay_pairs.py --data_path_1 <target1> --data_path_2 <target2> --save_dir save_dir --smiles_path all_smiles --balance 
```

`data_path_1` and `data_path_2` specify the paths to two CSV files containing processed assays such that the corresponding targets belong to the same protein family.

`save_dir` specify the path to the output directory where the processed assays are stored.

`smiles_path` specify the path to a file containing a list of SMILES (each in one line) collected from PubChem.

`balance` denotes whether to create class-balanced assays. 

- To create all possible pairs for selected targets, run

```bash
bash pairing.sh <choosen_targets_for_pairing> <log>
```
This script runs `process_assay_pairs.py` which creates pairs of assays such that each assay is balanced in terms of active and inactive compounds, and there are no duplicate compounds with same/different labels within and across assays.

`choosen_targets_for_pairing` contains the list of selected targets to be used to get assay pairs. 

`log` file contains the compound statistics printed for each assay pair, before and after processing.

Details are available in our paper in `Section 5.1.2`.

### Get assay pairwise similarities
Compute assay pair similarity for all pairs chosen with at least 50 active in each assay of the pair
```bash
bash run_pairwise_similarity.sh <choosen_targets_for_pairing> <output>
```

This script runs `pairwise_similarity.py`. Run `python pairwise_similarity.py -h` for details.

`choosen_targets_for_pairing` contains the list of selected targets to be used to get assay pairs. 

`output` file contains the average pairwise similarities computed using Tanimoto coefficient.

### Filter pairs based on the similarities

1. We first selected a set of bioassay pairs $\mathcal{P}_0$ such that in each pair, the active compounds of two bioassays are more similar.
2. From $\mathcal{P}_0$, we further selected a set of bioassay pairs $\mathcal{P}$, such that in each pair, the active compounds in two bioassays have a similarity above a certain threshold. Check our paper for more details.

We stored the assay pairs in `./data/pairs/` in this directory structure: `./data/pairs/<pfam>/<targetA-targetB>/`, where `<pfam>` denotes the protein family, `targetA` and `targetB` are the corresponding protein accession IDs.

Each such directory further contains the two assay files named as `<target>-0.0.csv`; and a directory `crossval_splits`.

The directory `crossval_splits` further contain two directories; one for each target. Each subdirectory contains the `stratified` directory that contain the 10-fold cross-validation indices for each fold stored as `.pckl` files. Each pickle file contains 3 lists; each for training, validation and testing. To read `.pckl` files, you need pickle library.

To create 10-fold cross validation splits and corresponding indices files, run

```bash
bash run_split.sh <pairs_list> <output_dir>
```

