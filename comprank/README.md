# Compound Prioritization

This repository is the official implementation of compound prioritization experiments presented in [Improving Compound Activity Classification via Deep Transfer and Representation Learning](link). 

## Requirements

Operating systems: Red Hat Enterprise Linux Server 7.9

To install requirements:

```bash
pip install -r requirements.txt
```

## Data 

### Use provided processed dataset

One can use our provided processed dataset in `./data/`: the dataset of assays. 
We provided our dataset of pairs as `data/active_unique_50.tar.gz` compressed file. Please use tar to de-compress it.

## Experiments

### 1. Running `dmpn`/`dmpna`

- To run `dmpn`,

```bash
python src/cross_validate.py --data_path <assay_csv_file> --smiles_path <smiles_tsv_dict> -d 50 -T 3 --max_iter 50 -b 512 -lr 5e-3 -reg 1e-6 -save <chkpt_dir> --optim adam -loss pair2 --do_train --do_test --do_train_eval
```

- To run `dmpna`, add these arguments to the above command

```bash
--attn_dim 10 --pool self-attention
```

`data_path` specify the path to the assay CSV file. First line contains a header `smiles,target`. Each of the following lines are comma-separated with the SMILES in the 1st column and the 0/1 label in the 2nd column.

`d` specifies the dimension of the learned compound representation out of `GNN`-based feature generators.

`T` specifies the number of message passing steps.

`lr` specifies the learning rate.

`b` specifies the batch size.

`reg` specifies the coefficient on L2 regularization.

`max_iter` specifies the total number of epochs.

`save` specifies the directory where the model, evaluation scores and predictions will be saved.

`pool` specifies which pooling mechanism to use to get the compound representation from the atom representations. Default set to `mean`: the atom-level representations from the message passing network are averaged over all atoms of a compound to yield the compound representation.

`attn_dim` specifies the dimension of the hidden layer in the 2-layer fully connected network used as the attention network.

Use `python src/cross_validate.py -h` to check the meaning and default values of other parameters.

### 2. Running baselines

### `morgan`

```bash
python src/cross_validate.py --data_path <assay_csv_file> --smiles_path <smiles_tsv_dict> -d 2048 -T 2 --max_iter 50 -b 512 -lr 5e-3 -reg 1e-6 -save <chkpt_dir> --model baseline --use_features morgan --optim adam -loss pair2 --do_train --do_test --do_train_eval
```

### `morgan-c`

```bash
python src/cross_validate.py --data_path <assay_csv_file> --smiles_path <smiles_tsv_dict> -d 2048 -T 2 --max_iter 50 -b 512 -lr 5e-3 -reg 1e-6 -save <chkpt_dir> --model baseline --use_features morgan_count --optim adam -loss pair2 --do_train --do_test --do_train_eval
```

### `morgan-ba`

```bash
python src/cross_validate.py --data_path <assay_csv_file> --smiles_path <smiles_tsv_dict> -d 2048 -T 2 --max_iter 50 -b 512 -lr 5e-3 -reg 1e-6 -save <chkpt_dir> --model baseline --use_features morgan_tanimoto_bioassay --optim adam -loss pair2 --do_train --do_test --do_train_eval
```

### `RDKit200`

```bash
python src/cross_validate.py --data_path <assay_csv_file> --smiles_path <smiles_tsv_dict> -d 2048 -T 2 --max_iter 50 -b 512 -lr 5e-3 -reg 1e-6 -save <chkpt_dir> --model baseline --use_features rdkit_2d --optim adam -loss pair2 --do_train --do_test --do_train_eval
```

