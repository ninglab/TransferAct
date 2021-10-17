```
 File              : README.md
 Author            : Vishal Dey <dey.78@osu.edu>
 Date              : Sat 16 Oct 2021 20:48:41
 Last Modified Date: Sat 16 Oct 2021 20:50:00
 Last Modified By  : Vishal Dey <dey.78@osu.edu>
```
# Improving Compound Activity Classification via Deep Transfer and Representation Learning

This repository is the official implementation of [Improving Compound Activity Classification via Deep Transfer and Representation Learning](link). This paper is under review by XXX.

## Requirements

Operating systems: Red Hat Enterprise Linux Server 7.9

To install requirements:

```bash
pip install -r requirements.txt
```

## Installation guide

Download the code and dataset with the command:

```bash
git clone https://github.com/ninglab/Tacfc.git
```

## Data Processing

### 1. Use provided processed dataset

One can use our provided processed dataset in `./data/pairs/`: the dataset of pairs of processed balanced assays $\mathcal{P}$ . Check the details of bioassay selection, processing, and assay pair selection in our paper in `Section 5.1.1` and `Section 5.1.2`, respectively. 
We provided our dataset of pairs as `data/pairs.tar.gz` compressed file. Please use tar to de-compress it.

### 2. Use own dataset

We provide necessary scripts in `./data/scripts/` with the processing steps in `./data/scripts/README.md`. 

## Training

### 1. Running `TAc`

- To run `TAc-dmpn`,

```bash
python code/train_aada.py --source_data_path <source_assay_csv_file> --target_data_path <target_assay_csv_file> --dataset_type classification --extra_metrics prc-auc precision recall accuracy f1_score --hidden_size 25 --depth 4 --init_lr 1e-3 --batch_size 10 --ffn_hidden_size 100 --ffn_num_layers 2 --epochs 40 --alpha 1 --lamda 0 --split_type index_predetermined --crossval_index_file <index_file> --save_dir <chkpt_dir> --class_balance --mpn_shared
```

- To run `TAc-dmpna`, add these arguments to the above command

```bash
--attn_dim 100 --aggregation self-attention --model aada_attention
```

`source_data_path` and `target_data_path` specify the path to the source and target assay CSV files of the pair, respectively. First line contains a header `smiles,target`. Each of the following lines are comma-separated with the SMILES in the 1st column and the 0/1 label in the 2nd column.

`dataset_type` specifies the type of task; always classification for this project.

`extra_metrics` specifies the list of evaluation metrics.

`hidden_size` specifies the dimension of the learned compound representation out of `GNN`-based feature generators.

`depth` specifies the number of message passing steps.

`init_lr` specifies the initial learning rate.

`batch_size` specifies the batch size.

`ffn_hidden_size` and `fan_num_layers` specify the number of hidden units and layers, respectively, in the fully connected network used as the classifier.

`epochs` specifies the total number of epochs.

`split_type` specifies the type of data split.

`crossval_index_file` specifies the path to the index file which contains the indices of data points for train, validation and test split for each fold.

`save_dir` specifies the directory where the model, evaluation scores and predictions will be saved.

`class_balance` indicates whether to use class-balanced batches during training.

`model` specifies which model to use.

`aggregation` specifies which pooling mechanism to use to get the compound representation from the atom representations. Default set to `mean`: the atom-level representations from the message passing network are averaged over all atoms of a compound to yield the compound representation.

`attn_dim` specifies the dimension of the hidden layer in the 2-layer fully connected network used as the attention network.

Use `python code/train_aada.py -h` to check the meaning and default values of other parameters.

### 2. Running `TAc-fc` variants and ablations

- To run `Tac-fc`, 

```bash
python code/train_aada.py --source_data_path <source_assay_csv_file> --target_data_path <target_assay_csv_file> --dataset_type classification --extra_metrics prc-auc precision recall accuracy f1_score --hidden_size 25 --depth 4 --init_lr 1e-3 --batch_size 10 --ffn_hidden_size 100 --ffn_num_layers 2 --local_discriminator_hidden_size 100 --local_discriminator_num_layers 2 --global_discriminator_hidden_size 100 --global_discriminator_num_layers 2 --epochs 40 --alpha 1 --lamda 1 --split_type index_predetermined --crossval_index_file <index_file> --save_dir <chkpt_dir> --class_balance --mpn_shared
```

- To run `TAc-fc-dmpna`, add these arguments to the above command

```bash
--attn_dim 100 --aggregation self-attention --model aada_attention
```

##### Ablations

- To run `TAc-f`, add `--exclude_global` to the above command.
- To run `TAc-c`, add `--exclude_local` to the above command.
- Adding both `--exclude_local` and `--exclude_global` is equivalent to running `TAc`.

### 3. Running Baselines

#### `DANN`

```bash
python code/train_aada.py --source_data_path <source_assay_csv_file> --target_data_path <target_assay_csv_file> --dataset_type classification --extra_metrics prc-auc precision recall accuracy f1_score --hidden_size 25 --depth 4 --init_lr 1e-3 --batch_size 10 --ffn_hidden_size 100 --ffn_num_layers 2 --global_discriminator_hidden_size 100 --global_discriminator_num_layers 2 --epochs 40 --alpha 1 --lamda 1 --split_type index_predetermined --crossval_index_file <index_file> --save_dir <chkpt_dir> --class_balance --mpn_shared
```

- To run `DANN-dmpn`, add `--model dann` to the above command.
- To run `DANN-dmpna`, add `--model dann_attention --attn_dim 100 --aggregation self-attention --model` to the above command.

Run the following baselines from `chemprop` as follows:

#### `FCN-morgan`

```bash
python chemprop/train.py --data_path <assay_csv_file> --dataset_type classification --extra_metrics prc-auc precision recall accuracy f1_score --init_lr 1e-3 --batch_size 10 --ffn_hidden_size 100 --ffn_num_layers 2 --epochs 40 --features_generator morgan --features_only --split_type index_predetermined --crossval_index_file <index_file> --save_dir <chkpt_dir> --class_balance
```

#### `FCN-morganc`

```bash
python chemprop/train.py --data_path <assay_csv_file> --dataset_type classification --extra_metrics prc-auc precision recall accuracy f1_score --init_lr 1e-3 --batch_size 10 --ffn_hidden_size 100 --ffn_num_layers 2 --epochs 40 --features_generator morgan_count --features_only --split_type index_predetermined --crossval_index_file <index_file> --save_dir <chkpt_dir> --class_balance
```

#### `FCN-dmpn`

```bash
python chemprop/train.py --data_path <assay_csv_file> --dataset_type classification --extra_metrics prc-auc precision recall accuracy f1_score --hidden_size 25 --depth 4 --init_lr 1e-3 --batch_size 10 --ffn_hidden_size 100 --ffn_num_layers 2 --epochs 40 --split_type index_predetermined --crossval_index_file <index_file> --save_dir <chkpt_dir> --class_balance
```

#### `FCN-dmpna`

Add the following to the above command:

```bash
--model mpnn_attention --attn_dim 100 --aggregation self-attention
```

For the above baselines, `data_path` specifies the path to the target assay CSV file.

#### `FCN-dmpn(DT)`

```bash
python chemprop/train.py --data_path <source_assay_csv_file> --target_data_path <target_assay_csv_file> --dataset_type classification --extra_metrics prc-auc precision recall accuracy f1_score  --hidden_size 25 --depth 4 --init_lr 1e-3 --batch_size 10 --ffn_hidden_size 100 --ffn_num_layers 2 --epochs 40 --split_type index_predetermined --crossval_index_file <index_file> --save_dir <chkpt_dir> --class_balance
```

#### `FCN-dmpna(DT)`

```bash
--model mpnn_attention --attn_dim 100 --aggregation self-attention
```

For `FCN-dmpn(DT)`and `FCN-dmpna(DT)`, `data_path` and `target_data_path` specify the path to the source and target assay CSV files.

Use `python chemprop/train.py -h` to check the meaning of other parameters.

## Testing

1. To predict the labels of the compounds in the test set for `Tac*`, `DANN` methods:

   ```bash
   python code/predict.py --test_path <test_csv_file> --checkpoint_dir <chkpt_dir> --preds_path <pred_file>
   ```

   `test_path` specifies the path to a CSV file containing a list of SMILES and ground-truth labels. First line contains a header `smiles,target`. Each of the following lines are comma-separated with the SMILES in the 1st column and the 0/1 label in the 2nd column.

   `checkpoint_dir` specifies the path to the checkpoint directory where the model checkpoint(s) `.pt` filles are saved (i.e., `save_dir` during training).

   `preds_path` specifies the path to a CSV file where the predictions will be saved.

2. To predict the labels of the compounds in the test set for other methods:

   ```bash
   python chemprop/predict.py --test_path <test_csv_file> --checkpoint_dir <chkpt_dir> --preds_path <pred_file>

