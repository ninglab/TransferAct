#!/bin/bash
# File              : run_split.sh
# Author            : Vishal Dey <dey.78@osu.edu>
# Date              : Sat 16 Oct 2021 19:34:14
# Last Modified Date: Sat 16 Oct 2021 19:39:17
# Last Modified By  : Vishal Dey <dey.78@osu.edu>

# Run split on the assay pairs

pairs_list=$1
data_dir=$2

pfams=()
targetA=()
targetB=()

while IFS=$'\t'
	read col1 col2 col3;
do
	pfams+=("$col1")
	targetA+=("$col2")
	targetB+=("$col3")
done < <(cat ${pairs_list})


for i in "${!targetA[@]}"; do

	#echo "${pfams[$i]}, ${targetA[$i]}, ${targetB[$i]}"
	python chemprop/scripts/create_crossval_splits.py --data_path "${data_dir}/${pfams[$i]}/${targetA[$i]}-${targetB[$i]}/${targetA[$i]}-0.0.csv" --split_type "stratified" --num_folds 10 --save_dir "${data_dir}/${pfams[$i]}/${targetA[$i]}-${targetB[$i]}/crossval_splits/${targetA[$i]}-0.0/"  --test_folds_to_test 10 --val_folds_per_test 1
	python chemprop/scripts/create_crossval_splits.py --data_path "${data_dir}/${pfams[$i]}/${targetA[$i]}-${targetB[$i]}/${targetB[$i]}-0.0.csv" --split_type "stratified" --num_folds 10 --save_dir "${data_dir}/${pfams[$i]}/${targetA[$i]}-${targetB[$i]}/crossval_splits/${targetB[$i]}-0.0/"  --test_folds_to_test 10 --val_folds_per_test 1
done
