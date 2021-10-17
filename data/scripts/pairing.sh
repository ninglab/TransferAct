#!/bin/bash

# call as $ bash scripts/pairing.sh choosen_targets_for_pairing_active100.txt processed_pairs_active100/preprocess_assay_pairs_active100_stat.txt 
target_list=$1
output=$2
dir="$(dirname $output)"

targets=()
pfams=()

while IFS=$'\t'
	read col1 col2 col3 col4;
do
	targets+=("$col1")
	pfams+=("$col4")
done < <(cat ${target_list})

printf "annotated_pfam,target_A,target_B,#active_A,#inactive_A,#active_B,#inactive_B,#common_active,#common_inactive,#common_smiles_same_labels,#common_smiles_disp_labels," > ${output}
printf "#common_smiles,%%common_active,%%common_inactive,%%common_smiles_same_labels,%%common_smiles_disp_labels,%%common_smiles\n" >> ${output}

for i in "${!targets[@]}"; do

	#printf "%s\t%s\t%s\n" "$i" "${targets[$i]}" "${pfams[$i]}"
	for j in "${!targets[@]}"; do

		if [[ $i < $j && ${pfams[$i]} == ${pfams[$j]} ]]; then
			printf "${pfams[$i]},${targets[$i]},${targets[$j]}," >> ${output}

			python scripts/process_assay_pairs.py --data_path_1 processed_assays/${targets[$i]}.csv --data_path_2 processed_assays/${targets[$j]}.csv --save_dir "${dir}/pairs/${pfams[$i]}/${targets[$i]}-${targets[$j]}/" --smiles_path pubchem/standardized_cid_smiles.txt --balance >> ${output}
		fi
	done
done

