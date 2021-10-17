#!/bin/bash
# File              : run_pairwise_similarity.sh
# Author            : Vishal Dey <dey.78@osu.edu>
# Date              : Sat 16 Oct 2021 19:05:44
# Last Modified Date: Sat 16 Oct 2021 19:05:44
# Last Modified By  : Vishal Dey <dey.78@osu.edu>

conda activate chemprop

target_list=$1
output=$2

dir="$(dirname ${output})"

targets=()
pfams=()

while IFS=$'\t'
	read col1 col2 col3 col4;
do
	targets+=("$col1")
	pfams+=("$col4")
done < <(cat ${target_list})

printf "data_1,data_2,all_1-all_1,all_2-all_2,all_1-all_2,active_1-active_1,inactive_1-inactive_1,active_1-inactive_1,active_2-active_2,inactive_2-inactive_2,active_2-inactive_2,active_1-active_2,inactive_1-inactive_2,active_1-inactive_2,inactive_1-active_2\n" > ${output}

for i in "${!targetA[@]}"; do

	for j in "${!targets[@]}"; do

		if [[ $i < $j && ${pfams[$i]} == ${pfams[$j]} ]]; then

			python scripts/pairwise_similarity.py --data_path_1 "${dir}/pairs/${pfams[$i]}/${targets[$i]}-${targets[$j]}/${targets[$i]}-0.0.csv" --data_path_2 "${dir}/pairs/${pfams[$i]}/${targets[$i]}-${targets[$j]}/${targets[$j]}-0.0.csv" >> ${output}
		fi
	done
done
