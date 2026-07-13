#!/bin/bash
rm scanpath_results_claude_6tokens_writing.csv
csv_file="scanpath_results_claude_6tokens_writing.csv"
header="fid,Normalized_LD,NDCG,Precision@k, Recall@k"
count=1
echo "$header" > "$csv_file"


fid_list=( 10222579 14467418 17121898 18418213 19498280 19682824 26285656 29601536 33719869 36405409 39120328 39299426 4114383 45047585 4627680 49250848 50994916 12723449 14477536 1782360 18421665 19505695 20787007 28631042 31696447 34105249 36634895 39233258 39840471 41287529 45130358 47570692 49866815 50995324 12725774 16777940 1810081 19218425 19507414 23014476 29318894 31789275 34427273 38221424 39233866 40865212 43137003 45147874 47571713 50026101 51577053 13482891 1694531 18354735 19413001 19507735 26215341 29572299 33718481 34604973 38737384 39298970 40936756 44521080 46026508 47571922 50891793 7957602
)

python3 data_processing_incontext_6tokens_writing.py
python3 retrieve_writing.py --batch_id="yourbatchid" --output_dir="scanpath_results_claude_6tokens_writing"

for fid in "${fid_list[@]}"; do        
	echo $fid
        id="${dir%/}"      
        id="${id##*_}"     
        python3 metrics_func.py --holdout=$fid --pred-dir="scanpath_results_claude_6tokens_writing" --out=$csv_file --ref-file=/nublar/eyeseq/writing/n10/ --pid=/nublar/eyeseq/writing/n10/allfids.pkl


    sleep 5
done
