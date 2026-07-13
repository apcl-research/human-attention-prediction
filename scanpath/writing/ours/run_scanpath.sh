#!/bin/bash


mkdir /scratch00/chiayi/scanpath/human_5tokens_ours
CSV_FILE=human_5tokens_ours.csv
header="fid,Normalized_LD,NDCG,Precision@k, Recall@k"
echo "$header" > "$CSV_FILE"

mkdir ./human_5tokens_ours

pred_dir="human_5tokens_ours"

for file in /nfs/projects/scanpath/scanpath_prediction/bansal_dataset/writing/byFunction_*/; do

    id=$(echo "$file" | grep -oP 'byFunction_\K[0-9]+')
	mkdir /scratch00/chiayi/scanpath/human_5tokens_ours/${id}
	cp  /scratch00/chiayi/ptgt/method_holdout/ckpt.pt /scratch00/chiayi/scanpath/human_5tokens_ours/${id}/

	CUDA_DEVICE_ORDER='PCI_BUS_ID' CUDA_VISIBLE_DEVICES='2,3' OMP_NUM_THREADS=2 time torchrun --rdzv-backend=c10d --rdzv-endpoint=localhost:4221 --nnodes=1 --nproc_per_node=2 train_scanpath.py config/finetune_scanpath.py --out_dir=/scratch00/chiayi/scanpath/human_5tokens_ours/${id} --data_path=$file/eyeseq_train --num_tokens=5

	CUDA_DEVICE_ORDER='PCI_BUS_ID' CUDA_VISIBLE_DEVICES='2' OMP_NUM_THREADS=2 time torchrun --rdzv-backend=c10d --rdzv-endpoint=localhost:4221 --nnodes=1 --nproc_per_node=1 sample_scanpath.py config/finetune_scanpath.py --testdir=data/scanpath_data_5tokens/${id}/test/ --pred_file=human_5tokens_ours/predict_${id}.txt --out_dir=/scratch00/chiayi/scanpath/human_5tokens_ours/${id}
	python3 metrics_func.py --holdout=$id --pred-dir=${pred_dir} --out=$CSV_FILE --N=5 --pids=/nfs/projects/scanpath/scanpath_prediction/bansal_dataset/writing/allfids.pkl --ref-file=/nfs/projects/scanpath/scanpath_prediction/bansal_dataset/writing

    sleep 5

done

rm -r /scratch00/chiayi/scanpath/human_5tokens_ours
rm -r ./human_5tokens_ours
