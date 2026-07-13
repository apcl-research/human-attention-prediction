#!/bin/bash


mkdir /scratch00/chiayi/scanpath/llm_5tokens_ase
CSV_FILE=llm_5tokens_ase.csv
header="fid,Normalized_LD,NDCG,Precision@k, Recall@k"
echo "$header" > "$CSV_FILE"

mkdir ./llm_5tokens_ase

pred_dir="llm_5tokens_ase"

for file in /nfs/projects/scanpath/scanpath_prediction/bansal_dataset/writing/byFunction_*/; do

    id=$(echo "$file" | grep -oP 'byFunction_\K[0-9]+')
	mkdir /scratch00/chiayi/scanpath/llm_5tokens_ase/${id}
	cp /nfs/dropbox/jam350m_jm_1024/ckpt.pt /scratch00/chiayi/scanpath/llm_5tokens_ase/${id}/

	CUDA_DEVICE_ORDER='PCI_BUS_ID' CUDA_VISIBLE_DEVICES='0,1' OMP_NUM_THREADS=2 time torchrun --rdzv-backend=c10d --rdzv-endpoint=localhost:4000 --nnodes=1 --nproc_per_node=2 train_scanpath.py config/finetune_scanpath.py --out_dir=/scratch00/chiayi/scanpath/llm_5tokens_ase/${id} --dataset=scanpath_data_5tokens/$id
	CUDA_DEVICE_ORDER='PCI_BUS_ID' CUDA_VISIBLE_DEVICES='1' OMP_NUM_THREADS=2 time torchrun --rdzv-backend=c10d --rdzv-endpoint=localhost:4000 --nnodes=1 --nproc_per_node=1 sample_scanpath.py config/finetune_scanpath.py --testdir=data/scanpath_data_5tokens/${id}/test/ --pred_file=${pred_dir}/predict_${id}.txt --out_dir=/scratch00/chiayi/scanpath/llm_5tokens_ase/${id}
	python3 metrics_func.py --holdout=$id --pred-dir=${pred_dir} --out=$CSV_FILE --N=5 --pids=/nfs/projects/scanpath/scanpath_prediction/bansal_dataset/writing/allfids.pkl --ref-file=/nfs/projects/scanpath/scanpath_prediction/bansal_dataset/writing

    sleep 5

done

rm -r /scratch00/chiayi/scanpath/llm_5tokens_ase
rm -r ./llm_5tokens_ase
