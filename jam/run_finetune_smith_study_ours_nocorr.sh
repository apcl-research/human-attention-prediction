#!/bin/bash
participant_list=( "p2"  "p4"  "p6" "p7" "p8" "p9" "p10" "p11"  "p13" "p14" "p15" "p16" "p17" "p18"  "p20" "p21") # For C filter
rm correlation_b_paticipant_c.csv

for p in "${participant_list[@]}"; do
	rm -r ./smith_study_ours/$p
	mkdir ./smith_study_ours/$p
	cp /nfs/projects/cam/ckpt.pt ./smith_study_ours/$p/ckpt_$p.pt
	sleep 2

	CUDA_DEVICE_ORDER='PCI_BUS_ID' CUDA_VISIBLE_DEVICES='0,1' OMP_NUM_THREADS=2 time torchrun --rdzv-backend=c10d --rdzv-endpoint=localhost:4111 --nnodes=1 --nproc_per_node=2 finetune_smith_study_ours_no_corr.py config/finetune_smith_study_ours_no_corr.py --outfilename=ckpt_$p.pt --holdout=$p --out_dir=./smith_study_ours/$p
	

	CUDA_DEVICE_ORDER='PCI_BUS_ID' CUDA_VISIBLE_DEVICES='1' OMP_NUM_THREADS=2 time torchrun --rdzv-backend=c10d --rdzv-endpoint=localhost:4111 --nnodes=1 --nproc_per_node=1 sample_smith_study_no_corr.py config/finetune_smith_study_ours_no_corr.py --testdir=data/ptgt_participant_data_b_c/participant$p/testset/  --out_dir=./smith_study_ours/$p --outfilename=ckpt_$p.pt


    sleep 5

done
