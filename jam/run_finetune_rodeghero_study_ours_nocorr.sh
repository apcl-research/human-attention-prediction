#!/bin/bash
participant_list=(KGT001 KGT002 KGT003 KGT004 KGT005 KGT007 KGT008 KGT009 KGT010) # For Java

rm correlation_b_paticipant_java_2013.csv
for p in "${participant_list[@]}"; do
	rm -r ./rodeghero_study_ours/$p
	mkdir ./rodeghero_study_ours/$p
	cp /nfs/dropbox/jam350m_jm_1024/ckpt.pt ./rodeghero_study_ours/$p/ckpt_$p.pt
	sleep 2

	CUDA_DEVICE_ORDER='PCI_BUS_ID' CUDA_VISIBLE_DEVICES='0,1' OMP_NUM_THREADS=2 time torchrun --rdzv-backend=c10d --rdzv-endpoint=localhost:4000 --nnodes=1 --nproc_per_node=2 finetune_rodeghero_study_ours.py config/finetune_rodeghero_study_ours_no_corr.py --outfilename=ckpt_$p.pt --holdout=$p --out_dir=./rodeghero_study_ours/$p

	CUDA_DEVICE_ORDER='PCI_BUS_ID' CUDA_VISIBLE_DEVICES='1' OMP_NUM_THREADS=2 time torchrun --rdzv-backend=c10d --rdzv-endpoint=localhost:4000 --nnodes=1 --nproc_per_node=1 sample_rodeghero_study_no_corr.py config/finetune_rodeghero_study_ours_no_corr.py --testdir=data/ptgt_participant_java_2013dataset_b_data/participant$p/testset/  --out_dir=./rodeghero_study_ours/$p --outfilename=ckpt_$p.pt

	

    sleep 5
done

