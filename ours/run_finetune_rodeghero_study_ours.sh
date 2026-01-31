participant_list=(KGT001 KGT002 KGT003 KGT004 KGT005 KGT007 KGT008 KGT009 KGT010) # For Java
rm correlation_ab_paticipant_java_2013_scratch.csv
for p in "${participant_list[@]}"; do
	rm -r ./rodeghero_study_ours/$p
	mkdir ./rodeghero_study_ours/$p
	cp /nfs/projects/ptgt_predictions/java/gpt2_ab_participant/P6/ckpt_P6.pt ./rodeghero_study_ours/$p/ckpt_$p.pt
	sleep 2

	CUDA_DEVICE_ORDER='PCI_BUS_ID' CUDA_VISIBLE_DEVICES='2,3' OMP_NUM_THREADS=2 time torchrun --rdzv-backend=c10d --rdzv-endpoint=localhost:4222 --nnodes=1 --nproc_per_node=2 finetune_rodeghero_study_ours.py config/finetune_rodeghero_study_ours.py --outfilename=ckpt_$p.pt --holdout=$p --out_dir=./rodeghero_study_ours/$p

	CUDA_DEVICE_ORDER='PCI_BUS_ID' CUDA_VISIBLE_DEVICES='2' OMP_NUM_THREADS=2 time torchrun --rdzv-backend=c10d --rdzv-endpoint=localhost:4222 --nnodes=1 --nproc_per_node=1 sample_rodeghero_study_ours.py config/finetune_rodeghero_study_ours.py --testdir=data/ptgt_participant_java_2013dataset_b_data/participant$p/testset/  --out_dir=./rodeghero_study_ours/$p --outfilename=ckpt_$p.pt

	

    sleep 5
done
