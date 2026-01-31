participant_list=("P3" "P4" "P5" "P6" "P7" "P8" "P9" "P10" "P11" "P12")
rm correlation_wallace_study_ours.csv
for p in "${participant_list[@]}"; do
	rm -r ./wallace_study_ours/$p
	mkdir ./wallace_study_ours/$p
	cp /nfs/dropbox/jam350m_jm_1024/ckpt.pt ./wallace_study_ours/$p/ckpt_$p.pt
	sleep 2

	CUDA_DEVICE_ORDER='PCI_BUS_ID' CUDA_VISIBLE_DEVICES='0,1' OMP_NUM_THREADS=2 time torchrun --rdzv-backend=c10d --rdzv-endpoint=localhost:4111 --nnodes=1 --nproc_per_node=2 finetune_wallace_study_ours.py config/finetune_wallace_study_ours.py --outfilename=ckpt_$p.pt --holdout=$p --out_dir=./wallace_study_ours/$p

	CUDA_DEVICE_ORDER='PCI_BUS_ID' CUDA_VISIBLE_DEVICES='1' OMP_NUM_THREADS=2 time torchrun --rdzv-backend=c10d --rdzv-endpoint=localhost:4111 --nnodes=1 --nproc_per_node=1 sample_wallace_study_ours.py config/finetune_wallace_study_ours.py --testdir=data/ptgt_participant_data_b/participant$p/testset/  --out_dir=./wallace_study_ours/$p --outfilename=ckpt_$p.pt


    sleep 5

done
