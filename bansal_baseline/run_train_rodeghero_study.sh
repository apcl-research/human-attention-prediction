
participant_list=(KGT001 KGT002 KGT003 KGT004 KGT005 KGT007 KGT008 KGT009 KGT010)
rm results/correlation_rodeghero_study.csv

for p in "${participant_list[@]}"; do
    rm -r ./results/rodeghero_study/$p
    CUDA_DEVICE_ORDER='PCI_BUS_ID' CUDA_VISIBLE_DEVICES='1' OMP_NUM_THREADS=1 python3 train.py --holdout=$p --fixation_path=/nfs/projects/rodeghero_study.pkl --vocab_path /nfs/projects/rodeghero_study_vocab.pkl
    sleep 5
done

