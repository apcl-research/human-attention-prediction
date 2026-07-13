
participant_list=( "p2"  "p4"  "p6" "p7" "p8" "p9" "p10" "p11"  "p13" "p14" "p15" "p16" "p17" "p18"  "p20" "p21") # For C filter

rm results/correlation_smith_study.csv

for p in "${participant_list[@]}"; do
    rm -r ./results/smith_study/$p
    CUDA_DEVICE_ORDER='PCI_BUS_ID' CUDA_VISIBLE_DEVICES='1' OMP_NUM_THREADS=1 python3 train_smith_study.py --holdout=$p --fixation_path=/nfs/projects/smith_study.pkl --vocab_path /nfs/projects/smith_study_vocab.pkl
    sleep 5
done
