participant_list=( "p2"  "p4"  "p6" "p7" "p8" "p9" "p10" "p11"  "p13" "p14" "p15" "p16" "p17" "p18"  "p20" "p21") # For C filter
rm ./smith_study/correlation_smith_study.csv

for p in "${participant_list[@]}"; do
    rm -r ./smith_study/$p
    mkdir -p ./smith_study/$p
    CUDA_DEVICE_ORDER='PCI_BUS_ID' CUDA_VISIBLE_DEVICES='1' OMP_NUM_THREADS=1 python3 train_smith_study.py --holdout=$p
    sleep 5
done
