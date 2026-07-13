participant_list=("KGT001" "KGT002" "KGT003" "KGT004" "KGT005" "KGT007" "KGT008" "KGT009" "KGT010") # For Java
rm ./rodeghero_study/correlation_rodeghero_study.csv

for p in "${participant_list[@]}"; do
    rm -r ./rodeghero_study/$p
    mkdir -p ./rodeghero_study/$p
    CUDA_DEVICE_ORDER='PCI_BUS_ID' CUDA_VISIBLE_DEVICES='2' OMP_NUM_THREADS=1 python3 train_rodeghero_study.py --holdout=$p
    sleep 5
done

