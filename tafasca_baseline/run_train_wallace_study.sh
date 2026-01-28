participant_list=("P3" "P4" "P5" "P6" "P7" "P8" "P9" "P10" "P11" "P12")
rm ./wallace_study/correlation_wallace_study.csv

for p in "${participant_list[@]}"; do
    rm -r ./wallace_study/$p
    mkdir -p ./wallace_study/$p
    CUDA_DEVICE_ORDER='PCI_BUS_ID' CUDA_VISIBLE_DEVICES='1' OMP_NUM_THREADS=1 python3 train_wallace_study.py --holdout=$p
    sleep 5
done

