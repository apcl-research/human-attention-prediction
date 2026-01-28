# Predicting Human Visual Attention on Words in Source Code

## Quick link
- [To-do list](#to-do-list)
- [Data Processing](#data-processing)
- [Finetuning and Inference](#finetuning-and-inference)
- [Metrics](#metrics)
- [Bansal Baseline](#bansal-baseline)
- [Tafasca Baseline](#tafasca-baseline)

## To-do list

- To set up your local environment, run the following command. We recommend the use of a virtual environment for running the experiments.
```
pip install -r requirements.txt
```

- Download the data in the [link](https://drive.google.com/drive/folders/1EwYCgbDlvyodcBF_WbqvJWRGLzqu-hMj?usp=drive_link) and place all data in ``/nfs/projects/``. Otherwise, you would need to change the directory in the data compiling script.
- Downlad the models pretrained models in the [link](https://drive.google.com/drive/folders/1EwYCgbDlvyodcBF_WbqvJWRGLzqu-hMj?usp=drive_link) and place C models in ``/nfs/projects/cam`` and Java models in ``/nfs/dropbox/jam350m_jm_1024``.
- For our model, refer to the [Finetuning and Inference](#finetuning-and-inference) section and the scripts are in [jam](./jam). 
- For baseline proposed by Bansal et al., refer to the [Bansal Baseline](#bansal-baseline) section and the scripts are in [bansal_baseline](./bansal_baseline)
- For baseline proposed by Tafasca et al., refer to the [Tafasca Baseline](#tafasca-baseline) and the scripts are in [Tafasca_baseline](./tafasca_baseline)

## Data Processing
For data processing, simply run the bash script for each dataset as follows:
```
./compile_data_rodeghero_study.sh
```
```
./compile_data_smith_study.sh
```
```
./compile_data_wallace_study.sh
```

The script would create the dataset for each participant holdout.

## Finetuning and Inference
To finetune the models and make the prediction, simply run the bash scripts for each model and dataset as follows:

- Model: ``ours``
```
./run_finetune_wallace_study_ours.sh
```
```
./run_finetune_smith_study_ours.sh
```
```
./run_finetune_rodeghero_study_ours.sh
```

- Model: ``ours-no-corr``
```
./run_finetune_wallace_study_ours_no_corr.sh
```
```
./run_finetune_smith_study_ours_nocorr.sh
```
```
./run_finetune_rodeghero_study_ours_nocorr.sh
```

- Note that the script will copy the pretrained model and create the relative directory. After running the script, it will generate a ``csv`` file that includes the Pearson corrlation for the related dataset and model. Please see the [Metrics](#metrics) section for how to compute all metrics.

- The script will also generate the ``.pkl`` file that include all of the results for computing metrics.

- Note that we used the model that has already been finetuned with Wllace study to finetune the model with the Rodeghero study because the Rodeghero study is much smaller comapred with other datasets.

## Metrics
To computing the metrics, run the following scripts for each study:

- Wallce study
```
python3 metrics_wallace_study.py
```

- Smith study
```
python3 metrics_smith_study.py
```

- Rodeghero study
```
python3 metrics_rodeghero_study.py
```

## Bansal Baseline
Follow the following steps to train the model proposed by Bansal et al. in our implementation:

Step 1: Run the following scripts to generate ScrML
```
python gen_srcml.py   --input_path /nfs/projects/wallace_study.pkl   --output_path ./wallace_study_srcml.pkl
```

Step 2: Build the vaocabularies for training 
```
python gen_vocab.py --fixation_data_path /nfs/projects/wallace_study.pkl --output_vocab_path ./wallace_study_vocab.pkl
```

Step 3: Train the models 
- Wallace study
```
./run_train_wallace_study.sh
```

- Smith study
```
./run_train_smith_study.sh
```

- Rodeghero study
```
./run_train_rodeghero_study.sh
```

- Note that the script will generate the model checkpoint, prediction results in a pickle file, a file that show Pearson correlation between human reference and model predicition.

- You will need to place those results in the right place or change the directory of the script in the [Metrics](#metrics) section to run the full evaluation.

## Tafasca Baseline
Run the following scripts to train the models proposed by Tafasca et al. in our implementation:

- Wallace study
```
./run_train_wallace_study.sh
```

- Smith study
```
./run_train_smith_study.sh
```

- Rodeghero study
```
./run_train_rodeghero_study.sh
```

- Note that the script will generate the model checkpoint, prediction results in a pickle file, a file that show Pearson correlation between human reference and model predicition.
- You will need to place those results in the right place or change the directory of the script in the [Metrics](#metrics) section to run the full evaluation.






