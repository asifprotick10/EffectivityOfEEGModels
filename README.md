# EffectivityOfEEGModels
Standard vanilla RNNs and hybrid CNN-biLSTMs are compared against foundation models like LaBraM and REVE on the TUSZ dataset.

Steps:

Step 1: 
Download the TUSZ dataset from the official website https://isip.piconepress.com/projects/nedc/html/tuh_eeg

Step 2: 
Preprocess the dataset using the two scripts, one for LaBraM and the other for REVE. The preprocessed data for REVE has been used for evaluating RNN and hybrid CNN-biLSTM model. 

Step 3:
Modify the config files as needed (such as config_dt.yaml, tusz.yaml, etc for REVE)

Step 4:
Fine tune the dataset and get the results

For finetuning Labram, we used the following command:
python run_class_finetuning.py   --output_dir ./checkpoints/finetune_tusz_bin_v1/ --log_dir ./log/finetune_tusz_bin_v1   --model labram_base_patch200_200 --finetune ./checkpoints/labram-base.pth   --dataset TUSZ --tusz_path 'path/to/TUSZ/processed/data' --weight_decay 0.05 --batch_size 64 --lr 5e-4 --update_freq 1 --warmup_epochs 5 --epochs 50   --layer_decay 0.65 --drop_path 0.1 --save_ckpt_freq 5   --disable_rel_pos_bias --abs_pos_emb --disable_qkv_bias   --clip_grad 1.0 --seed 0

For finetuning REVE, we used the following command:
python dt.py
