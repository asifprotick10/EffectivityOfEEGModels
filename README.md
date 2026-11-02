# EffectivityOfEEGModels
Standard vanilla RNNs and hybrid CNN-biLSTMs are compared against foundation models like LaBraM and REVE on the TUSZ dataset.



For finetuning Labram, run the following command:
python run_class_finetuning.py   --output_dir ./checkpoints/finetune_tusz_bin_v1/ --log_dir ./log/finetune_tusz_bin_v1   --model labram_base_patch200_200 --finetune ./checkpoints/labram-base.pth   --dataset TUSZ --tusz_path 'path/to/TUSZ/processed/data' --weight_decay 0.05 --batch_size 64 --lr 5e-4 --update_freq 1 --warmup_epochs 5 --epochs 50   --layer_decay 0.65 --drop_path 0.1 --save_ckpt_freq 5   --disable_rel_pos_bias --abs_pos_emb --disable_qkv_bias   --clip_grad 1.0 --seed 0
