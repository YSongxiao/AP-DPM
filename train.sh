# Stage 1:
python main_seg_dual_path.py --mode train --image_size 512 --train_batch_size 4 --val_batch_size 1 --model DPMSwinUMamba --data_path path/to/your/data
# Stage 2:
python main_seg_dual_path_gan.py --mode train --image_size 512 --train_batch_size 4 --val_batch_size 1 --model DPMSwinUMamba --data_path path/to/your/data --stage1_checkpoint path/to/ckpt
