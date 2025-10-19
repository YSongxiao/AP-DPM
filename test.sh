# Test
python main_seg_dual_path_gan.py --mode test --model DPMSwinUMamba --data_path path/to/your/data --checkpoint path/to/ckpt --save_csv --save_overlay
# Inference
python main_seg_dual_path_gan.py --mode infer --model DPMSwinUMamba --data_path path/to/your/data --checkpoint path/to/ckpt --save_npy
