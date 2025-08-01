# AP-DPM: A Dual-Path Merging Network via Adversarial Anatomical Prior Guidance for Wrist Bone Segmentation

This is the official implementation of AP-DPM: A Dual-Path Merging Network via Adversarial Anatomical Prior Guidance for Wrist Bone Segmentation.


## Setup
- Install the conda environment
```
conda create -n ramw600 python=3.10
conda activate ramw600
```
- Install Pytroch
```
# CUDA 12.6
pip3 install torch torchvision torchaudio
```
- Install other requirements, such as albumentations.
```
pip install -r requirements.txt
```
## Dataset
Please download the dataset from <https://huggingface.co/datasets/TokyoTechMagicYang/RAM-W600>.

## Run
- Training. The training configurations are in ./main_seg_dual_path.py and ./main_seg_dual_path_gan.py. We also provide scripts in ./train.sh. Before running, you should refer to ./main_seg_dual_path.py and ./main_seg_dual_path_gan.py and add your paths to the bash files.
After running, the checkpoints will be saved in ./ckpts/.
- The training of AP-DPM contains 2 stages. You need to use ./main_seg_dual_path.py for the 1st stage and ./main_seg_dual_path_gan.py for the 2nd stage.

```
bash train.sh
```

- Testing. The testing configurations are in ./main_seg_dual_path.py and ./main_seg_dual_path_gan.py. We also provide scripts in ./test.sh. 
After running, the results of the visualization will be saved in the folder you chose for testing.
```
bash test.sh
```