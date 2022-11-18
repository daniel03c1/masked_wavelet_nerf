Tested on Ubuntu 18.04 + Pytorch 1.10.2



Install environment:

conda create -n MaskDWT python=3.8
conda activate MaskDWT
pip install torch torchvision
pip install tqdm scikit-image opencv-python configargparse lpips imageio-ffmpeg kornia lpips tensorboard



Install pytorch wavelets:

git clone https://github.com/fbcotter/pytorch_wavelets
cd pytorch_wavelets
pip install .



How to run:

Train a model
python3 train.py --config=configs/chair.txt --use_mask --mask_weight=1e-10 --grid_bit=8 --use_dwt --dwt_level=4


Compress the model
python3 compress.py --compress=1 --compress_levelwise=1 --ckpt=PATH_TO_CHECKPOINT


Decompress the compressed model and evaluate the model
python3 compress.py --decompress=1 --decompress_levelwise=1 --config=configs/chair.txt --ckpt=PATH_TO_CHECKPOINT