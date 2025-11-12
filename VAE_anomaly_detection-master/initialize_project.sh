
# Install conda
wget https://repo.anaconda.com/archive/Anaconda3-2025.06-1-Linux-x86_64.sh
bash ./Anaconda3-2025.06-1-Linux-x86_64.sh

# Create virtual environment
conda create --name cow

# Import libraries
../anaconda3/bin/pip install torch
../anaconda3/bin/pip install torchvision
../anaconda3/bin/pip install torchmetrics
../anaconda3/bin/pip install tensorboard
../anaconda3/bin/pip install path
../anaconda3/bin/pip install pyyaml
../anaconda3/bin/pip install pytorch-lightning
../anaconda3/bin/pip install path



