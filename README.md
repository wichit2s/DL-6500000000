# DL-6500000000

## 1. create **kerasenv** and activate

    python -m venv venv/kerasenv
    # activate kerasenv 
    venv/kerasenv/scripts/activate
    pip install -r projects/keras.txt

## 2. create **torchenv** and activate 

### 2.1 using WSL

    python -m venv venv/torchenv
    source venv/torchenv/bin/activate
    pip install -r projects/torch-wsl-venv.txt

### 2.2 using conda

    conda env create -f torch-environment.yml
    conda activate torchenv


## start jupyter notebook

    jupyter notebook