# Fall project repo
This is the repo for my fall project regarding recommender systems in higher education.
The repository consists of a [sub repository](https://github.com/asash/bert4rec_repro). 
To run the experiments, follow the installations steps provided in their readme.


### Running TF
1. Login
2. **Activate** `rec-sys` venv `source ~/fall_project/rec-sys/bin/activate`
3. Load TF module with fossCuda (latest): `ml TensorFlow/2.6.0-foss-2021a-CUDA-11.3.1`
4. Verify GPUs with `nvidia-smi`
5. Verify TF working with GPUs: `python3 -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"`

## Transformers4Rec Guide

1. Clean, no modules loaded
2. Using custom conda (could probably have used the module) - MiniConda3 4.12 python 3.9.13
3. Create new env and install `conda create --name transformers4rec -c nvidia -c conda-forge transformers4rec`
    - Get import protobuf error - [Docs say manually install](https://github.com/NVIDIA-Merlin/Transformers4Rec/issues/424)
        - conda install protobuf
    - Import `torchmetrics`- error: `conda install -c conda-forge torchmetrics==0.3.2` -- Newer versions gives errors
4. Running the basic test code on the front page works now without errors
5. For nvtabular: Must install separately + CUDA
    - CUDA: `ml fosscuda/2020b`
    - nvtabular: `conda install -c nvidia -c rapidsai -c numba -c conda-forge nvtabular`

## In python with Tensorflow
### It is slightly [unstable](https://github.com/NVIDIA-Merlin/Transformers4Rec/pull/448)
1. Load TensorFlow-Foss cuda module from README
2. Installing jupyter-lab, pyarrow, dask with pip
3. Install transformers4rec with tensorflow and nvtabular: `pip install --user transformers4rec[tensorflow,nvtabular]`
4. **Torch** `python -m pip install --user torch>=1.0 --extra-index-url https://download.pytorch.org/whl/cu116`
5. **Torchmetrics** `pip install torchmetrics==0.3.2 --user`
    - Alternatively maybe just with `pip install --user transformers4rec[all]`
    
## Bert4Rec Repro -- With TF module
1. Can't have dashes in package names -> add as git submodule
    and renamed to `aprec`
2. Must have `pwd` on PYTHONPATH to resolve `aprec`-module
    * `export PYTHONPATH=`pwd`:$PYTHONPATH`
3. Install additional dependencies
    - `pip install --user "scikit-learn>=1.0" "lightgbm>=3.3.0" "seaborn>=0.11.2" "mmh3>=3.0.0" "recbole>=1.0.1" "wget>=3.2 "lightfm>=1.16"`
4. Had to fix `run_n_experiments` as I didn't install it in a conda env
    - Changed `unbuffer python ...` to `expect \`which unbuffer\` python...`

## Concept and field translation
- using HF model Helsinki-NLP/opus-mt-zh-en
- Needs `sentencepiece`-tokenizer -> `pip install --user transformers[sentencepiece]`


# Containers
## Docker is not used by default, but 