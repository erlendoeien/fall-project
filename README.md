# Fall project repo
This is the repo for my fall project regarding recommender systems in higher education.


## Setup notes

### Running TF
1. Login
2. **Activate** `rec-sys` venv `source ~/fall_project/rec-sys/bin/activate`
3. Load TF module with fossCuda (latest): `ml TensorFlow/2.6.0-foss-2021a-CUDA-11.3.1`
4. Verify GPUs with `nvidia-smi`
5. Verify TF working with GPUs: `python3 -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"`
