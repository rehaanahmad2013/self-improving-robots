# Autonomous RL Baselines
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](https://github.com/rlworkgroup/metaworld/blob/master/LICENSE)

## Instructions

Install [MuJoCo](http://www.mujoco.org/) if it is not already the case:

* Obtain a license on the [MuJoCo website](https://www.roboti.us/license.html).
* Download MuJoCo binaries [here](https://www.roboti.us/index.html).
* Unzip the downloaded archive into `~/.mujoco/mujoco200` and place your license key file `mjkey.txt` at `~/.mujoco`.
* Use the env variables `MUJOCO_PY_MJKEY_PATH` and `MUJOCO_PY_MUJOCO_PATH` to specify the MuJoCo license key path and the MuJoCo directory path.
* Append the MuJoCo subdirectory bin path into the env variable `LD_LIBRARY_PATH`.

Install the following libraries:
```sh
sudo apt update
sudo apt install libosmesa6-dev libgl1-mesa-glx libglfw3
```

Install dependencies:
```sh
conda env create -f conda_env.yml
conda activate arl116
```

Install EARL benchmark (optional, but necessary if you want to use the codebase directly):
```sh
git clone https://github.com/architsharma97/earl_benchmark.git
cd earl_benchmark/
pip install -e .
```
Train an autonomous RL agent using MEDAL:
```sh
python3 medalplusplus.py
```

Monitor results:
```sh
tensorboard --logdir exp_local
```

## Acknowledgements

The codebase is built on top of the PyTorch implementation of [DrQ-v2](https://arxiv.org/abs/2107.09645), original codebase linked [here](https://github.com/facebookresearch/drqv2). We thank the authors for an easy codebase to work with!
