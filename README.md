# SAE-RL

**Spatial Autoencoders (SAEs) for Reinforcement Learning (RL)**

This repository contains the code to reproduce the results of the paper _Tracking Object Positions in Reinforcement Learning: A Metric for Keypoint Detection_.

We present a computationally lightweight metric to evaluate latent representations of spatial autoencoder architectures.

Content:

1. [Installing dependencies](#installation)
2. [Executing scripts](#execution)
3. [More useful information](#more-information)

## Installation

You can launch a virtual environment with [pipenv] or run a [Docker] container to execute this code in. For both options, ensure your current working directory is this repository's root directory.

### Launch a [pipenv] environment

The following command creates a [pipenv] virtual environment and installs all necessary dependencies. Additionally, pass the `--dev` option to install packages only recommended for development.

```sh
pipenv install
```

Activate this virtual environment in your current shell with

```sh
pipenv shell
```

### Run a [Docker] container

A docker container with all necessary dependencies can, specifying the desired tag, easily be pulled from [this Docker Hub repository](https://hub.docker.com/repository/docker/jonasreiher/sae-rl) with

```sh
docker pull jonasreiher/sae-rl:<tag>
```

This is the recommended way to obtain a [Docker] image. Pay attention to available version specifier tags to pull your desired version.

Run an interactive container from this image, specifying the desired tag, with

```sh
docker run -it --gpus all jonasreiher/sae-rl:<tag>
```

Alternatively, the following command builds a [Docker] image locally with all necessary dependencies installed, building upon the [pipenv] environment.

```sh
docker build . -t jonasreiher/sae-rl:latest
```

You can push an updated version of this image to [Docker Hub], specifying a suitable version tag, with

```sh
docker push jonasreiher/sae-rl:<tag>
```

## Execution

With your virtual environment activated or within your container, you can run any of the scripts deposited in [`scripts/`](./scripts/). It is recommended to call them from the repository root where logs will then be saved.

> **Note**: All scripts use [Hydra] for configuration management. The base configuration files for everything can be found in [configs/](./configs/). Familiarize yourself with the possible configuration options there. To disable [Hydra]'s directory creation and logging, pass `hydra=hush`.

### Record a dataset

To record a dataset, make use of [`record.py`](./scripts/record.py) via

```sh
python scripts/record.py \
    environment=panda-push \
    recording.n_sequences=1000
```

> **Hint**: Run `python scripts/record.py --help` to see a list of all configuration parameters available as command line arguments.

This will store an image and keypoint dataset named `PandaPush-v3_labelled.hdf5` in [`data/`](./data/).

#### Paper replication

To replicate the dataset we used in the paper, run

```sh
python scripts/record.py \
    environment=panda-push-custom \
    environment.camera=panda-close \
    environment.camera.renderer=Tiny \
    save.tag=_fast
```

### Train a spatial autoencoder

To train a spatial autoencoder on the dataset generated above, make use of [`train_sae.py`](./scripts/train_sae.py) via

```sh
python scripts/train_sae.py \
    dataset=panda-push \
    model=keynet \
    training.epochs=500 \
    wandb=off
```

> **Hint**: Run `python scripts/train_sae.py --help` to see a list of all configuration parameters available as command line arguments.

This will store logging information, including the final SAE checkpoint, under the current timestamp in [logs/sae/panda_push/keynet+keynet/](./logs/sae/panda_push/keynet+keynet/). To view these logs with [Tensorboard], run the following command in this repository's root directory.

```sh
tensorboard --log-dir logs/sae/
```

> **Note**: To concurrently log to [Weights and Biases], adjust `wandb.entity` and `wandb.project` to your profile and set `wandb.enabled=True` (the default when omitting `wandb=off`).

#### Paper replication

All configurations for experiments conducted in the paper are collected in [configs/experiments/](./configs/experiments/). To train, e.g., the "Basic-vel-std-bg" SAE (also see [Naming](#naming)), you can just run

```sh
python scripts/train_sae.py \
    +experiment=sae-basic-vel-var-bg
```

### Train a reinforcement learning agent

To train a reinforcement learning agent with feature points from the spatial autoencoder trained above, make use of [`train_rl.py`](./scripts/train_rl.py) via

```sh
python scripts/train_rl.py \
    environment=panda-push \
    algorithm=sac \
    training.observation.immeasurable=False \
    training.observation.feature_points=True \
    training.sae_checkpoint=logs/sae/panda_push/keynet+keynet/YYYY-MM-DD--HH-MM-SS--0/checkpoint_final.pth \
    training.steps=1000000 \
    wandb=off
```

> **Hint**: Run `python scripts/train_rl.py --help` to see a list of all configuration parameters available as command line arguments.

This will store logging information, including the final agent model, under the current timestamp in [logs/rl/PandaPush-v3/](./logs/rl/PandaPush-v3/). To view these logs with [TensorBoard], run the following command in this repository's root directory.

```sh
tensorboard --log-dir logs/rl/
```

> **Note**: To concurrently log to [Weights and Biases], adjust `wandb.entity` and `wandb.project` to your profile and set `wandb.enabled=True` (the default when omitting `wandb=off`).

#### Paper replication

As before, all configurations for experiments conducted in the paper are collected in [configs/experiments/](./configs/experiments/). To train, e.g., the "full state" RL agent, you can just run

```sh
python scripts/train_rl.py \
    +experiment=rl-full
```

To train an RL agent with SAE-encoded keypoints, e.g. with "Basic-kp32" (also see [Naming](#naming)), additionally specify a checkpoint:

```sh
python scripts/train_rl.py \
    +experiment=rl-feat-only \
    training.sae_checkpoint=logs/sae/panda_push_custom/basic+basic/YYYY-MM-DD--HH-MM-SS--0/checkpoint_final.pth \
    training.sae_name=random-name-42 \
    training.sae_experiment=sae-basic-fp32
```

## More information

### Documentation

All classes and functions have docstrings, specifying what they do.

### Plotting

IPython notebooks with the source code for all plots can be found in [notebooks/](./notebooks/). Some of these plots require saved weights from logged runs. You can adjust the file paths accordingly to load your own checkpoints. Other plots require access to our private [Weights and Biases] project. You can insert your own `entity` and `project` there if you logged to [Weights and Biases] yourself.

### Naming

The "-std" modification mentioned in the paper is called "-var" in this codebase. "Ground-truth points" in the paper are "keypoints" in this codebase and "keypoints" in the paper are "feature points" here.

[Docker]: https://docs.docker.com
[Docker Hub]: https://hub.docker.com
[Hydra]: https://hydra.cc
[pipenv]: https://docs.pipenv.org
[TensorBoard]: https://www.tensorflow.org/tensorboard
[Weights and Biases]: https://wandb.ai/site
