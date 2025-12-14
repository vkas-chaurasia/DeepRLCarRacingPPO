# DeepRL Car Racing PPO

This repository contains a Proximal Policy Optimization (PPO) implementation for the `CarRacing-v2` environment in Gymnasium.

## Demo

![Car Racing Agent](assets/eval_moving_avg_episode-episode-7.gif)

## Structure

- `param_1000/`, `param_2000/`: Checkpoints.
- `logs/`: Training logs.
- `assets/`: Images and GIFs.
- `network.py`: Neural network architecture.
- `agent.py`: PPO Agent implementation.
- `environment.py`: Environment wrapper.
- `train.py`: Training script.
- `test.py`: Testing and validation script.

## Installation

1. Clone the repository.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
   *Note: You may need `swig` installed on your system for Box2D.*

## Usage

### Training

To train the agent:
```bash
python3 train.py --epochs 1000
```
This will save checkpoints to `checkpoints/` and logs to `logs/`.

### Testing

To test a trained model:
```bash
python3 test.py --param_path checkpoints/param_1000/param_1000_ppo_net_params.pkl --runs 5
```

### Notebook Demo

You can also use the `demo.ipynb` notebook to run training and testing interactively.
