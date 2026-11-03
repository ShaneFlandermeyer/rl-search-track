# rl-search-track

## Usage

To run experiments, you must install this repository and its dependencies as follows:
```[bash]
pip install -e .
pip install jax[cuda12]
```

This installs the following packages:
- [MOTPY](https://github.com/ShaneFlandermeyer/MOTpy/tree/develop): Multi-object tracking in python
- [flax-gnn](https://github.com/ShaneFlandermeyer/flax-gnn/tree/develop): Graph neural networks in Jax/Flax
- [tdmpc2-jax](https://github.com/ShaneFlandermeyer/tdmpc2-jax/tree/develop): Model-based RL engine

Next, activate the venv that contains the packages above. 

To see an example of the behavior of each sensor environment, run the following from the project root:
```[bash]
python3 examples/random_actions --env=BeamOptimization-v0 # or --env=MobileSurveillance-v0
```

To train the agent, run the following from the project root:

```[bash]
python3 train.py
```

## Citation

```
@article{flandermeyer2026,
  author={Flandermeyer, Shane A. and Metcalf, Justin G.},
  journal={IEEE Transactions on Aerospace and Electronic Systems}, 
  title={Relational Reinforcement Learning for Autonomous Search-and-Track}, 
  year={2026},
  volume={62},
  number={},
  pages={18577-18596},
  keywords={Tracking;Modeling;Planning;Image sensors;Optimization;Measurement;Timing;Labeling;Equations;Filtering;Deep learning;model predictive control (MPC);multiobject tracking (MOT);reinforcement learning (RL);relational reasoning;sensor control},
  doi={10.1109/TAES.2026.3737757}}
```

## License
This project is licensed under the MIT License (see the LICENSE file for details). This repository relies on third-party code which is subject to their respective licenses.
