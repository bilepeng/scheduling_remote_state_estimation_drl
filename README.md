# Communication Scheduling by Deep Reinforcement Learning for Remote Traffic State Estimation with Bayesian Inference

This repository provides the source code of transmit power control for a remote
traffic state estimation system.

## File List
The following files are provided in this repository:
- env.py: defining the environment
- normalize.py: a script to normalize the state
- test.py: testing and visualizing the model
- train.py: training the model
- vehicle_movement.py: functions for carrying out state updates and estimations.


## Usage

You can run train.py with arguments 
- '--algorithm' being either 'SAC' or 'PPO'
- '--scenario' being either 'intersection' or 'lane_changing'

and then run test.py with the same arguments to test the trained models in
the directory 'checkpoints'.

## Acknowledgements
The of Bile Peng and Eduard Jorswieck was in part supported by the Federal Ministry of Education and Research (BMBF, Germany) as part of the 6G Research and Innovation Cluster 6G-RIC under Grant 16KISK020K.
The work of Gonzalo Seco-Granados is partly funded by the Spanish Ministry of Science and Innovation PID2020-118984GB-I00 and by the Catalan ICREA Academia Programme.

## If You Use This Code
If you use or extend this code, please cite our work.

B. Peng, Y. Xie, G. Seco-Granados, H. Wymeersch and E. Jorswieck, "Communication Scheduling by Deep Reinforcement Learning for Remote Traffic State Estimation with Bayesian Inference," in IEEE Transactions on Vehicular Technology (accepted).