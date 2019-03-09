# Predictive Probabilistic Merging of Policies
Predictive Probabilistic Merging of Policies is a deep reinforcement learning framework where corrective feedback is used to improve sample efficiency. 
This code was made for the study ['Deep Reinforcement Learning with Feedback-based Exploration'](https://arxiv.org/).
In this repository, PPMP is demonstrated in combination with DDPG, as in the paper.

## Getting Started
The dependencies of this code are partially contained in the conda `ppmp_env.txt` environment file. In addition, you will need:
1. gym
2. tflearn
3. pandas
4. seaborn
5. tensorflow
6. box2d, box2d-kengz

To quickly setup remote intances, we've used the `server_setup.sh` file. It is not recommended to run this on your personal computer, but feel free to have a look at the installation procedures. 

## Acknowledgement
This algorithm was developed by Jan Scholten under the supervision of Jens Kober and Carlos Celemin. 
We especially thank Daan Wout for countless fruitful discussions. We acknowledge Patrick Emami for providing the DDPG baseline [code](https://github.com/pemami4911/deep-rl).

## Licence
You are free to use, share and adapt this work under the conditions stipulated in `LICENCE.md`. 

## Reference
