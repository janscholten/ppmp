# Predictive Probabilistic Merging of Policies
Predictive Probabilistic Merging of Policies is a deep reinforcement learning framework where corrective feedback is used to improve sample efficiency. 
This code was made for the study ['Deep Reinforcement Learning with Feedback-based Exploration'](https://arxiv.org/).
In this repository, PPMP is demonstrated in combination with DDPG, as in the paper.

## Getting Started
### Setting up the environment
We've tested this code using Ubuntu 18.10 (64 bit) and Python 3.6.
The most important dependencies of this code are gym, tflearn, seaborn and tensorflow. If you do not have these around yet, you may use conda+pip to quickly setup an appropriate environment by 
```
conda create --name ppmp tensorflow seaborn
source activate ppmp
pip install tflearn gym
```
and you should be ready. If not, the specific testing environment is reproduced with the `ppmp_env.txt` environment file.

To quickly setup remote intances, we've used the `server_setup.sh` file. It is not recommended to run this on your personal computer, but feel free to have a look at the installation procedures. 
### Running
The main file is `ppmp.py`. If you've activated your environment, you may invoke it from the root directory in terminal by
```
python ppmp.py
``` 
and it should output a csv (batch mode: for the header, use the `--header` argument).
By default, PPMP learns the `Pendulum-v0` problem using synthesised feedback. If you'd like to correct yourself, try
```
python ppmp.py --algorithm ppmp_human
```
It should start rendering the environments, but the problem is paused at startup. Press spacebar first, and then the arrow keys to provide feedback. 

Different environments are called with the `env` argument, e.g. `python ppmp.py --env MountainCarContinuous-v0`.
For other arguments (hyperparameters, environment settings, ...), see `python ppmp.py --help` or open the code. 

To record single runs, you may like to navigate to the testbench and call a script that saves the results for you:
```
cd single_analysis/pendulum
./run_pendulum --heads 7
```
A live plot is then available in `live_view.pdf`.


## Acknowledgement
This algorithm was developed by Jan Scholten under the supervision of Jens Kober and Carlos Celemin. 
We especially thank Daan Wout for countless fruitful discussions. We acknowledge Patrick Emami for providing the DDPG baseline [code](https://github.com/pemami4911/deep-rl).

## Licence
You are free to use, share and adapt this work under the conditions stipulated in `LICENCE.md`. 

## Reference
