#!/bin/bash
# These configuration steps can be used as a guideline for setting up remote instances. 
# Beware that these commands are quite hefty on a system-level: not recommended for your personal computer

# First, run these by hand

# cd
# export LC_ALL=C
# sudo apt-get update
# sudo apt-get -y --force-yes upgrade
# echo LogLevel=quiet >> .ssh/config
# git clone git@github.com:janscholten/ppmp
# cd ppmp

# Then run this file, twice if necessary:

set -x

apt-get install -yqq python-pip htop
apt-get install -yqq python-numpy python-dev cmake zlib1g-dev libjpeg-dev xvfb libav-tools xorg-dev python-opengl libboost-all-dev libsdl2-dev swig
yes | pip install --upgrade pip
wget https://repo.continuum.io/archive/Anaconda3-5.0.0.1-Linux-x86_64.sh
bash Anaconda3-5.0.0.1-Linux-x86_64.sh -b -p $HOME/miniconda
echo export PATH="$HOME/miniconda/bin:$PATH" >> ~/.bashrc
echo source activate ddpg >> ~/.bashrc
export PATH="$HOME/miniconda/bin:$PATH"
yes | conda create --name ddpg --file ddpg_conda_env.txt
source activate ddpg
yes | pip -qq install tensorflow tflearn seaborn gym box2d box2d-kengz
yes | pip -qq install tensorflow tflearn seaborn gym box2d box2d-kengz

git config --global push.default matching

source ~/.bashrc