FROM ubuntu:latest

RUN apt-get update \
    && apt-get upgrade \
    && apt-get purge -y python2.7-minimal

RUN apt-get -qq -y install \
    python3-pip \
    python3-numpy \
    python3-dev \
    cmake \
    zlib1g-dev \
    libjpeg-dev \
    xvfb \
    xorg-dev \
    python3-opengl \
    libboost-all-dev \
    libsdl2-dev \
    swig \
    git

RUN yes | pip3 --no-cache install tensorflow==1.8 tflearn seaborn gym box2d box2d-kengz imageio \
    || yes | pip3 --no-cache install tensorflow==1.8 tflearn seaborn gym box2d box2d-kengz imageio

# make it feel like home
RUN ln -sf /usr/bin/python3 /usr/bin/python && \
    ln -sf /usr/bin/pip3 /usr/bin/pip