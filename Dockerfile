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
    swig

RUN yes | pip3 --no-cache install tflearn seaborn gym box2d box2d-kengz imageio h5py \
    https://github.com/sigilioso/tensorflow-build/raw/master/tensorflow-1.4.0-cp36-cp36m-linux_x86_64.whl

# make it feel like home
RUN ln -sf /usr/bin/python3 /usr/bin/python && \
    ln -sf /usr/bin/pip3 /usr/bin/pip

# This 1000 should match local $UID
RUN useradd -u 1000 -ms /bin/bash ppmp
USER 1000 
