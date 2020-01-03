FROM pytorch/pytorch:1.1.0-cuda10.0-cudnn7.5-runtime
#FROM ubuntu:latest

COPY requirements.txt /dcp/requirements.txt
WORKDIR /dcp

RUN apt-get update \
    && apt-get install -y \
        python3-pip \
        vim \
        unzip \
        wget \
        git \
        libsm6 \
        libgl1-mesa-glx \
        libxext6 \
        libglfw3-dev \
        libxrender-dev \
        parallel \
        imagemagick
RUN pip3 install --upgrade pip
RUN python3.6 -m pip install -r requirements.txt
#RUN python3.6 -m pip install \
#    tensorboardX \
#    matplotlib \
#    scikit-image
#RUN pip3 install -U opencv-contrib-python==3.4.0.12
