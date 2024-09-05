FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime
#FROM pytorch/pytorch:2.0.0-cuda11.7-cudnn8-runtime
#FROM continuumio/anaconda3

ARG BUILDDATE
ARG MAINTAINER
ARG DESCRIPTION
LABEL maintainer=${MAINTAINER} \
	created=${BUILDDATE} \
	description=${DESCRIPTION}

RUN apt-get update && \
	DEBIAN_FRONTEND=noninteractive apt-get install -y graphviz git htop imagemagick inxi libsm6 libsixel-bin nano libxext6 python3 unzip vim zip && \
	rm -rf /var/lib/apt/lists/*
# RUN git clone https://github.com/Rita12321/nnUNetv2.git
# RUN cd nnUNetv2
# RUN pip install -e .
# COPY nnunetv2_pytorch_env.yml /tmp/ 
# #RUN conda env create -f /tmp/nnunetv2_pytorch_env.yml
# RUN conda env update --file /tmp/nnunetv2_pytorch_env.yml --prune


# COPY pip_requirements.txt /tmp/ 
# RUN pip install -r /tmp/pip_requirements.txt
# COPY . /tmp/
#RUN conda activate nnunetv2_pytorch
RUN conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
#RUN pip install nnunetv2

WORKDIR /
COPY ./ /etc/nnDoseNet
WORKDIR "/etc/nnDoseNet"
RUN pip install -e .
RUN echo "cat /etc/nnDoseNet" >> /etc/bash.bashrc



