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
# RUN conda install \
# 	asttokens=2.0.5 \
# 	backcall=0.2.0 \
# 	decorator=5.1.1 \
# 	executing=0.8.3 \
# 	ipython=8.8.0 \
# 	jedi=0.18.1 \
# 	matplotlib-inline=0.1.6 \
# 	parso=0.8.3 \
# 	pexpect=4.8.0 \
# 	pickleshare=0.7.5 \
# 	prompt-toolkit=3.0.36 \
# 	ptyprocess=0.7.0 \
# 	pure_eval=0.2.2 \
# 	pygments=2.11.2 \
# 	six=1.16.0 \
# 	stack_data=0.2.0 \
# 	traitlets=5.7.1 \
# 	wcwidth=0.2.5 \
# 	&& conda clean --all -y
# RUN pip install --upgrade pip \
# 	&& pip install \
# 	acvl-utils==0.2 \
# 	argparse==1.4.0 \
# 	batchgenerators==0.25 \
# 	certifi==2023.5.7 \
# 	charset-normalizer==3.1.0 \
# 	cmake==3.26.4 \
# 	connected-components-3d==3.12.1 \
# 	contourpy==1.1.0 \
# 	cycler==0.11.0 \
# 	dicom2nifti==2.4.8 \
# 	dynamic-network-architectures==0.2 \
# 	filelock==3.12.2 \
# 	fonttools==4.40.0 \
# 	future==0.18.3 \
# 	hiddenlayer==0.2 \
# 	idna==3.4 \
# 	imagecodecs==2023.3.16 \
# 	imageio==2.31.1 \
# 	importlib-resources==5.12.0 \
# 	jinja2==3.1.2 \
# 	joblib==1.3.1 \
# 	kiwisolver==1.4.4 \
# 	lazy-loader==0.3 \
# 	linecache2==1.0.0 \
# 	lit==16.0.6 \
# 	markupsafe==2.1.3 \
# 	matplotlib==3.7.1 \
# 	medpy==0.4.0 \
# 	mpmath==1.3.0 \
# 	networkx==3.1 \
# 	nibabel==5.1.0 \
# 	numpy==1.25.0 \
# 	nvidia-cublas-cu11==11.10.3.66 \
# 	nvidia-cuda-cupti-cu11==11.7.101 \
# 	nvidia-cuda-nvrtc-cu11==11.7.99 \
# 	nvidia-cuda-runtime-cu11==11.7.99 \
# 	nvidia-cudnn-cu11==8.5.0.96 \
# 	nvidia-cufft-cu11==10.9.0.58 \
# 	nvidia-curand-cu11==10.2.10.91 \
# 	nvidia-cusolver-cu11==11.4.0.1 \
# 	nvidia-cusparse-cu11==11.7.4.91 \
# 	nvidia-nccl-cu11==2.14.3 \
# 	nvidia-nvtx-cu11==11.7.91 \
# 	packaging==23.1 \
# 	pandas==2.0.3 \
# 	pillow==10.0.0 \ 
# 	pydicom==2.4.1 \
#     pyparsing==3.1.0 \
#     python-dateutil==2.8.2 \
# 	python-gdcm==3.0.22 \
# 	pytz==2023.3 \
# 	pywavelets==1.4.1 \
# 	pyyaml==6.0 \
# 	requests==2.31.0 \
# 	rt-utils==1.2.7\
# 	scikit-image==0.21.0 \
# 	scikit-learn==1.3.0 \
# 	scipy==1.11.1 \
# 	seaborn==0.12.2 \
# 	simpleitk==2.2.1 \
# 	sympy==1.12 \
# 	threadpoolctl==3.1.0 \
# 	tifffile==2023.4.12 \
# 	torch==2.0.1 \
# 	torchviz==0.0.2 \
# 	tqdm==4.65.0 \
# 	traceback2==1.4.0 \
# 	triton==2.0.0 \
# 	typing-extensions==4.7.1 \
# 	tzdata==2023.3 \
# 	unittest2==1.1.0 \
# 	urllib3==2.0.3 \
# 	yacs==0.1.8 \
# 	zipp==3.15.0


