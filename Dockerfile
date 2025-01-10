FROM tensorflow/tensorflow:2.9.3-gpu
# MAINTAINER	anonymous

ARG USERNAME=pcir
ARG USER_UID=1000
ARG USER_GID=$USER_UID

# Create the user
RUN groupadd --gid $USER_GID $USERNAME \
    && useradd --uid $USER_UID --gid $USER_GID -m $USERNAME \
    #
    # [Optional] Add sudo support. Omit if you don't need to install software after connecting.
    && apt -y update \
    && apt install -y sudo \
    && echo $USERNAME ALL=\(root\) NOPASSWD:ALL > /etc/sudoers.d/$USERNAME \
    && chmod 0440 /etc/sudoers.d/$USERNAME

# Install OpenEXR
RUN apt update && apt -y install openexr libopenexr-dev
# Install useful pacakges
RUN apt install -y git vim python3-dev libgl1-mesa-glx zsh wget ffmpeg

# Install required python packages
RUN	pip install --upgrade pip
RUN pip install scipy scikit-image Pillow tensorboard_plugin_profile opencv-python matplotlib tqdm

RUN pip install git+https://github.com/jamesbowman/openexrpython.git

# Make a symoblic link for CUDA
RUN ln -s /usr/local/cuda/lib64/libcudart.so /usr/lib/libcudart.so

#WORKDIR /
USER $USERNAME
WORKDIR /home/$USERNAME

# Install oh-my-zsh for convenience
RUN wget https://github.com/robbyrussell/oh-my-zsh/raw/master/tools/install.sh -O - | zsh || true

# Change theme in .zshrc to agnoster
RUN sed -i 's/ZSH_THEME="robbyrussell"/ZSH_THEME="agnoster"/g' ~/.zshrc

# FIX for tensorflow
RUN sudo sed -i 's/#include "third_party\/gpus\/cuda\/include\/cuda_fp16.h"/#include "cuda_fp16.h"/' /usr/local/lib/python3.8/dist-packages/tensorflow/include/tensorflow/core/util/gpu_kernel_helper.h
RUN sudo sed -i 's/#include "third_party\/gpus\/cuda\/include\/cuda.h"/#include "cuda.h"/' /usr/local/lib/python3.8/dist-packages/tensorflow/include/tensorflow/core/util/gpu_device_functions.h

# Change default shell to bash
SHELL ["/bin/bash", "-c"]
