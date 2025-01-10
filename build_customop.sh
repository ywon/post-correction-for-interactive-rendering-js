#!/bin/bash

TF_CFLAGS=( $(python -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_compile_flags()))') )
TF_LFLAGS=( $(python -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_link_flags()))') )

sed -i '/#define GOOGLE_CUDA/c\/\/ #define GOOGLE_CUDA' scripts/reproject.cu.cc
nvcc -std=c++14 -c -o ops/reproject.cu.o scripts/reproject.cu.cc ${TF_CFLAGS[@]} -D GOOGLE_CUDA=1 -x cu -Xcompiler -fPIC -expt-relaxed-constexpr -gencode arch=compute_75,code=sm_75
g++ -std=c++14 -I/usr/local/cuda/include -shared -o ops/reproject.so scripts/reproject.cc ops/reproject.cu.o ${TF_CFLAGS[@]} -fPIC -lcudart ${TF_LFLAGS[@]}
sed -i 's~// #define GOOGLE_CUDA$~#define GOOGLE_CUDA~' scripts/reproject.cu.cc
