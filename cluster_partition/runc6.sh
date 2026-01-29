#!/bin/bash
#SBATCH -x paraai-n32-h-01-agent-[7,8],paraai-n32-h-01-agent-28,paraai-n32-h-01-agent-16,paraai-n32-h-01-agent-17,paraai-n32-h-01-agent-1,paraai-n32-h-01-agent-48,paraai-n32-h-01-agent-4,paraai-n32-h-01-agent-[224,225,226,228]
# 1. 加载代码运行所需的服务器模块 (如 CUDA, GCC)
#echo "加载服务器模块..."
#module load compilers/cuda/12.1 compilers/gcc/11.3.0 cudnn/8.8.1.3_cuda12.x






module load miniforge3/24.1 compilers/cuda/12.1 compilers/gcc/11.3.0 cudnn/8.8.1.3_cuda12.x
source /home/bingxing2/apps/miniforge3/24.1.2/bin/activate torch
export PROJ_LIB=/home/bingxing2/home/scx7l1f/.conda/envs/torch/share/proj
export PATH=/home/bingxing2/home/scx7l1f/.conda/envs/torch/bin:$PATH
export PYTHONUNBUFFERED=1
export LD_PRELOAD=$LD_PRELOAD:/home/bingxing2/home/scx7l1f/.conda/envs/torch/lib/python3.9/site-packages/scikit_learn.libs/libgomp-d22c30c5.so.1.0.0
export LD_PRELOAD=$LD_PRELOAD:/home/bingxing2/home/scx7l1f/.conda/envs/torch/lib/libgomp.so.1
# python stack_sp1.py
python class813.py
