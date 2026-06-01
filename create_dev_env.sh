#!/bin/bash
# 初始化conda
source ~/miniconda3/etc/profile.d/conda.sh
# 进入环境
conda activate /home/mgc/code/gac-robotics/env
# 后台启动ollama（如果没启动）
if ! pgrep -x ollama > /dev/null; then
    nohup ollama serve > ~/ollama.log 2>&1 &
fi
echo "Dev environment ready."


