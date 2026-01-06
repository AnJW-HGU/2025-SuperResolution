#!/bin/bash

#SBATCH -J=e2e_5k_all_loss      # 작업명 지정
#SBATCH -p cas_v100_4           # queue  name  or  partiton name
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH -o %j.out          #작업 로그파일명 지정
#SBATCH -e %j.err          # 에러 로그 파일명 지정
#SBATCH --time=70:00:00         # 최대 작업 수행 시간 지정
#SBATCH --gres=gpu:1

python main.py