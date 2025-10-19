# Class       : 2024-2 Mechatronics Integration Project
# Created     : 11/18/2024
# Name        : Eunji Ko
# Number      : 22100034
# Description:
#               - This code train the Real-ESRGAN.

# flake8: noqa
import os.path as osp
from basicsr.train import train_pipeline

import realesrgan.archs
import realesrgan.data
import realesrgan.models

if __name__ == '__main__':
    root_path = osp.abspath(osp.join(__file__, osp.pardir, osp.pardir))
    train_pipeline(root_path)
