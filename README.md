# UNet-segmentation
## Description
基于pytoch实现UNet的图像分割，数据集来自[kaggle](https://www.kaggle.com/franciscoescobar/satellite-images-of-water-bodies/tasks)。
## Dataset
数据集下载自[kaggle](https://www.kaggle.com/franciscoescobar/satellite-images-of-water-bodies/tasks),在train.py同级创建data文件夹，将下载的数据集文件放至data文件夹下。
## Train
数据集放置好后执行train.py文件进行训练。本项目在RTX2080Ti上完成，若在CPU上训练需修改train.py中DEVICE='CPU'
## Test
执行如下命令进行测试
`python test.py`
