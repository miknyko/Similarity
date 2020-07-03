# FINE TUNED相似度分类模型

## 介绍

迁移ImageNet预训练好的特征提取器提取N类图片的特征，再使用Triplet Similarity Loss进一步微调图片，使某关注类图片聚簇缩小，计算聚类中心特征，通过计算待测图片特征与该聚类中心特征的几何距离，判断该图片是否属于此类图片。



## 环境安装

`Python3`

`pip install tensorflow-gpu==2.0.0 opencv-contrib-python`



## 一般使用步骤

### 训练

1. 为配合tf的DataGenerator，首先将N类训练图片放在同一文件夹下的不同文件夹

2. 运行app.py（使用前请阅读，并修改参数）

3. 训练模型，保存参数，生成聚类中心.npy文件

### 推理

1. 运行app.py ， 读取模型
2. 预测，生成结果文件

   

   