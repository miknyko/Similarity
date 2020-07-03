#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#Author: Rockstar He
#Date: 2020-07-02
#Description:

import tensorflow as tf
import numpy as np
import os
import cv2

from PIL import Image
from tf.keras.preprocessing.image import ImageDataGenerator
from tf.keras.applications import Xception
from tf.keras import Sequential
from tf.keras.callbacks import ModelCheckpoint
from tf.keras.utils import Sequence

from triplet_loss import batch_hard_triplet_loss,batch_all_triplet_loss,batch_hard_triplet_loss_onevsall
from triplet_loss import _get_anchor_positive_triplet_mask_onevsall


class ImageLoader(Sequence):
    """图片读取器"""
    def __init__(self, path, batch_size):
        self.path = path
        self.batch_size = batch_size
        self.filename_list = os.listdir(self.path)

    def __len__(self):
        no_image = len(self.filename_list)
        return math.ceil(no_image / self.batch_size)

    def __getitem__(self, idx):
        index_raw = self.filename_list[idx * self.batch_size:(idx+1) * self.batch_size]
        X = np.zeros((self.batch_size,299,299,3))
        for i,img_name in enumerate(index_raw):
            img_path = os.path.join(self.path,img_name)
            img = cv2.imread(img_path)
            if img is not None:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = cv2.resize(img,(299,299))
                img = img.astype(np.float32) / 255.

                X[i,] = img
        return X

class SimilarityModel():
    """一种fine tune之后的图片特征提取器"""
    def __init__(self):
        # 最终feature map size
        self.embedding_size = 128
        # 使用Xception作为原始特征提取器，可替换为vgg16或resnet
        self.base_model = Xception(include_top=False, weights='imagenet', input_shape=(299,299,3), pooling='avg')
        for layer in self.base_model.layers:
            layer.trainable = False
        # 最终完整网络
        self.model = Sequential(
            [self.base_model,
            tf.keras.layers.Dense(self.embedding_size, activation='relu'),
            tf.keras.layers.Lambda(lambda x: tf.math.l2_normalize(x, axis=-1))]
        )
    
    def load_train_data(self, path, batch_size=1024):
        """
        读取各类图片，并进行数据增强，产生标签
        @param path(str):所有类图片的根文件夹
        @param batch_size(int):batch大小，影响结果，建议越大越好 
        """
        # 数据增强可适当调整
        self.train_data = ImageDataGenerator(rescale=1./255, shear_range=0.2, zoom_range=0.2, horizontal_flip=True)
        self.train_generator = train_data.flow_from_directory(path, target_size=(299,299), batch_size=1024, class_mode='sparse')
        # 需注意分类标签
        print(self.train_generator.class_indices)

    def train(self, focus, epochs=30, margin=1):
        """
        训练模型，即对其进行微调
        @param margin(int):关注图片和异类图片之间的特征距离
        @param focus(int):关注图片的标签，若使用tf自带DataGenerator，需注意此标签
        """
        
        def custom_loss(labels, embedding):
            return batch_hard_triplet_loss_onevsall(labels, embedding, margin=margin, focus=focus)

        self.model.compile(loss=custom_loss, optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5))
        self.model.fit(self.train_generator, epochs=epochs, callbacks=[ModelCheckpoint(filepath='ckps/ckp.h5',save_weights_only=True)])
        self.model.save_weights(f'models/focus_{focus}_model.h5')
    
    def inference_center(self, focus_path, weights_path=[]):
        """
        推理一次，生成关注图片类的特征中心，保存为npy文件
        @param focus_path(str):特征图片的文件夹
        @param weights_path(str):optional, 模型文件路径，如果不提供，则默认刚训练好的模型
        """
        if weights_path:
            self.model.load_weights(weights_path)
        focus_data = ImageLoader(focus_path,128)
        res = self.model.predict(focus_data)
        cluster_center = np.mean(res, axis=0)
        np.save(f'models/focus_{name}_center',cluster_center)

    def batch_predict(self,img_path,weight_path,center_path):
        """
        批量预测图片，产生分数
        @param img_path(str):待预测的图片文件夹路径
        @param weight_path(str):模型参数保存路径
        @param center_path(str):特征中心保存路径
        @param 
        @return score(dict):图片得分，键为图片名
        """
        def custom_loss(labels, embedding):
            return batch_hard_triplet_loss_onevsall(labels, embedding, margin=1, focus=0) # 定义loss只为顺利加载模型，不影响推理

        try:
            self.model.load_weights(weight_path)
        except:
            self.model = tf.keras.models.load_model(weight_path, custom_objects={'custom_loss':custom_loss}) # 老模型是保存的整个模型，而非参数

        data = ImageLoader(img_path,128)
        res = self.model.predict(data)
        center = np.load(center_path)
        batch_distance = np.sqrt(np.sum((res - center) ** 2, axis=1))
        score = {}
        for image,dis in zip(data.filename_list, batch_distance):
            score[image] = dis
        
        return score
        


def train():
    """训练模型示范"""
    train_data_path = ''
    # 初始化一个模型
    model = SimilarityModel()
    # 创建数据读取器, 观察关注图片标签是哪一类
    model.load_train_data(train_data_path)
    # 然后训练
    model.train(focus)
    # 生成特征中心文件
    model.inference_center(focus_path)

def predict():
    """使用训练好的模型预测示范"""
    data_path = ''
    model = SimilarityModel()
    model.batch_predict(data_path,weight_path,center_path)

if __name__ == "__main__":
    main()
