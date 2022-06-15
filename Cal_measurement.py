# -*- coding:utf-8 -*-
import tensorflow as tf
import numpy as np

class Measurement:
    def __init__(self, predict, label, shape, total_classes):
        self.predict = predict
        self.label = label
        self.total_classes = total_classes
        self.shape = shape

    def MIOU(self):
        # 0-crop, 1-weed, 2-background
        self.predict = np.reshape(self.predict, self.shape)
        #predict_ = np.array([self.predict[1], self.predict[0]])
        self.label = np.reshape(self.label, self.shape)
        #label_ = np.array([self.label[1], self.label[0]])

        predict_count = np.bincount(self.predict, minlength=self.total_classes)
        label_count = np.bincount(self.label, minlength=self.total_classes)

        temp = self.total_classes * np.array(self.label, dtype="int") + np.array(self.predict, dtype="int")  # Get category metrics

        temp_count = np.bincount(temp, minlength=self.total_classes * self.total_classes)
        cm = np.reshape(temp_count, [self.total_classes, self.total_classes])
        cm = np.diag(cm)

        U = label_count + predict_count - cm

        out = np.zeros((self.total_classes))
        miou = np.divide(cm, U, out=out, where=U != 0)
        miou = np.nanmean(miou)

        cm = tf.math.confusion_matrix(self.label, 
                                      self.predict,
                                      num_classes=self.total_classes).numpy()

        return miou, cm
