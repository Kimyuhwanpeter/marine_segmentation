# -*- coding:utf-8 -*-
from model_1 import *
from random import shuffle, random
from Cal_measurement import *
from model_profiler import model_profiler

import matplotlib.pyplot as plt
import numpy as np
import easydict
import os

FLAGS = easydict.EasyDict({"img_size": 352,

                           "train_txt_path": "/yuhwan/yuhwan/Dataset/Segmentation/marine/train.txt",

                           "val_txt_path": "/yuhwan/yuhwan/Dataset/Segmentation/marine/val.txt",

                           "test_txt_path": "/yuhwan/yuhwan/Dataset/Segmentation/marine/test.txt",
                           
                           "label_path": "/yuhwan/yuhwan/Dataset/Segmentation/marine/masks/",
                           
                           "image_path": "/yuhwan/yuhwan/Dataset/Segmentation/marine/images/",
                           
                           "pre_checkpoint": False,
                           
                           "pre_checkpoint_path": "/yuhwan/yuhwan/checkpoint/Segmenation/6th_paper/Apple_A/checkpoint/62",
                           
                           "lr": 0.0001,

                           "min_lr": 1e-7,
                           
                           "epochs": 400,

                           "total_classes": 2,

                           "ignore_label": 0,

                           "batch_size": 10,

                           "sample_images": "/yuhwan/yuhwan/checkpoint/Segmenation/6th_paper/Apple_A/sample_images",

                           "save_checkpoint": "/yuhwan/yuhwan/checkpoint/Segmenation/6th_paper/Apple_A/checkpoint",

                           "save_print": "/yuhwan/yuhwan/checkpoint/Segmenation/6th_paper/Apple_A/train_out.txt",

                           "train_loss_graphs": "/yuwhan/Edisk/yuwhan/Edisk/Segmentation/V2/BoniRob/train_loss.txt",

                           "train_acc_graphs": "/yuwhan/Edisk/yuwhan/Edisk/Segmentation/V2/BoniRob/train_acc.txt",

                           "val_loss_graphs": "/yuwhan/Edisk/yuwhan/Edisk/Segmentation/V2/BoniRob/val_loss.txt",

                           "val_acc_graphs": "/yuwhan/Edisk/yuwhan/Edisk/Segmentation/V2/BoniRob/val_acc.txt",
                           
                           "test_images": "/yuhwan/yuhwan/checkpoint/Segmenation/MTS_CNN_related/BoniRob_v1_DSLR/test_images",

                           "train": True})


optim = tf.keras.optimizers.Adam(FLAGS.lr)
optim2 = tf.keras.optimizers.Adam(FLAGS.lr)
color_map = np.array([[0, 0, 0], [255, 255, 255]], dtype=np.uint8)

def tr_func(image_list, label_list):

    h = tf.random.uniform([1], 1e-2, 30)
    h = tf.cast(tf.math.ceil(h[0]), tf.int32)
    w = tf.random.uniform([1], 1e-2, 30)
    w = tf.cast(tf.math.ceil(w[0]), tf.int32)

    img = tf.io.read_file(image_list)
    img = tf.image.decode_jpeg(img, 3)
    img = tf.image.resize(img, [FLAGS.img_size, FLAGS.img_size])      # ????????????????????!?!?!?!?!?!?!?!?!?!?! tq
    img = tf.cast(img, tf.float32)
    img = tf.image.random_brightness(img, max_delta=50.)
    img = tf.image.random_saturation(img, lower=0.5, upper=1.5)
    img = tf.image.random_hue(img, max_delta=0.2)
    img = tf.image.random_contrast(img, lower=0.5, upper=1.5)
    img = tf.clip_by_value(img, 0, 255)
    # img = tf.image.random_crop(img, [FLAGS.img_size, FLAGS.img_size, 3], seed=123)
    no_img = img
    # img = img[:, :, ::-1] - tf.constant([103.939, 116.779, 123.68]) # ???հ? ????
    img = img / 255.

    lab = tf.io.read_file(label_list)
    lab = tf.image.decode_png(lab, 1)
    lab = tf.image.resize(lab, [FLAGS.img_size, FLAGS.img_size], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    # lab = tf.image.random_crop(lab, [FLAGS.img_size, FLAGS.img_size, 1], seed=123)
    lab = tf.image.convert_image_dtype(lab, tf.uint8)

    if random() > 0.5:
        img = tf.image.flip_left_right(img)
        lab = tf.image.flip_left_right(lab)

    return img, no_img, lab

def test_func(image_list, label_list):

    img = tf.io.read_file(image_list)
    img = tf.image.decode_jpeg(img, 3)
    img = tf.image.resize(img, [FLAGS.img_size, FLAGS.img_size])
    img = tf.clip_by_value(img, 0, 255)
    # img = img[:, :, ::-1] - tf.constant([103.939, 116.779, 123.68]) # ???հ? ????
    img = img / 255.

    lab = tf.io.read_file(label_list)
    lab = tf.image.decode_png(lab, 1)
    lab = tf.image.resize(lab, [FLAGS.img_size, FLAGS.img_size], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    lab = tf.image.convert_image_dtype(lab, tf.uint8)

    return img, lab

# @tf.function
def run_model(model, images, training=True):
    return model(images, training=training)

def true_dice_loss(y_true, y_pred):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.math.sigmoid(y_pred)
    numerator = 2 * tf.reduce_sum(y_true * y_pred)
    denominator = tf.reduce_sum(y_true + y_pred)

    return 1 - tf.math.divide(numerator, denominator)

def tversky_loss(y_true, y_pred, alpha):

    # get true pos (TP), false neg (FN), false pos (FP).
    true_pos  = tf.keras.backend.sum(y_true * y_pred)
    false_neg = tf.keras.backend.sum(y_true * (1-y_pred))
    false_pos = tf.keras.backend.sum((1-y_true) * y_pred)

    # TP/(TP + a*FN + b*FP); a+b = 1
    coef_val = (true_pos + 1)/(true_pos + alpha*false_neg + (1-alpha)*false_pos + 1)

    return 1 - coef_val

def reverse_tversky_loss_v2(y_true, y_pred, alpha):
    
    # get true pos (TP), false neg (FN), false pos (FP).
    true_neg  = tf.keras.backend.sum((1-y_true) * (1-y_pred))
    false_neg = tf.keras.backend.sum(y_true * (1-y_pred))
    false_pos = tf.keras.backend.sum((1-y_true) * y_pred)

    # TP/(TP + a*FN + b*FP); a+b = 1
    coef_val = (true_neg + 1)/(true_neg + alpha*false_neg + (1-alpha)*false_pos + 1)

    return 1 - coef_val

def reverse_tversky_loss(y_true, y_pred):

    # get true pos (TP), false neg (FN), false pos (FP).
    #true_pos = tf.keras.backend.sum(y_true * y_pred)
    true_neg = tf.keras.backend.sum((1 - y_true) * (1 - y_pred))
    false_neg = tf.keras.backend.sum(y_true * (1-y_pred))
    false_pos = tf.keras.backend.sum((1-y_true) * y_pred)

    # TP/(TP + a*FN + b*FP); a+b = 1
    coef_val = (true_neg + 1)/(true_neg + false_pos + 1)

    return 1 - coef_val

def true_dice_loss(y_true, y_pred):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.math.sigmoid(y_pred)
    numerator = 2 * tf.reduce_sum(y_true * y_pred)
    denominator = tf.reduce_sum(y_true + y_pred)

    return 1 - tf.math.divide(numerator, denominator)

def false_dice_loss(y_true, y_pred):
    y_true = 1 - tf.cast(y_true, tf.float32)
    y_pred = 1 - tf.math.sigmoid(y_pred)
    numerator = 2 * tf.reduce_sum(y_true * y_pred)
    denominator = tf.reduce_sum(y_true + y_pred)

    return 1 - tf.math.divide(numerator, denominator)

def structure_loss(y_true, y_pred):

    y_true = tf.expand_dims(y_true, -1)
    denominator = 1+5*tf.abs(tf.keras.layers.AvgPool2D((31, 31), strides=1, padding="same")(tf.cast(y_true, tf.float32)) - tf.cast(y_true, tf.float32))
    denominator = tf.squeeze(denominator, -1)
    numerator = tf.keras.losses.BinaryCrossentropy(from_logits=True, reduction=tf.keras.losses.Reduction.NONE)(y_true, y_pred)
    loss = tf.math.divide(tf.reduce_sum(denominator * numerator, [1, 2]), tf.reduce_sum(denominator, [1,2]))
    loss = tf.reduce_mean(loss)

    pred = tf.nn.sigmoid(y_pred)
    inter = tf.reduce_sum((pred*tf.cast(y_true, tf.float32))*tf.expand_dims(denominator, -1), [1,2])
    union = tf.reduce_sum((pred+tf.cast(y_true, tf.float32))*tf.expand_dims(denominator, -1), [1,2])
    wiou = 1-(inter)/(union-inter+tf.keras.backend.epsilon())
    wiou = tf.squeeze(wiou, -1)
    wiou = tf.reduce_mean(wiou)

    return loss+wiou

def cal_loss(model, model2, images, labels, object_buf, bin):
    
    with tf.GradientTape() as tape: # background - 0; object - 1
        batch_labels = tf.reshape(labels, [-1,])
        batch_labels = tf.cast(batch_labels, tf.float32)
        background_logits, object_logits = run_model(model, images, True)
        temp_background_logits = tf.nn.sigmoid(object_logits) * background_logits
        temp_object_logits = tf.nn.sigmoid(background_logits) * object_logits
        object_logits = temp_object_logits
        background_logits = temp_background_logits
        b_logits = tf.reshape(background_logits, [-1, ])
        o_logits = tf.reshape(object_logits, [-1, ])

        m = max(object_buf[0], object_buf[1])
        if bin[0] != 0 and bin[1] != 0:
            background_loss = false_dice_loss(batch_labels, b_logits)
            object_loss = structure_loss(labels, object_logits)

            total_loss = background_loss * object_buf[0] + object_loss

        if bin[0] != 0 and bin[1] == 0:
            background_loss = false_dice_loss(batch_labels, b_logits) * object_buf[0]
            total_loss = background_loss
        
    grads = tape.gradient(total_loss, model.trainable_variables)
    optim.apply_gradients(zip(grads, model.trainable_variables))

    with tf.GradientTape() as tape2:
        batch_labels = tf.reshape(labels, [-1,])
        batch_labels = tf.cast(batch_labels, tf.float32)
        background_logits, object_logits = run_model(model, [images, background_logits, object_logits], True)
        temp_background_logits = tf.nn.sigmoid(object_logits) * background_logits
        temp_object_logits = tf.nn.sigmoid(background_logits) * object_logits
        object_logits = temp_object_logits
        background_logits = temp_background_logits
        b_logits = tf.reshape(background_logits, [-1, ])
        o_logits = tf.reshape(object_logits, [-1, ])

        m = max(object_buf[0], object_buf[1])
        if bin[0] != 0 and bin[1] != 0:
            background_loss = false_dice_loss(batch_labels, b_logits)
            object_loss = structure_loss(labels, object_logits)

            total_loss = background_loss * object_buf[0] + object_loss

        if bin[0] != 0 and bin[1] == 0:
            background_loss = false_dice_loss(batch_labels, b_logits) * object_buf[0]
            total_loss = background_loss

    grads2 = tape.gradient(total_loss, model.trainable_variables)
    optim2.apply_gradients(zip(grads2, model.trainable_variables))
    
    return total_loss

def main():
    tf.keras.backend.clear_session()

    model = parallel_Unet(input_shape=(FLAGS.img_size, FLAGS.img_size, 3), nclasses=1)
    prob = model_profiler(model, FLAGS.batch_size)
    model.summary()
    print(prob)

    if FLAGS.pre_checkpoint:
        ckpt = tf.train.Checkpoint(model=model, optim=optim)
        ckpt_manager = tf.train.CheckpointManager(ckpt, FLAGS.pre_checkpoint_path, 5)

        if ckpt_manager.latest_checkpoint:
            ckpt.restore(ckpt_manager.latest_checkpoint)
            print("Restored!!")
    
    if FLAGS.train:
        count = 0
        output_text = open(FLAGS.save_print, "w")
        
        train_list = np.loadtxt(FLAGS.train_txt_path, dtype="<U200", skiprows=0, usecols=0)
        val_list = np.loadtxt(FLAGS.val_txt_path, dtype="<U200", skiprows=0, usecols=0)
        test_list = np.loadtxt(FLAGS.test_txt_path, dtype="<U200", skiprows=0, usecols=0)

        train_img_dataset = [FLAGS.image_path + data + ".jpg" for data in train_list]
        val_img_dataset = [FLAGS.image_path + data + ".jpg" for data in val_list]
        test_img_dataset = [FLAGS.image_path + data + ".jpg" for data in test_list]

        train_lab_dataset = [FLAGS.label_path + data + ".png" for data in train_list]
        val_lab_dataset = [FLAGS.label_path + data + ".png" for data in val_list]
        test_lab_dataset = [FLAGS.label_path + data + ".png" for data in test_list]

        val_ge = tf.data.Dataset.from_tensor_slices((val_img_dataset, val_lab_dataset))
        val_ge = val_ge.map(test_func)
        val_ge = val_ge.batch(1)
        val_ge = val_ge.prefetch(tf.data.experimental.AUTOTUNE)

        test_ge = tf.data.Dataset.from_tensor_slices((test_img_dataset, test_lab_dataset))
        test_ge = test_ge.map(test_func)
        test_ge = test_ge.batch(1)
        test_ge = test_ge.prefetch(tf.data.experimental.AUTOTUNE)
        count = 0
        for epoch in range(FLAGS.epochs):
            A = list(zip(train_img_dataset, train_lab_dataset))
            shuffle(A)
            train_img_dataset, train_lab_dataset = zip(*A)
            train_img_dataset, train_lab_dataset = np.array(train_img_dataset), np.array(train_lab_dataset)

            train_ge = tf.data.Dataset.from_tensor_slices((train_img_dataset, train_lab_dataset))
            train_ge = train_ge.shuffle(len(train_img_dataset))
            train_ge = train_ge.map(tr_func)
            train_ge = train_ge.batch(FLAGS.batch_size)
            train_ge = train_ge.prefetch(tf.data.experimental.AUTOTUNE)
            tr_iter = iter(train_ge)
            tr_idx = len(train_img_dataset) // FLAGS.batch_size

            tr_iter = iter(train_ge)
            tr_idx = len(train_img_dataset) // FLAGS.batch_size
            for step in range(tr_idx):
                batch_images, print_images, batch_labels = next(tr_iter)  
                batch_labels = batch_labels.numpy()
                batch_labels = np.where(batch_labels == 255, 1, batch_labels)
                batch_labels = np.squeeze(batch_labels, -1)

                class_imbal_labels_buf = 0.
                class_imbal_labels = batch_labels
                for i in range(FLAGS.batch_size):
                    class_imbal_label = class_imbal_labels[i]
                    class_imbal_label = np.reshape(class_imbal_label, [FLAGS.img_size*FLAGS.img_size, ])
                    count_c_i_lab = np.bincount(class_imbal_label, minlength=FLAGS.total_classes)
                    class_imbal_labels_buf += count_c_i_lab

                bin = class_imbal_labels_buf
                object_buf = np.array(bin, dtype=np.float32)
                object_buf = (np.max(object_buf / np.sum(object_buf)) + 1 - (object_buf / np.sum(object_buf)))
                object_buf = tf.nn.softmax(object_buf).numpy()

                loss = cal_loss(model, batch_images, batch_labels, object_buf, bin)
                if count % 10 == 0:
                    print("Epoch: {} [{}/{}] loss = {}".format(epoch, step+1, tr_idx, loss))

                if count % 100 == 0:
                    background_logits, object_logits = run_model(model, batch_images, False)
                    temp_background_logits = tf.nn.sigmoid(object_logits) * background_logits
                    temp_object_logits = tf.nn.sigmoid(background_logits) * object_logits

                    background_logits = tf.nn.sigmoid(temp_background_logits[:, :, :, 0])
                    object_logits = tf.nn.sigmoid(temp_object_logits[:, :, :, 0])

                    for i in range(FLAGS.batch_size):
                        label = tf.cast(batch_labels[i], tf.int32).numpy()
                        b_logits = tf.where(background_logits[i] >= 0.5, 1, 0)
                        o_logits = tf.where(object_logits[i] >= 0.5, 1, 0)

                        image = o_logits + b_logits
                        image = tf.where(image == 2, 1, 0).numpy()

                        pred_mask_color = color_map[image]
                        label_mask_color = color_map[label]

                        plt.imsave(FLAGS.sample_images + "/{}_batch_{}".format(count, i) + "_label.png", label_mask_color)
                        plt.imsave(FLAGS.sample_images + "/{}_batch_{}".format(count, i) + "_predict.png", pred_mask_color)


                count +=1

            tr_iter = iter(train_ge)
            iou = 0.
            precision_ = 0.
            recall_ = 0.
            f1_score_ = 0.
            C = 0.
            cm = 0.
            miou = 0.
            for i in range(tr_idx):
                batch_images, _, batch_labels = next(tr_iter)
                batch_labels = tf.where(batch_labels == 255, 1, batch_labels).numpy()
                for j in range(FLAGS.batch_size):
                    batch_image = tf.expand_dims(batch_images[j], 0)
                    background_logits, object_logits = run_model(model, batch_image, False)
                    temp_background_logits = tf.nn.sigmoid(object_logits) * background_logits
                    temp_object_logits = tf.nn.sigmoid(background_logits) * object_logits
                    background_logits = tf.nn.sigmoid(temp_background_logits[0, :, :, 0])
                    object_logits = tf.nn.sigmoid(temp_object_logits[0, :, :, 0])

                    b_logits = tf.where(background_logits >= 0.5, 1, 0)
                    o_logits = tf.where(object_logits >= 0.5, 1, 0)

                    final_output = o_logits + b_logits
                    final_output = tf.where(final_output == 2, 1, 0)
                    final_output = tf.where(final_output == 0, 1, 0).numpy()

                    batch_label = tf.cast(batch_labels[j, :, :, 0], tf.uint8).numpy()
                    batch_label = np.where(batch_label == 0, 1, 0)
                    batch_label = np.array(batch_label, np.int32)

                    a = np.bincount(np.reshape(batch_label, [-1]), minlength=2)[1]
                    if a != 0:
                        C +=1.
                        miou_, cm_ = Measurement(predict=final_output,
                                            label=batch_label, 
                                            shape=[FLAGS.img_size*FLAGS.img_size, ], 
                                            total_classes=2).MIOU()
                    
                        cm_ = tf.cast(cm_, tf.float32)
                        cm += cm_
                        miou += miou_

                        precision = tf.math.divide_no_nan(cm_[0,0], (cm_[0,0] + cm_[1,0]))
                        recall = tf.math.divide_no_nan(cm_[0,0], (cm_[0,0] + cm_[0,1]))

                        precision_ += precision
                        recall_ += recall
                        f1_score_ += tf.math.divide_no_nan((2*precision*recall), (precision + recall))

            iou = cm[0,0]/(cm[0,0] + cm[0,1] + cm[1,0])
            print("train mIoU = %.4f, train F1_score = %.4f, train sensitivity(recall) = %.4f, train precision = %.4f" % (miou / C,
                                                                                                                        f1_score_ / C,
                                                                                                                        recall_ / C,
                                                                                                                        precision_ / C))
            output_text.write("Epoch: ")
            output_text.write(str(epoch))
            output_text.write("===================================================================")
            output_text.write("\n")
            output_text.write("train IoU: ")
            output_text.write("%.4f" % (miou / C))
            output_text.write(", train F1_score: ")
            output_text.write("%.4f" % (f1_score_ / C))
            output_text.write(", train sensitivity: ")
            output_text.write("%.4f" % (recall_  / C))
            output_text.write(", train precision: ")
            output_text.write("%.4f" % (precision_ / C))
            output_text.write("\n")


            val_iter = iter(val_ge)
            iou = 0.
            precision_ = 0.
            recall_ = 0.
            f1_score_ = 0.
            C = 0.
            cm = 0.
            miou = 0.
            for i in range(len(val_img_dataset)):
                batch_images, batch_labels = next(val_iter)
                batch_labels = tf.where(batch_labels == 255, 1, batch_labels).numpy()
                background_logits, object_logits = run_model(model, batch_images, False)
                temp_background_logits = tf.nn.sigmoid(object_logits) * background_logits
                temp_object_logits = tf.nn.sigmoid(background_logits) * object_logits
                background_logits = tf.nn.sigmoid(temp_background_logits[0, :, :, 0])
                object_logits = tf.nn.sigmoid(temp_object_logits[0, :, :, 0])

                b_logits = tf.where(background_logits >= 0.5, 1, 0)
                o_logits = tf.where(object_logits >= 0.5, 1, 0)

                final_output = o_logits + b_logits
                final_output = tf.where(final_output == 2, 1, 0)
                final_output = tf.where(final_output == 0, 1, 0).numpy()

                batch_label = tf.cast(batch_labels[0, :, :, 0], tf.uint8).numpy()
                batch_label = np.where(batch_label == 0, 1, 0)
                batch_label = np.array(batch_label, np.int32)

                a = np.bincount(np.reshape(batch_label, [-1]), minlength=2)[1]
                if a != 0:
                    C +=1.
                    miou_, cm_ = Measurement(predict=final_output,
                                        label=batch_label, 
                                        shape=[FLAGS.img_size*FLAGS.img_size, ], 
                                        total_classes=2).MIOU()
                    
                    cm_ = tf.cast(cm_, tf.float32)
                    cm += cm_
                    miou += miou_

                    precision = tf.math.divide_no_nan(cm_[0,0], (cm_[0,0] + cm_[1,0]))
                    recall = tf.math.divide_no_nan(cm_[0,0], (cm_[0,0] + cm_[0,1]))

                    precision_ += precision
                    recall_ += recall
                    f1_score_ += tf.math.divide_no_nan((2*precision*recall), (precision + recall))

            iou = cm[0,0]/(cm[0,0] + cm[0,1] + cm[1,0])
            print("val mIoU = %.4f, val F1_score = %.4f, val sensitivity(recall) = %.4f, val precision = %.4f" % (miou / C,
                                                                                                                f1_score_ / C,
                                                                                                                recall_ / C,
                                                                                                                precision_ / C))

            output_text.write("val IoU: ")
            output_text.write("%.4f" % (miou / C ))
            output_text.write(", val F1_score: ")
            output_text.write("%.4f" % (f1_score_ / C))
            output_text.write(", val sensitivity: ")
            output_text.write("%.4f" % (recall_  / C))
            output_text.write(", val precision: ")
            output_text.write("%.4f" % (precision_ / C))
            output_text.write("\n")

            test_iter = iter(test_ge)
            iou = 0.
            precision_ = 0.
            recall_ = 0.
            f1_score_ = 0.
            C = 0.
            cm = 0.
            miou = 0.
            for i in range(len(test_img_dataset)):
                batch_images, batch_labels = next(test_iter)
                batch_labels = tf.where(batch_labels == 255, 1, batch_labels).numpy()
                background_logits, object_logits = run_model(model, batch_images, False)
                temp_background_logits = tf.nn.sigmoid(object_logits) * background_logits
                temp_object_logits = tf.nn.sigmoid(background_logits) * object_logits
                background_logits = tf.nn.sigmoid(temp_background_logits[0, :, :, 0])
                object_logits = tf.nn.sigmoid(temp_object_logits[0, :, :, 0])

                b_logits = tf.where(background_logits >= 0.5, 1, 0)
                o_logits = tf.where(object_logits >= 0.5, 1, 0)

                final_output = o_logits + b_logits
                final_output = tf.where(final_output == 2, 1, 0)
                final_output = tf.where(final_output == 0, 1, 0).numpy()

                batch_label = tf.cast(batch_labels[0, :, :, 0], tf.uint8).numpy()
                batch_label = np.where(batch_label == 0, 1, 0)
                batch_label = np.array(batch_label, np.int32)

                a = np.bincount(np.reshape(batch_label, [-1]), minlength=2)[1]
                if a != 0:
                    C +=1.
                    miou_, cm_ = Measurement(predict=final_output,
                                        label=batch_label, 
                                        shape=[FLAGS.img_size*FLAGS.img_size, ], 
                                        total_classes=2).MIOU()
                    
                    cm_ = tf.cast(cm_, tf.float32)
                    cm += cm_
                    miou += miou_

                    precision = tf.math.divide_no_nan(cm_[0,0], (cm_[0,0] + cm_[1,0]))
                    recall = tf.math.divide_no_nan(cm_[0,0], (cm_[0,0] + cm_[0,1]))

                    precision_ += precision
                    recall_ += recall
                    f1_score_ += tf.math.divide_no_nan((2*precision*recall), (precision + recall))

            iou = cm[0,0]/(cm[0,0] + cm[0,1] + cm[1,0])
            print("test mIoU = %.4f, test F1_score = %.4f, test sensitivity(recall) = %.4f, test precision = %.4f" % (miou / C,
                                                                                                                    f1_score_ / C,
                                                                                                                    recall_ / C,
                                                                                                                    precision_ / C))
            output_text.write("test IoU: ")
            output_text.write("%.4f" % (miou / C))
            output_text.write(", test F1_score: ")
            output_text.write("%.4f" % (f1_score_ / C))
            output_text.write(", test sensitivity: ")
            output_text.write("%.4f" % (recall_  / C))
            output_text.write(", test precision: ")
            output_text.write("%.4f" % (precision_ / C))
            output_text.write("\n")
            output_text.write("===================================================================")
            output_text.write("\n")
            output_text.flush()

            model_dir = "%s/%s" % (FLAGS.save_checkpoint, epoch)
            if not os.path.isdir(model_dir):
                print("Make {} folder to store the weight!".format(epoch))
                os.makedirs(model_dir)
            ckpt = tf.train.Checkpoint(model=model, optim=optim)
            ckpt_dir = model_dir + "/marine_{}.ckpt".format(epoch)
            ckpt.save(ckpt_dir)

    else:
        test_list = np.loadtxt(FLAGS.test_txt_path, dtype="<U200", skiprows=0, usecols=0)

        test_img_dataset = [FLAGS.image_path + data + ".jpg" for data in test_list]
        test_lab_dataset = [FLAGS.label_path + data + ".png" for data in test_list]

        test_ge = tf.data.Dataset.from_tensor_slices((test_img_dataset, test_lab_dataset))
        test_ge = test_ge.map(test_func)
        test_ge = test_ge.batch(1)
        test_ge = test_ge.prefetch(tf.data.experimental.AUTOTUNE)

        test_iter = iter(test_ge)
        iou = 0.
        precision_ = 0.
        recall_ = 0.
        f1_score_ = 0.
        C = 0.
        cm = 0.
        miou = 0.
        for i in range(len(test_img_dataset)):
            batch_images, batch_labels = next(test_iter)
            batch_labels = tf.where(batch_labels == 255, 1, batch_labels).numpy()
            background_logits, object_logits = run_model(model, batch_images, False)
            temp_background_logits = tf.nn.sigmoid(object_logits) * background_logits
            temp_object_logits = tf.nn.sigmoid(background_logits) * object_logits
            background_logits = tf.nn.sigmoid(temp_background_logits[0, :, :, 0])
            object_logits = tf.nn.sigmoid(temp_object_logits[0, :, :, 0])

            b_logits = tf.where(background_logits >= 0.5, 1, 0)
            o_logits = tf.where(object_logits >= 0.5, 1, 0)

            final_output = o_logits + b_logits
            final_output = tf.where(final_output == 2, 1, 0)

            pred_mask_color = color_map[final_output]

            final_output = tf.where(final_output == 0, 1, 0).numpy()

            batch_label = tf.cast(batch_labels[0, :, :, 0], tf.uint8).numpy()
            batch_label = np.where(batch_label == 0, 1, 0)
            batch_label = np.array(batch_label, np.int32)

            a = np.bincount(np.reshape(batch_label, [-1]), minlength=2)[1]
            if a != 0:
                C +=1.
                miou_, cm_ = Measurement(predict=final_output,
                                    label=batch_label, 
                                    shape=[FLAGS.img_size*FLAGS.img_size, ], 
                                    total_classes=2).MIOU()

                cm_ = tf.cast(cm_, tf.float32)
                cm += cm_
                miou += miou_

                precision = tf.math.divide_no_nan(cm_[0,0], (cm_[0,0] + cm_[1,0]))
                recall = tf.math.divide_no_nan(cm_[0,0], (cm_[0,0] + cm_[0,1]))

                precision_ += precision
                recall_ += recall
                f1_score_ += tf.math.divide_no_nan((2*precision*recall), (precision + recall))

            plt.imsave("C:/Users/Yuhwan/Downloads/t/" + test_img_dataset[i].split('/')[-1], pred_mask_color)

        iou = cm[0,0]/(cm[0,0] + cm[0,1] + cm[1,0])
        print("test mIoU = %.4f, test F1_score = %.4f, test sensitivity(recall) = %.4f, test precision = %.4f" % (miou / C,
                                                                                                                f1_score_ / C,
                                                                                                                recall_ / C,
                                                                                                                precision_ / C))
        print(miou / C)




        #    a = np.bincount(np.reshape(batch_label, [-1]), minlength=2)[1]
        #    if a != 0:
        #        C +=1.
        #        cm = Measurement(predict=final_output,
        #                            label=batch_label, 
        #                            shape=[FLAGS.img_size*FLAGS.img_size, ], 
        #                            total_classes=2).MIOU()
        #        cm = tf.cast(cm, tf.float32)
        #        iou += tf.math.divide_no_nan(cm[0,0], (cm[0,0] + cm[0,1] + cm[1,0]))
        #        precision_ += tf.math.divide_no_nan(cm[0,0], (cm[0,0] + cm[1,0]))
        #        recall_ += tf.math.divide_no_nan(cm[0,0], (cm[0,0] + cm[0,1]))

        #f1_score = tf.math.divide_no_nan((2*(precision_ / C)*(recall_ / C)), ((precision_ / C) + (recall_ / C)))
        #print("test mIoU = %.4f, test F1_score = %.4f, test sensitivity(recall) = %.4f, test precision = %.4f" % (iou / C,
        #                                                                                                        f1_score,
        #                                                                                                        recall_ / C,
        #                                                                                                        precision_ / C))
        print(C)
if __name__ == "__main__":
    main()
