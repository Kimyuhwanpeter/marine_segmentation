# -*- coding:utf-8 -*-
import tensorflow as tf

def conv_relu(input, filters, name):
    return tf.keras.layers.Conv2D(filters=filters, kernel_size=3, padding="same", use_bias=True, activation='relu', name=name)(input)

def conv(input, filters, name, bias=True):
    return tf.keras.layers.Conv2D(filters=filters, kernel_size=3, padding="same", use_bias=bias, name=name)(input)

def conv_bn_relu(input, filters, name):
    input = tf.keras.layers.Conv2D(filters=filters, kernel_size=3, padding="same", use_bias=False, name=name)(input)
    input = tf.keras.layers.BatchNormalization()(input)
    input = tf.keras.layers.ReLU()(input)
    return input

def deconv_bn_relu(input, filters, name):
    input = tf.keras.layers.Conv2DTranspose(filters=filters, kernel_size=4, strides=2, padding='same', use_bias=False, name=name)(input)
    input = tf.keras.layers.BatchNormalization()(input)
    input = tf.keras.layers.ReLU()(input)
    return input

def parallel_Unet(input_shape=(512, 512, 3), nclasses=1):

    h = inputs = tf.keras.Input(input_shape)

    h_1 = conv_relu(h, 64, 'block1_conv1')
    h_1 = conv(h_1, 64, 'block1_conv2')
    h_1_f = conv_relu(h, 64, 'block1_conv1_f')
    h_1_f = conv(h_1_f, 64, 'block1_conv2_f')
    temp_h_1 = tf.nn.sigmoid(h_1_f) * h_1
    temp_h_1 = tf.keras.layers.ReLU()(temp_h_1)
    temp_h_1_f = tf.nn.sigmoid(h_1) * h_1_f
    temp_h_1_f = tf.keras.layers.ReLU()(temp_h_1_f)
    h_1 = temp_h_1
    h_1_f = temp_h_1_f

    h_1_pool = tf.keras.layers.MaxPool2D((2,2), 2)(h_1)
    h_1_pool_f = tf.keras.layers.MaxPool2D((2,2), 2)(h_1_f)

    h_2 = conv_relu(h_1_pool, 128, 'block2_conv1')
    h_2 = conv(h_2, 128, 'block2_conv2')
    h_2_f = conv_relu(h_1_pool_f, 128, 'block2_conv1_f')
    h_2_f = conv(h_2_f, 128, 'block2_conv2_f')
    temp_h_2 = tf.nn.sigmoid(h_2_f) * h_2
    temp_h_2 = tf.keras.layers.ReLU()(temp_h_2)
    temp_h_2_f = tf.nn.sigmoid(h_2) * h_2_f
    temp_h_2_f = tf.keras.layers.ReLU()(temp_h_2_f)
    h_2 = temp_h_2
    h_2_f = temp_h_2_f
    block2_conv2 = h_2
    block2_conv2_f = h_2_f

    h_2_pool = tf.keras.layers.MaxPool2D((2,2), 2)(h_2)
    h_2_pool_f = tf.keras.layers.MaxPool2D((2,2), 2)(h_2_f)

    h_3 = conv_relu(h_2_pool, 256, 'block3_conv1')
    h_3 = conv_relu(h_3, 256, 'block3_conv2')
    h_3 = conv(h_3, 256, 'block3_conv3')
    h_3_f = conv_relu(h_2_pool_f, 256, 'block3_conv1_f')
    h_3_f = conv_relu(h_3_f, 256, 'block3_conv2_f')
    h_3_f = conv(h_3_f, 256, 'block3_conv3_f')
    temp_h_3 = tf.nn.sigmoid(h_3_f) * h_3
    temp_h_3 = tf.keras.layers.ReLU()(temp_h_3)
    temp_h_3_f = tf.nn.sigmoid(h_3) * h_3_f
    temp_h_3_f = tf.keras.layers.ReLU()(temp_h_3_f)
    h_3 = temp_h_3
    h_3_f = temp_h_3_f
    block3_conv3 = h_3
    block3_conv3_f = h_3_f

    h_3_pool = tf.keras.layers.MaxPool2D((2,2), 2)(h_3)
    h_3_pool_f = tf.keras.layers.MaxPool2D((2,2), 2)(h_3_f)

    h_4 = conv_relu(h_3_pool, 512, 'block4_conv1')
    h_4 = conv_relu(h_4, 512, 'block4_conv2')
    h_4 = conv(h_4, 512, 'block4_conv3')
    h_4_f = conv_relu(h_3_pool_f, 512, 'block4_conv1_f')
    h_4_f = conv_relu(h_4_f, 512, 'block4_conv2_f')
    h_4_f = conv(h_4_f, 512, 'block4_conv3_f')
    temp_h_4 = tf.nn.sigmoid(h_4_f) * h_4
    temp_h_4 = tf.keras.layers.ReLU()(temp_h_4)
    temp_h_4_f = tf.nn.sigmoid(h_4) * h_4_f
    temp_h_4_f = tf.keras.layers.ReLU()(temp_h_4_f)
    h_4 = temp_h_4
    h_4_f = temp_h_4_f
    block4_conv3 = h_4
    block4_conv3_f = h_4_f

    h_4_pool = tf.keras.layers.MaxPool2D((2,2), 2)(h_4)
    h_4_pool_f = tf.keras.layers.MaxPool2D((2,2), 2)(h_4_f)

    h_5 = conv_relu(h_4_pool, 512, 'block5_conv1')
    h_5 = conv_relu(h_5, 512, 'block5_conv2')
    h_5 = conv(h_5, 512, 'block5_conv3')
    h_5_f = conv_relu(h_4_pool_f, 512, 'block5_conv1_f')
    h_5_f = conv_relu(h_5_f, 512, 'block5_conv2_f')
    h_5_f = conv(h_5_f, 512, 'block5_conv3_f')
    temp_h_5 = tf.nn.sigmoid(h_5_f) * h_5
    temp_h_5 = tf.keras.layers.ReLU()(temp_h_5)
    temp_h_5_f = tf.nn.sigmoid(h_5) * h_5_f
    temp_h_5_f = tf.keras.layers.ReLU()(temp_h_5_f)
    h_5 = temp_h_5
    h_5_f = temp_h_5_f
    block5_conv3 = h_5
    block5_conv3_f = h_5_f

    h_5_pool = tf.keras.layers.MaxPool2D((2,2), 2)(h_5)
    h_5_pool_f = tf.keras.layers.MaxPool2D((2,2), 2)(h_5_f)

    center_block_conv1 = conv_bn_relu(h_5_pool, 512, 'center_conv1')
    center_block_conv1 = conv(center_block_conv1, 512, 'center_conv2', False)
    center_block_conv1_f = conv_bn_relu(h_5_pool_f, 512, 'center_conv1_f')
    center_block_conv1_f = conv(center_block_conv1_f, 512, 'center_conv2_f', False)
    temp_center_block_conv1 = tf.nn.sigmoid(center_block_conv1_f) * center_block_conv1
    temp_center_block_conv1 = tf.keras.layers.BatchNormalization()(temp_center_block_conv1)
    temp_center_block_conv1 = tf.keras.layers.ReLU()(temp_center_block_conv1)
    temp_center_block_conv1_f = tf.nn.sigmoid(center_block_conv1) * center_block_conv1_f
    temp_center_block_conv1_f = tf.keras.layers.BatchNormalization()(temp_center_block_conv1_f)
    temp_center_block_conv1_f = tf.keras.layers.ReLU()(temp_center_block_conv1_f)
    center_block_conv1 = temp_center_block_conv1
    center_block_conv1_f = temp_center_block_conv1_f

    de_h_4 = deconv_bn_relu(center_block_conv1, 256, 'deconv1')
    de_h_4 = tf.concat([de_h_4, block5_conv3], -1)
    de_h_4_f = deconv_bn_relu(center_block_conv1_f, 256, 'deconv1_f')
    de_h_4_f = tf.concat([de_h_4_f, block5_conv3_f], -1)

    center_block_conv2 = conv(de_h_4, 256, 'center_conv3', False)
    center_block_conv2_f = conv(de_h_4_f, 256, 'center_conv3_f', False)
    temp_center_block_conv2 = tf.nn.sigmoid(center_block_conv2_f) * center_block_conv2
    temp_center_block_conv2 = tf.keras.layers.BatchNormalization()(temp_center_block_conv2)
    temp_center_block_conv2 = tf.keras.layers.ReLU()(temp_center_block_conv2)
    temp_center_block_conv2_f = tf.nn.sigmoid(center_block_conv2) * center_block_conv2_f
    temp_center_block_conv2_f = tf.keras.layers.BatchNormalization()(temp_center_block_conv2_f)
    temp_center_block_conv2_f = tf.keras.layers.ReLU()(temp_center_block_conv2_f)
    center_block_conv2 = temp_center_block_conv2
    center_block_conv2_f = temp_center_block_conv2_f

    de_h_3 = deconv_bn_relu(center_block_conv2, 128, 'deconv2')
    de_h_3 = tf.concat([de_h_3, block4_conv3], -1)
    de_h_3_f = deconv_bn_relu(center_block_conv2_f, 128, 'deconv2_f')
    de_h_3_f = tf.concat([de_h_3_f, block4_conv3_f], -1)

    center_block_conv3 = conv(de_h_3, 128, 'center_conv4', False)
    center_block_conv3_f = conv(de_h_3_f, 128, 'center_conv4_f', False)
    temp_center_block_conv3 = tf.nn.sigmoid(center_block_conv3_f) * center_block_conv3
    temp_center_block_conv3 = tf.keras.layers.BatchNormalization()(temp_center_block_conv3)
    temp_center_block_conv3 = tf.keras.layers.ReLU()(temp_center_block_conv3)
    temp_center_block_conv3_f = tf.nn.sigmoid(center_block_conv3) * center_block_conv3_f
    temp_center_block_conv3_f = tf.keras.layers.BatchNormalization()(temp_center_block_conv3_f)
    temp_center_block_conv3_f = tf.keras.layers.ReLU()(temp_center_block_conv3_f)
    center_block_conv3 = temp_center_block_conv3
    center_block_conv3_f = temp_center_block_conv3_f

    de_h_2 = deconv_bn_relu(center_block_conv3, 64, 'deconv3')
    de_h_2 = tf.concat([de_h_2, block3_conv3], -1)
    de_h_2_f = deconv_bn_relu(center_block_conv3_f, 64, 'deconv3_f')
    de_h_2_f = tf.concat([de_h_2_f, block3_conv3_f], -1)

    center_block_conv4 = conv(de_h_2, 64, 'center_conv5', False)
    center_block_conv4_f = conv(de_h_2_f, 64, 'center_conv5_f', False)
    temp_center_block_conv4 = tf.nn.sigmoid(center_block_conv4_f) * center_block_conv4
    temp_center_block_conv4 = tf.keras.layers.BatchNormalization()(temp_center_block_conv4)
    temp_center_block_conv4 = tf.keras.layers.ReLU()(temp_center_block_conv4)
    temp_center_block_conv4_f = tf.nn.sigmoid(center_block_conv4) * center_block_conv4_f
    temp_center_block_conv4_f = tf.keras.layers.BatchNormalization()(temp_center_block_conv4_f)
    temp_center_block_conv4_f = tf.keras.layers.ReLU()(temp_center_block_conv4_f)
    center_block_conv4 = temp_center_block_conv4
    center_block_conv4_f = temp_center_block_conv4_f

    de_h_1 = deconv_bn_relu(center_block_conv4, 32, 'deconv4')
    de_h_1 = tf.concat([de_h_1, block2_conv2], -1)
    de_h_1_f = deconv_bn_relu(center_block_conv4, 32, 'deconv4_f')
    de_h_1_f = tf.concat([de_h_1_f, block2_conv2_f], -1)

    center_block_conv5 = conv(de_h_1, 32, 'center_conv6', False)
    center_block_conv5_f = conv(de_h_1_f, 32, 'center_conv6_f', False)
    temp_center_block_conv5 = tf.nn.sigmoid(center_block_conv5_f) * center_block_conv5
    temp_center_block_conv5 = tf.keras.layers.BatchNormalization()(temp_center_block_conv5)
    temp_center_block_conv5 = tf.keras.layers.ReLU()(temp_center_block_conv5)
    temp_center_block_conv5_f = tf.nn.sigmoid(center_block_conv5) * center_block_conv5_f
    temp_center_block_conv5_f = tf.keras.layers.BatchNormalization()(temp_center_block_conv5_f)
    temp_center_block_conv5_f = tf.keras.layers.ReLU()(temp_center_block_conv5_f)
    center_block_conv5 = temp_center_block_conv5
    center_block_conv5_f = temp_center_block_conv5_f

    de_h_0 = deconv_bn_relu(center_block_conv5, 16, 'deconv5')
    center_block_conv6 = conv(de_h_0, 16, 'center_conv7', False)
    de_h_0_f = deconv_bn_relu(center_block_conv5_f, 16, 'deconv5_f')
    center_block_conv6_f = conv(de_h_0_f, 16, 'center_conv7_f', False)
    temp_center_block_conv6 = tf.nn.sigmoid(center_block_conv6_f) * center_block_conv6
    temp_center_block_conv6 = tf.keras.layers.BatchNormalization()(temp_center_block_conv6)
    temp_center_block_conv6 = tf.keras.layers.ReLU()(temp_center_block_conv6)
    temp_center_block_conv6_f = tf.nn.sigmoid(center_block_conv6) * center_block_conv6_f
    temp_center_block_conv6_f = tf.keras.layers.BatchNormalization()(temp_center_block_conv6_f)
    temp_center_block_conv6_f = tf.keras.layers.ReLU()(temp_center_block_conv6_f)
    center_block_conv6 = temp_center_block_conv6
    center_block_conv6_f = temp_center_block_conv6_f

    background_output = tf.keras.layers.Conv2D(filters=nclasses, kernel_size=3, padding="same")(center_block_conv6)
    object_output = tf.keras.layers.Conv2D(filters=nclasses, kernel_size=3, padding="same")(center_block_conv6_f)

    model = tf.keras.Model(inputs=inputs, outputs=[background_output, object_output])
    
    backbone = tf.keras.applications.vgg16.VGG16(input_shape=(224, 224, 3))
    
    model.get_layer('block1_conv1').set_weights(backbone.get_layer('block1_conv1').get_weights())
    model.get_layer('block1_conv1_f').set_weights(backbone.get_layer('block1_conv1').get_weights())

    model.get_layer('block1_conv2').set_weights(backbone.get_layer('block1_conv2').get_weights())
    model.get_layer('block1_conv2_f').set_weights(backbone.get_layer('block1_conv2').get_weights())
    
    model.get_layer('block2_conv1').set_weights(backbone.get_layer('block2_conv1').get_weights())
    model.get_layer('block2_conv1_f').set_weights(backbone.get_layer('block2_conv1').get_weights())
    
    model.get_layer('block2_conv2').set_weights(backbone.get_layer('block2_conv2').get_weights())
    model.get_layer('block2_conv2_f').set_weights(backbone.get_layer('block2_conv2').get_weights())
    
    model.get_layer('block3_conv1').set_weights(backbone.get_layer('block3_conv1').get_weights())
    model.get_layer('block3_conv1_f').set_weights(backbone.get_layer('block3_conv1').get_weights())
    
    model.get_layer('block3_conv2').set_weights(backbone.get_layer('block3_conv2').get_weights())
    model.get_layer('block3_conv2_f').set_weights(backbone.get_layer('block3_conv2').get_weights())
    
    model.get_layer('block3_conv3').set_weights(backbone.get_layer('block3_conv3').get_weights())
    model.get_layer('block3_conv3_f').set_weights(backbone.get_layer('block3_conv3').get_weights())
    
    model.get_layer('block4_conv1').set_weights(backbone.get_layer('block4_conv1').get_weights())
    model.get_layer('block4_conv1_f').set_weights(backbone.get_layer('block4_conv1').get_weights())
    
    model.get_layer('block4_conv2').set_weights(backbone.get_layer('block4_conv2').get_weights())
    model.get_layer('block4_conv2_f').set_weights(backbone.get_layer('block4_conv2').get_weights())
    
    model.get_layer('block4_conv3').set_weights(backbone.get_layer('block4_conv3').get_weights())
    model.get_layer('block4_conv3_f').set_weights(backbone.get_layer('block4_conv3').get_weights())
    
    model.get_layer('block5_conv1').set_weights(backbone.get_layer('block5_conv1').get_weights())
    model.get_layer('block5_conv1_f').set_weights(backbone.get_layer('block5_conv1').get_weights())
    
    model.get_layer('block5_conv2').set_weights(backbone.get_layer('block5_conv2').get_weights())
    model.get_layer('block5_conv2_f').set_weights(backbone.get_layer('block5_conv2').get_weights())
    
    model.get_layer('block5_conv3').set_weights(backbone.get_layer('block5_conv3').get_weights())
    model.get_layer('block5_conv3_f').set_weights(backbone.get_layer('block5_conv3').get_weights())
    return model
