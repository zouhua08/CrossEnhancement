from __future__ import print_function

import os
import time
import random

from PIL import Image
import tensorflow as tf
import numpy as np
import math

from utils import *



def CBAM(input, reduction=8):
    """
    @Convolutional Block Attention Module
    """

    _, width, height, channel = input.get_shape()  # (B, W, H, C)

    # channel attention
    x_mean = tf.reduce_mean(input, axis=(1, 2), keepdims=True)   # (B, 1, 1, C)
    x_mean = tf.layers.conv2d(x_mean, channel // reduction, 1, activation=tf.nn.relu)  # (B, 1, 1, C // r)
    x_mean = tf.layers.conv2d(x_mean, channel, 1)   # (B, 1, 1, C)

    x_max = tf.reduce_max(input, axis=(1, 2), keepdims=True)  # (B, 1, 1, C)
    x_max = tf.layers.conv2d(x_max, channel // reduction, 1, activation=tf.nn.relu)
    # (B, 1, 1, C // r)
    x_max = tf.layers.conv2d(x_max, channel, 1)  # (B, 1, 1, C)

    x = tf.add(x_mean, x_max)   # (B, 1, 1, C)
    x = tf.nn.sigmoid(x)        # (B, 1, 1, C)
    x = tf.multiply(input, x)   # (B, W, H, C)

    # spatial attention
    y_mean = tf.reduce_mean(x, axis=3, keepdims=True)  # (B, W, H, 1)
    y_max = tf.reduce_max(x, axis=3, keepdims=True)  # (B, W, H, 1)
    y = tf.concat([y_mean, y_max], axis=-1)     # (B, W, H, 2)
    y = tf.layers.conv2d(y, 1, 7, padding='same', activation=tf.nn.sigmoid)    # (B, W, H, 1)
    y = tf.multiply(x, y)  # (B, W, H, C)

    return y


def resBlock(x):
    tmp = tf.layers.conv2d(x, 64, 3, 1, padding='same', activation=None)
    tmp = tf.nn.relu(tmp)

    tmp = CBAM(tmp)
    tmp = tmp + x

    tmp = tf.layers.conv2d(tmp, 64, 3, 1, padding='same', activation=None)
    
    return tmp


def AttnNet(low_light_input, illumination_input, is_training=True):

    num_blocks = 8

    x = tf.concat([low_light_input, illumination_input], axis=3)

    x = tf.layers.conv2d(x, 64, 3, 1, padding='same', activation=None)
    conv_1 = x

    for i in range(num_blocks):
        x = resBlock(x)

    x = tf.layers.conv2d(x, 64, 3, 1, padding='same', activation=None)
    x += conv_1
    x = tf.layers.conv2d(x, 3, 3, 1, padding='same', activation=None)

    return x

def AttnNet_0(low_input, is_training=True):

    num_blocks = 8

    x = low_input

    x = tf.layers.conv2d(x, 64, 3, 1, padding='same', activation=None)
    conv_1 = x

    for i in range(num_blocks):
        x = resBlock(x)

    x = tf.layers.conv2d(x, 64, 3, 1, padding='same', activation=None)
    x += conv_1
    x = tf.layers.conv2d(x, 3, 3, 1, padding='same', activation=None)

    return x


def AttnNet_1(ref_input, is_training=True):

    num_blocks = 8

    x = ref_input

    x = tf.layers.conv2d(x, 64, 3, 1, padding='same', activation=None)
    conv_1 = x

    for i in range(num_blocks):
        x = resBlock(x)

    x = tf.layers.conv2d(x, 64, 3, 1, padding='same', activation=None)
    x += conv_1
    x = tf.layers.conv2d(x, 3, 3, 1, padding='same', activation=None)

    return x


def AttnNet_2(illu_input, is_training=True):

    num_blocks = 8

    x = illu_input

    x = tf.layers.conv2d(x, 64, 3, 1, padding='same', activation=None)
    conv_1 = x

    for i in range(num_blocks):
        x = resBlock(x)

    x = tf.layers.conv2d(x, 64, 3, 1, padding='same', activation=None)
    x += conv_1
    x = tf.layers.conv2d(x, 3, 3, 1, padding='same', activation=None)

    return x


def AttnNet_1_2(ref_input, illu_input, is_training=True):

    output_1_2 = tf.multiply(ref_input, illu_input)    

    return output_1_2


def AttnNet_3(ref_input, illu_input, is_training=True):

    num_blocks = 8

    x = tf.concat([ref_input, illu_input], axis=3)

    x = tf.layers.conv2d(x, 64, 3, 1, padding='same', activation=None)
    conv_1 = x

    for i in range(num_blocks):
        x = resBlock(x)

    x = tf.layers.conv2d(x, 64, 3, 1, padding='same', activation=None)
    x += conv_1
    x = tf.layers.conv2d(x, 3, 3, 1, padding='same', activation=None)

    return x


def Endecoder(part1, part2, channel=64, kernel_size=3):


    input_im = tf.concat([part1, part2], axis=3)

    conv0 = tf.layers.conv2d(input_im, channel, kernel_size, padding='same', activation=None)
    conv1 = tf.layers.conv2d(conv0, channel, kernel_size, strides=2, padding='same', activation=tf.nn.relu)
    conv2 = tf.layers.conv2d(conv1, channel, kernel_size, strides=2, padding='same', activation=tf.nn.relu)
    conv3 = tf.layers.conv2d(conv2, channel, kernel_size, strides=2, padding='same', activation=tf.nn.relu)
        
    up1 = tf.image.resize_nearest_neighbor(conv3, (tf.shape(conv2)[1], tf.shape(conv2)[2]))
    deconv1 = tf.layers.conv2d(up1, channel, kernel_size, padding='same', activation=tf.nn.relu) + conv2
    up2 = tf.image.resize_nearest_neighbor(deconv1, (tf.shape(conv1)[1], tf.shape(conv1)[2]))
    deconv2= tf.layers.conv2d(up2, channel, kernel_size, padding='same', activation=tf.nn.relu) + conv1
    up3 = tf.image.resize_nearest_neighbor(deconv2, (tf.shape(conv0)[1], tf.shape(conv0)[2]))
    deconv3 = tf.layers.conv2d(up3, channel, kernel_size, padding='same', activation=tf.nn.relu) + conv0
        
    output = tf.layers.conv2d(deconv3, 3, kernel_size, 1, padding='same', activation=None)
     
    return output


def RBK(inp, dim):

    x = inp

    x = tf.layers.conv2d(x, dim, 1, 1, padding='same', activation=None)

    x0 = tf.layers.conv2d(inp, dim, 3, 1, padding='same', activation=tf.nn.relu)
    x1 = tf.layers.conv2d(x0, dim, 3, 1, padding='same', activation=None)
    out = x1 + x

    return out


def Structure_Generator(input1, input2, input3):

    dim = 64

    inp = tf.concat([input1, input2, input3], axis=3)   

    x0 = tf.layers.conv2d(inp, dim, 7, 1, padding='same', activation=tf.nn.relu)

    conv1 = tf.layers.conv2d(x0, dim, 4, 2, padding='same', activation=tf.nn.relu)
    x1 = tf.layers.conv2d(conv1, dim, 5, 1, padding='same', activation=tf.nn.relu)

    conv3 = tf.layers.conv2d(x1, dim, 4, 2, padding='same', activation=tf.nn.relu)
    x2 = tf.layers.conv2d(conv3, dim, 5, 1, padding='same', activation=tf.nn.relu)

    x3 = tf.layers.conv2d(x2, dim, 4, 2, padding='same', activation=tf.nn.relu)

    u11 = RBK(x3, dim)
    u11 = tf.image.resize_nearest_neighbor(u11, (tf.shape(u11)[1]*2, tf.shape(u11)[2]*2))
    u12 = tf.layers.conv2d(x2, dim, 5, 1, padding='same', activation=tf.nn.relu)
    u1 = u11 + u12

    u21 = RBK(u1, dim)
    u21 = tf.image.resize_nearest_neighbor(u21, (tf.shape(u21)[1]*2, tf.shape(u21)[2]*2))
    u22 = tf.layers.conv2d(x1, dim, 5, 1, padding='same', activation=tf.nn.relu)
    u2 = u21 + u22

    u31 = RBK(u2, dim)
    u31 = tf.image.resize_nearest_neighbor(u31, (tf.shape(u31)[1]*2, tf.shape(u31)[2]*2))
    u32 = tf.layers.conv2d(x0, dim, 5, 1, padding='same', activation=tf.nn.relu)
    u3 = u31 + u32

    u = tf.layers.conv2d(u3, 3, 3, 1, padding='same', activation=tf.nn.relu)

    return u


def Texture_Generator(input1, input2, input3, output_AF, channel=64, kernel_size=3):

    input_im = tf.concat([input1, input2, input3, output_AF], axis=3)

    conv0 = tf.layers.conv2d(input_im, channel, kernel_size, padding='same', activation=None)
    conv1 = tf.layers.conv2d(conv0, channel, kernel_size, strides=2, padding='same', activation=tf.nn.relu)
    conv2 = tf.layers.conv2d(conv1, channel, kernel_size, strides=2, padding='same', activation=tf.nn.relu)
    conv3 = tf.layers.conv2d(conv2, channel, kernel_size, strides=2, padding='same', activation=tf.nn.relu)
        
    up1 = tf.image.resize_nearest_neighbor(conv3, (tf.shape(conv2)[1], tf.shape(conv2)[2]))
    deconv1 = tf.layers.conv2d(up1, channel, kernel_size, padding='same', activation=tf.nn.relu) + conv2
    up2 = tf.image.resize_nearest_neighbor(deconv1, (tf.shape(conv1)[1], tf.shape(conv1)[2]))
    deconv2= tf.layers.conv2d(up2, channel, kernel_size, padding='same', activation=tf.nn.relu) + conv1
    up3 = tf.image.resize_nearest_neighbor(deconv2, (tf.shape(conv0)[1], tf.shape(conv0)[2]))
    deconv3 = tf.layers.conv2d(up3, channel, kernel_size, padding='same', activation=tf.nn.relu) + conv0
        
    output = tf.layers.conv2d(deconv3, 3, kernel_size, 1, padding='same', activation=None)
     
    return output



def Appearance_Flow(input1, input2, input3, channel=64, kernel_size=3):
    
    input_im = tf.concat([input1, input2, input3], axis=3)

    conv0 = tf.layers.conv2d(input_im, channel, kernel_size, padding='same', activation=None)
    conv1 = tf.layers.conv2d(conv0, channel, kernel_size, strides=2, padding='same', activation=tf.nn.relu)
    conv2 = tf.layers.conv2d(conv1, channel, kernel_size, strides=2, padding='same', activation=tf.nn.relu)
    conv3 = tf.layers.conv2d(conv2, channel, kernel_size, strides=2, padding='same', activation=tf.nn.relu)
        
    up1 = tf.image.resize_nearest_neighbor(conv3, (tf.shape(conv2)[1], tf.shape(conv2)[2]))
    deconv1 = tf.layers.conv2d(up1, channel, kernel_size, padding='same', activation=tf.nn.relu) + conv2
    up2 = tf.image.resize_nearest_neighbor(deconv1, (tf.shape(conv1)[1], tf.shape(conv1)[2]))
    deconv2= tf.layers.conv2d(up2, channel, kernel_size, padding='same', activation=tf.nn.relu) + conv1
    up3 = tf.image.resize_nearest_neighbor(deconv2, (tf.shape(conv0)[1], tf.shape(conv0)[2]))
    deconv3 = tf.layers.conv2d(up3, channel, kernel_size, padding='same', activation=tf.nn.relu) + conv0
        
    output = tf.layers.conv2d(deconv3, 3, kernel_size, 1, padding='same', activation=None)
     
    return output




def DownsampleRBK(input, dim_out):

    x1 = input

    x11 = tf.layers.conv2d(x1, dim_out, 3, 1, padding='same', activation=tf.nn.relu)
    x12 = tf.layers.conv2d(x11, dim_out, 3, 1, padding='same', activation=tf.nn.relu)
    x13 = tf.layers.average_pooling2d(x12, 2, [2,2])

    y11 = tf.layers.average_pooling2d(x1, 2, [2,2])
    y12 = tf.layers.conv2d(y11, dim_out, 1, 1, padding='same', activation=tf.nn.relu)

    return x13 + y12



def Discriminator(inp):

    n_layers = 3

    x = tf.layers.conv2d(inp, 64, 4, 2, padding='same', activation=tf.nn.relu)

    for i in range(n_layers-1):
        x = DownsampleRBK(x, 64)

    x = tf.layers.conv2d(x, 3, 1, activation=None)

    logits = tf.layers.dense(x, 1024)
    outputs = tf.nn.sigmoid(logits)

    return logits, outputs




def CrossTask(input1, input2, input3):

    output1 = Endecoder(input1, input2)
    output2 = Endecoder(input2, input3)
    output3 = Endecoder(input1, input3)

    output = tf.concat([output1, output2, output3], axis=3)

    output = tf.layers.conv2d(output, 3, 1, 1, padding='same', activation=None)

    return output


def Merge(input1, input2, input3):

    x = tf.concat([input1, input2, input3], axis=3)
    x = tf.layers.conv2d(x, 3, 1, 1, padding='same', activation=None)

    return x




def color_loss(image, label, len_reg=0):
    
    vec1 = tf.reshape(image, [-1, 3])
    vec2 = tf.reshape(label, [-1, 3])
    clip_value = 0.999999
    norm_vec1 = tf.nn.l2_normalize(vec1, 1)
    norm_vec2 = tf.nn.l2_normalize(vec2, 1)
    dot = tf.reduce_sum(norm_vec1*norm_vec2, 1)
    dot = tf.clip_by_value(dot, -clip_value, clip_value)
    angle = tf.acos(dot) * (180/math.pi)

    return tf.reduce_mean(angle)


def smoothness_loss(image):
    clip_low, clip_high = 0.000001, 0.999999
    image = tf.clip_by_value(image, clip_low, clip_high)
    image_h, image_w = tf.shape(image)[1], tf.shape(image)[2]
    tv_x = tf.reduce_mean((image[:, 1:, :, :]-image[:, :image_h-1, :, :])**2)
    tv_y = tf.reduce_mean((image[:, :, 1:, :]-image[:, :, :image_w-1, :])**2)
    total_loss = (tv_x + tv_y)/2
    '''
    log_image = tf.log(image)
    log_tv_x = tf.reduce_mean((log_image[:, 1:, :, :]-
                              log_image[:, :image_h-1, :, :])**1.2)
    log_tv_y = tf.reduce_mean((log_image[:, :, 1:, :]-
                               log_image[:, :, :image_w-1, :])**1.2)
    total_loss = tv_x / (log_tv_x + 1e-4) + tv_y / (log_tv_y + 1e-4)
    '''
    return total_loss


class lowlight_enhance(object):
    def __init__(self, sess):
        self.sess = sess
        self.base_lr = 0.001

        self.input_low = tf.placeholder(tf.float32, [None, None, None, 3], name='input_low')
        self.input_illu = tf.placeholder(tf.float32, [None, None, None, 3], name='input_illu')
        self.input_ref = tf.placeholder(tf.float32, [None, None, None, 3], name='input_ref')
        self.input_high = tf.placeholder(tf.float32, [None, None, None, 3], name='input_high')
        self.is_training = tf.placeholder(tf.bool)

        self.output_0 = AttnNet_0(self.input_low, self.is_training)
        self.output_1 = AttnNet_1(self.input_ref, self.is_training)
        self.output_2 = AttnNet_2(self.input_illu, self.is_training)
        self.output_1_2 = AttnNet_1_2(self.output_1, self.output_2)
        self.output_3 = AttnNet_3(self.input_ref, self.input_illu, self.is_training)

        #self.output = CrossTask(self.output_0, self.output_1_2, self.output_3)
        #self.output = Merge(self.output_0, self.output_1_2, self.output_3)
        self.output_SG = Structure_Generator(self.output_0, self.output_1_2, self.output_3)
        self.output_AF = Appearance_Flow(self.output_0, self.output_SG, self.output_3)
        self.output = Texture_Generator(self.output_0, self.output_SG, self.output_3, self.output_AF)

        self.logits_real, self.output_real = Discriminator(self.input_high)
        self.logits_fake, self.output_fake = Discriminator(self.output)

        self.loss_0 = tf.reduce_mean(tf.abs(self.output_0 - self.input_high))
        self.loss_1 = smoothness_loss(self.output_1)
        self.loss_2 = smoothness_loss(self.output_2)
        self.loss_1_2 = tf.reduce_mean(tf.abs(self.output_1_2 - self.input_high))
        self.loss_3 = tf.reduce_mean(tf.abs(self.output_3 - self.input_high))
        self.loss_las = tf.reduce_mean(tf.abs((self.output - self.input_high)))
        self.loss_SG = tf.reduce_mean(tf.abs(self.output_SG - self.input_high))

        self.loss_C_AF = color_loss(self.output_AF, self.input_high)
        self.loss_C_TG = color_loss(self.output, self.input_high)
        #self.dis_loss = tf.losses.mean_squared_error(self.output_dis_real, self.output_dis_fake)
        self.loss_adv_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.logits_real, 
                                            labels=tf.ones_like(self.output_real)*0.9))
        self.loss_adv_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.logits_fake,
                                            labels=tf.zeros_like(self.output_fake)))
        self.loss_adv = self.loss_adv_real + self.loss_adv_fake
        self.loss_gen = self.loss_las + 0.01 * self.loss_0 + 0.01 * self.loss_1_2 + 0.01 * self.loss_3
        self.loss_S = self.loss_1 + self.loss_2
        self.loss_C = self.loss_C_AF + self.loss_C_TG

        self.loss = self.loss_gen + 2*self.loss_C + 0.5*self.loss_adv + 0.5*self.loss_S

        self.global_step = tf.Variable(0, trainable = False)
        self.lr = tf.train.exponential_decay(self.base_lr, self.global_step, 100, 0.96)
        optimizer = tf.train.AdamOptimizer(self.lr, name='AdamOptimizer')
        self.train_op = optimizer.minimize(self.loss, global_step=self.global_step)

        self.sess.run(tf.global_variables_initializer())
        self.saver = tf.train.Saver()
        print("[*] Initialize model successfully...")

    def evaluate(self, epoch_num, eval_low_data, eval_illu_data, eval_ref_data, sample_dir):

        print("[*] Evaluating for epoch %d..." % (epoch_num))


        for idx in range(len(eval_low_data)):
            input_low_eval = np.expand_dims(eval_low_data[idx], axis=0)
            input_illu_eval = np.expand_dims(eval_illu_data[idx], axis=0)
            input_ref_eval = np.expand_dims(eval_ref_data[idx], axis=0)

            result = self.sess.run(self.output, feed_dict={self.input_low: input_low_eval, self.input_illu: input_illu_eval, 
                                    self.input_ref: input_ref_eval, self.is_training: False})
            save_images(os.path.join(sample_dir, 'eval_%d_%d.png' % (idx + 1, epoch_num)), input_low_eval, result)


    def train(self, train_low_data, train_illu_data, train_ref_data, train_high_data, eval_low_data, eval_illu_data, eval_ref_data, 
                batch_size, patch_size, epoch, sample_dir, ckpt_dir, eval_every_epoch):

        assert len(train_low_data) == len(train_high_data)
        numBatch = len(train_low_data) // int(batch_size)

        load_model_status, global_step = self.load(self.saver, ckpt_dir)
        if load_model_status:
            iter_num = global_step
            start_epoch = global_step // numBatch
            start_step = global_step % numBatch
            print("[*] Model restore success!")
        else:
            iter_num = 0
            start_epoch = 0
            start_step = 0
            print("[*] Not find pretrained model!")

        print("[*] Start training with start epoch %d start iter %d : " % (start_epoch, iter_num))

        start_time = time.time()
        image_id = 0

        for epoch in range(start_epoch, epoch):
            for batch_id in range(start_step, numBatch):
                # generate data for a batch
                batch_input_low = np.zeros((batch_size, patch_size, patch_size, 3), dtype="float32")
                batch_input_illu = np.zeros((batch_size, patch_size, patch_size, 3), dtype="float32")
                batch_input_ref = np.zeros((batch_size, patch_size, patch_size, 3), dtype="float32")
                batch_input_high = np.zeros((batch_size, patch_size, patch_size, 3), dtype="float32")
                for patch_id in range(batch_size):
                    h, w, _ = train_low_data[image_id].shape
                    x = random.randint(0, h - patch_size)
                    y = random.randint(0, w - patch_size)
            
                    rand_mode = random.randint(0, 7)
                    batch_input_low[patch_id, :, :, :] = data_augmentation(train_low_data[image_id][x : x+patch_size, y : y+patch_size, :], rand_mode)
                    batch_input_illu[patch_id, :, :, :] = data_augmentation(train_illu_data[image_id][x : x+patch_size, y : y+patch_size, :], rand_mode)
                    batch_input_ref[patch_id, :, :, :] = data_augmentation(train_ref_data[image_id][x : x+patch_size, y : y+patch_size, :], rand_mode)
                    batch_input_high[patch_id, :, :, :] = data_augmentation(train_high_data[image_id][x : x+patch_size, y : y+patch_size, :], rand_mode)
                    
                    image_id = (image_id + 1) % len(train_low_data)
                    if image_id == 0:
                        tmp = list(zip(train_low_data, train_illu_data, train_ref_data, train_high_data))
                        random.shuffle(list(tmp))
                        train_low_data, train_illu_data, train_ref_data, train_high_data  = zip(*tmp)

                # train
                _, loss = self.sess.run([self.train_op, self.loss], feed_dict={self.input_low: batch_input_low, self.input_illu: batch_input_illu, 
                                            self.input_ref: batch_input_ref, self.input_high: batch_input_high, self.is_training: True})

                print("Epoch: [%2d] [%4d/%4d] time: %4.4f, loss: %.6f" \
                      % (epoch + 1, batch_id + 1, numBatch, time.time() - start_time, loss))
                iter_num += 1

            # evalutate the model and save a checkpoint file for it
            if (epoch + 1) % eval_every_epoch == 0:
                self.evaluate(epoch + 1, eval_low_data, eval_illu_data, eval_ref_data, sample_dir=sample_dir)
                self.save(self.saver, iter_num, ckpt_dir, "Retinex+AttenNet")

        print("[*] Finish training")

    def save(self, saver, iter_num, ckpt_dir, model_name):
        if not os.path.exists(ckpt_dir):
            os.makedirs(ckpt_dir)
        print("[*] Saving model %s" % model_name)
        saver.save(self.sess, \
                   os.path.join(ckpt_dir, model_name), \
                   global_step=iter_num)

    def load(self, saver, ckpt_dir):
        ckpt = tf.train.get_checkpoint_state(ckpt_dir)
        if ckpt and ckpt.model_checkpoint_path:
            full_path = tf.train.latest_checkpoint(ckpt_dir)
            try:
                global_step = int(full_path.split('/')[-1].split('-')[-1])
            except ValueError:
                global_step = None
            saver.restore(self.sess, full_path)
            return True, global_step
        else:
            print("[*] Failed to load model from %s" % ckpt_dir)
            return False, 0

    def test(self, test_low_data, test_illu_data, test_ref_data, test_high_data, test_low_data_names, test_illu_data_names, 
                test_ref_data_names, save_dir):
        
        tf.global_variables_initializer().run()

        print("[*] Reading checkpoint...")
        load_model_status, _ = self.load(self.saver, './model/')
        if load_model_status:
            print("[*] Load weights successfully...")
        
        print("[*] Testing...")
        total_run_time = 0.0
        for idx in range(len(test_low_data)):
            print(test_low_data_names[idx])
            print(test_illu_data_names[idx])
            print(test_ref_data_names[idx])
            [_, name] = os.path.split(test_low_data_names[idx])
            suffix = name[name.find('.') + 1:]
            name = name[:name.find('.')]

            input_low_test = np.expand_dims(test_low_data[idx], axis=0)
            input_illu_test = np.expand_dims(test_illu_data[idx], axis=0)
            input_ref_test = np.expand_dims(test_ref_data[idx], axis=0)
            
            start_time = time.time()
            result = self.sess.run(self.output, feed_dict = {self.input_low: input_low_test, self.input_illu: input_illu_test, 
                                    self.input_ref: input_ref_test, self.is_training: False})
            total_run_time += time.time() - start_time
            save_images(os.path.join(save_dir, name + "_MCA."   + suffix), result)

        ave_run_time = total_run_time / float(len(test_low_data))
        print("[*] Average run time: %.4f" % ave_run_time)
