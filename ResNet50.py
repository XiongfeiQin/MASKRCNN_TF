from Network import Network
import tensorflow as tf
import numpy as np
from functools import reduce

class ResNet50(Network):
    def __init__(self, Network_path=None, trainable=True, dropout=0.5):
        Network.__init__(Network_path, trainable, dropout)
        print('ResNet50')

    def build(self, image, batch_size, train_mode=True):
        x = self.conv_layer(image, 3, 64, name='conv1', conv_size=7, stride=2)
        x = self.batch_norm_layer(x,'bn_conv1')
        x = tf.nn.relu(x)
        C1 = x = self.max_pool(x,name='mp_conv1')
        # Stage 2
        x = self.bottleneck1(x, 64, 64, 64, 256, 'bottle21', stride=1)
        x = self.bottleneck0(x, 256, 64, 64, 256, 'bottle22')
        C2 = x = self.bottleneck0(x, 256, 64, 64, 256, 'bottle23')
        # Stage 3
        x = self.bottleneck1(x, 256, 128, 128, 512, 'bottle31')
        x = self.bottleneck0(x, 512, 128, 128, 512, 'bottle32')
        x = self.bottleneck0(x, 512, 128, 128, 512, 'bottle33')
        C3 = x = self.bottleneck0(x, 512, 128, 128, 512, 'bottle34')
        # Stage 4
        x = conv_block(x, 3, [256, 256, 1024], stage=4, block='a')
        block_count = {"resnet50": 5, "resnet101": 22}[architecture]
        for i in range(block_count):
            x = identity_block(x, 3, [256, 256, 1024], stage=4, block=chr(98 + i))
        C4 = x
        pass


    def bottleneck0(self, inputs, in_channels, m1, m2, out_channels, bottle_name):
        conv1 = self.conv_layer(inputs,in_channels, m1, bottle_name+'_0conv1',conv_size=1)
        bn1 = self.batch_norm_layer(conv1, bottle_name + '_0bn1')
        r1 = tf.nn.relu(bn1)
        conv2 = self.conv_layer(r1, m1, m2, bottle_name + '_0conv2')
        bn2 = self.batch_norm_layer(conv2, bottle_name + '_0bn2')
        r2 = tf.nn.relu(bn2)
        conv3 = self.conv_layer(r2, m2, out_channels, bottle_name + '_0conv3',conv_size=1)
        bn2 = self.batch_norm_layer(conv3, bottle_name + '_0bn3')
        out = tf.nn.relu(tf.add(bn2, inputs))
        return out

    def bottleneck1(self, inputs,in_channels, m1,m2, out_channels, bottle_name, stride=2):
        conv1 = self.conv_layer(inputs,in_channels, m1, bottle_name+'_1conv1', conv_size=1, stride=stride)
        bn1 = self.batch_norm_layer(conv1, bottle_name + '_1bn1')
        r1 = tf.nn.relu(bn1)
        conv2 = self.conv_layer(r1, m1, m2, bottle_name + '_1conv2')
        bn2 = self.batch_norm_layer(conv2, bottle_name + '_1bn2')
        r2 = tf.nn.relu(bn2)
        conv3 = self.conv_layer(r2, m2, out_channels, bottle_name + '_1conv3',conv_size=1)
        bn2 = self.batch_norm_layer(conv3, bottle_name + '_1bn3')
        # shortcut
        convs = self.conv_layer(inputs,in_channels,out_channels,bottle_name + '_1convs', conv_size=1, stride=stride)
        bns = self.batch_norm_layer(convs, bottle_name + '_1bns')

        out = tf.nn.relu(tf.add(bn2, bns))
        return out


