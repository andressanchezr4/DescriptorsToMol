#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: andres
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Conv3D, Conv3DTranspose
from tensorflow.keras.layers import BatchNormalization, LeakyReLU
from tensorflow.keras.losses import BinaryFocalCrossentropy

class VoxelEncoder(keras.Model):
    def __init__(self, input_shape):
        super(VoxelEncoder, self).__init__()

        self.conv2 = Conv3D(32, kernel_size=3, strides=2, padding="same", use_bias=False)
        self.bn2 = BatchNormalization()
        self.act2 = LeakyReLU()

        self.conv3 = Conv3D(64, kernel_size=3, strides=1, padding="same", use_bias=False)
        self.bn3 = BatchNormalization()
        self.act3 = LeakyReLU()

        self.conv4 = Conv3D(128, kernel_size=3, strides=2, padding="same", use_bias=False)
        self.bn4 = BatchNormalization()
        self.act4 = LeakyReLU()

        self.conv5 = Conv3D(512, kernel_size=3, strides=2, padding="same", use_bias=False)
        self.bn5 = BatchNormalization()
        self.act5 = LeakyReLU()

        self.conv6 = Conv3D(1024, kernel_size=3, strides=1, padding="same", use_bias=False)
        self.bn6 = BatchNormalization()
        self.act6 = LeakyReLU()
        
        # silently "auto build"
        dummy_input = tf.random.normal(input_shape)
        _ = self(dummy_input)
        
    def call(self, inputs):

        x = self.conv2(inputs)
        x = self.bn2(x)
        x = self.act2(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = self.act3(x)

        x = self.conv4(x)
        x = self.bn4(x)
        x = self.act4(x)

        x = self.conv5(x)
        x = self.bn5(x)
        x = self.act5(x)

        x = self.conv6(x)
        x = self.bn6(x)
        x = self.act6(x)

        return x

class VoxelDecoder(keras.Model):
    def __init__(self, batch_size):
        super(VoxelDecoder, self).__init__()

        self.deconv1 = Conv3DTranspose(512, kernel_size=3, strides=2, padding="same", use_bias=False)
        self.bn1 = BatchNormalization()
        self.act1 = LeakyReLU()

        self.deconv2 = Conv3DTranspose(256, kernel_size=3, strides=2, padding="same", use_bias=False)
        self.bn2 = BatchNormalization()
        self.act2 = LeakyReLU()

        self.deconv3 = Conv3DTranspose(128, kernel_size=3, strides=1, padding="same", use_bias=False)
        self.bn3 = BatchNormalization()
        self.act3 = LeakyReLU()

        self.deconv4 = Conv3DTranspose(64, kernel_size=3, strides=1, padding="same", use_bias=False)
        self.bn4 = BatchNormalization()
        self.act4 = LeakyReLU()

        self.deconv5 = Conv3DTranspose(32, kernel_size=3, strides=2, padding="same", use_bias=False)
        self.bn5 = BatchNormalization()
        self.act5 = LeakyReLU()

        self.final_layer = Conv3DTranspose(8, kernel_size=3, strides=1, padding="same", activation='sigmoid', name='FINAL_LAYER')
        
        # silently "auto build"
        dummy_input = tf.random.normal((batch_size, 3, 3, 3, 1024))
        _ = self(dummy_input)
        
    def call(self, inputs):
        x = self.deconv1(inputs)
        x = self.bn1(x)
        x = self.act1(x)

        x = self.deconv2(x)
        x = self.bn2(x)
        x = self.act2(x)

        x = self.deconv3(x)
        x = self.bn3(x)
        x = self.act3(x)

        x = self.deconv4(x)
        x = self.bn4(x)
        x = self.act4(x)

        x = self.deconv5(x)
        x = self.bn5(x)
        x = self.act5(x)

        x = self.final_layer(x)
        return x

class convAE(tf.keras.Model):
    def __init__(self, encoder_desc, decoder_atom, **kwargs):
        super(convAE, self).__init__(**kwargs)
        self.encoder_desc = encoder_desc
        self.decoder_atom = decoder_atom
        self.atom_loss_tracker = keras.metrics.Mean(name="atom_loss")
    
    @property
    def metrics(self):
        return [self.atom_loss_tracker]
    
    def call(self, inputs):
        voxel_desc, voxel_atoms = inputs
        latspace = self.encoder_desc(voxel_desc)
        predicted_voxel_atoms = self.decoder_atom(latspace)
        
        return predicted_voxel_atoms 
    
    def train_step(self, data):
        voxel_desc, voxel_atoms = data
        
        with tf.GradientTape() as tape:
            latent_space = self.encoder_desc(voxel_desc)
            predicted_voxel_atoms = self.decoder_atom(latent_space)
        
            atom_loss = BinaryFocalCrossentropy()(voxel_atoms, predicted_voxel_atoms)
            
        grads = tape.gradient(atom_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        
        self.atom_loss_tracker.update_state(atom_loss)
        
        return {
            "atom_loss": self.atom_loss_tracker.result(),
               }
    
    def test_step(self, data):

        voxel_desc, voxel_atoms = data
        
        latent_space= self.encoder_desc(voxel_desc)
        predicted_voxel_atoms = self.decoder_atom(latent_space)
    
        val_atom_loss = BinaryFocalCrossentropy()(voxel_atoms, predicted_voxel_atoms)
        # print(val_atom_loss) # to check value
        return {
                "atom_loss": val_atom_loss,
               }
        
