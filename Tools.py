#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: andres
"""

import os
import tensorflow as tf
import numpy as np
from sklearn.metrics import mean_squared_error

class VoxelTFRecordLoader(object):
    def __init__(self, directory_path, shape, validation_size=0.1):
        self.directory_path = directory_path
        self.validation_size = int(len(os.listdir(directory_path))*validation_size)
        self.shape = shape
        
        self.train_dataset = None
        self.validation_dataset = None
        
    def parse_example(self, element):
        parse_dic = {
            'voxel_descriptor': tf.io.FixedLenFeature([], tf.string),
            'voxel_atom': tf.io.FixedLenFeature([], tf.string),
                    }
        
        example = tf.io.parse_single_example(element, parse_dic)
        
        feature_d = tf.io.parse_tensor(example['voxel_descriptor'], out_type=tf.float16)
        feature_a = tf.io.parse_tensor(example['voxel_atom'], out_type=tf.float16)
        
        feature_d = tf.reshape(feature_d, self.shape) 
        feature_a = tf.reshape(feature_a, self.shape)
        return feature_d, feature_a

    def load_dataset(self):
        file_paths = [os.path.join(self.directory_path, fname) for fname in os.listdir(self.directory_path) if fname.endswith('.tfrecord')]
        dataset = tf.data.TFRecordDataset(file_paths)
        dataset = dataset.map(self.parse_example, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        return dataset

    def split_dataset(self):
        self.validation_dataset = self.dataset.take(self.validation_size)
        self.train_dataset = self.dataset.skip(self.validation_size)

    def get_dataset_generators(self):
        self.dataset = self.load_dataset()
        self.split_dataset()
        return self.train_dataset, self.validation_dataset

def check_predicted_voxels(my_model, dataset, n_atoms = 8, box_size = 24):
    t_coincidencias = []
    t_rmse = []
    
    for y, (desc, atom) in enumerate(dataset):
        pred = my_model.predict([desc, atom])
        atom = atom.numpy()
        for at, p in zip(atom, pred):
    
            real_indexes = np.argwhere(at).tolist()
            n_real_atoms = [len(np.argwhere(at[:,:,:,i]).tolist()) for i in range(n_atoms)]
 
            ### Top probabilidades de toda la molecula dados el numero total de atomos ####
            n = len(real_indexes) 
            pred_coordinates = []
            for element, n in enumerate(n_real_atoms):
                if n == 0:
                    continue
                indices = np.argpartition(p[:,:,:,element], -n, axis=None)[-n:]
                indices_4d = np.array(np.unravel_index(indices, (box_size,box_size,box_size))).tolist()
                indices_4d.append([element]*n)
                pred_coordinates.append(np.array(indices_4d).transpose())
            pred_coordinates = np.vstack(pred_coordinates)
            
            ### Atom position RMSE
            rmse = []
            for r in real_indexes:
                shortest = []
                for pr in pred_coordinates:
                    shortest.append(np.linalg.norm(r-pr))
                shortest_dist = pred_coordinates[shortest.index(min(shortest))]
                rmse.append(mean_squared_error(r, shortest_dist))
            total_rmse = sum(rmse)
            t_rmse.append(total_rmse)
            
            ### Exact coincidences
            matches1 = 0
            for pred_coord in pred_coordinates:
                if at[pred_coord[0], pred_coord[1], pred_coord[2], pred_coord[3]] == 1:
                    matches1 += 1
            total_coincidencias = matches1/sum(n_real_atoms)
            t_coincidencias.append(total_coincidencias)
        
        print(f'batch {y}: rmse {total_rmse:.3f} | %exact match: {total_coincidencias:.2f} | n match: {matches1}/{sum(n_real_atoms)}')


