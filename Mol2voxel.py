#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@andressanchezr4
"""

import time
start = time.time()
import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
import math
import numpy as np
import os
import pickle
import random
from natsort import natsorted
import os
import pickle
import random
from sklearn.metrics import mean_squared_error
from sklearn.metrics import auc
import sys

from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.rdmolfiles import *
from rdkit.Chem.rdMolTransforms import ComputeCentroid
from rdkit.Chem.rdchem import *
from rdkit.Chem.Draw import IPythonConsole
from rdkit.Chem import Draw

import moleculekit
from moleculekit.home import home
from moleculekit.smallmol.smallmol import SmallMol
from moleculekit.tools.voxeldescriptors import _getChannelRadii, rotateCoordinates
from moleculekit.periodictable import periodictable
from moleculekit.util import *
from moleculekit.util import rotationMatrix
import logging
logger = logging.getLogger("moleculekit")
logger.setLevel(logging.ERROR)
from rdkit.Geometry import Point3D

from multiprocessing import Pool
import multiprocessing as mp
from rdkit.Chem import rdmolfiles

class RdkitMols2Voxels(object):
    def __init__(self, mol_list, box_size, 
                 batch_size, path2save,
                 atoms = ['C', 'H', 'N', 'O', 'S', 'Cl', 'Br', 'F']
                 ):
        
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
        
        self.batch_size = batch_size
        self.mol_list = mol_list
        self.mol_batches = [self.mol_list[i:i + self.batch_size] for i in range(0, len(self.mol_list), self.batch_size)]
        self.index_dict = {tuple(sub): i for i, sub in enumerate(self.mol_batches)}

        self.box_size = box_size
        self.path2save = path2save
        if not os.path.exists(self.path2save):
            os.mkdir(self.path2save)
        self.dict_char_set = dict((c, i) for i, c in enumerate(atoms))
        self.n_atoms = len(atoms)
        
        self.failed2voxelize = []
        self.failed2conformer = []
        
    def Make3D(self):
        mol_list_fix = []
        for molecule in self.mol_list:
            try:
                molecule = Chem.AddHs(molecule)
                AllChem.EmbedMolecule(molecule)
                AllChem.UFFOptimizeMolecule(molecule)
                mol_list_fix.append(molecule)
            except:
                print('Failed to 3Dize, check the attribute "failed2conformer"')
                self.failed2conformer.append([molecule])
        
        self.mol_batches = [mol_list_fix[i:i + self.batch_size] for i in range(0, len(mol_list_fix), self.batch_size)]
        self.index_dict = {tuple(sub): i for i, sub in enumerate(self.mol_batches)}
        
    def _bytes_feature(self, value):
        if isinstance(value, type(tf.constant(0))): 
            value = value.numpy() 
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

    def serialize_array(self, array):
      array = tf.io.serialize_tensor(array)
      return array

    def smi2vox(self, batch_mol):
        all_voxel_at = []
        all_voxel_desc = []
        for pos, mol_rdkit in enumerate(batch_mol):
            
            mol_voxel = np.zeros((self.box_size**3, 8)) #at
            mol = SmallMol(mol_rdkit)
            
            mol_atoms = mol.get('element').tolist()
            mol_coord = np.squeeze(mol.get('coords'))
            mol_center = mol.getCenter()
            
            voxel_desc, voxel_centers, N = moleculekit.tools.voxeldescriptors.getVoxelDescriptors(mol, center=mol_center, boxsize=[self.box_size,
                                                                                                                                   self.box_size, 
                                                                                                                                   self.box_size])
            all_voxel_desc.append(voxel_desc.reshape((self.box_size, self.box_size, self.box_size, self.n_atoms)))
            
            for mc, a in zip(mol_coord, mol_atoms):
                column = self.dict_char_set.get(a, '')
                if column == '':
                    continue
                
                shortest = []
                for vc in voxel_centers:
                    shortest.append(np.linalg.norm(mc - vc))
                    
                shortest_index = shortest.index(min(shortest))

                if mol_voxel[shortest_index, column] != 0:
                    print('Collisioned atom')
                    self.failed2voxelize.append(mol_rdkit)
                    
                mol_voxel[shortest_index, column] = 1
                
            all_voxel_at.append(mol_voxel.reshape((self.box_size, self.box_size, self.box_size, self.n_atoms)))
        
        all_voxel_at2save = np.array(all_voxel_at, dtype ='float16')
        all_voxel_desc2save = np.array(all_voxel_desc, dtype = 'float16')
        
        n_batch = self.index_dict[tuple(batch_mol)]
        path2savesinglebatch = f'{self.path2save}batch_{n_batch}.tfrecord'
        with tf.io.TFRecordWriter(path2savesinglebatch) as writer:
            serialized_voxel_atom = self.serialize_array(all_voxel_at2save)
            serialized_voxel_descriptor = self.serialize_array(all_voxel_desc2save)
            feature = {'voxel_descriptor': self._bytes_feature(serialized_voxel_descriptor),
                       'voxel_atom': self._bytes_feature(serialized_voxel_atom)}
            
            example_message = tf.train.Example(features=tf.train.Features(feature=feature))
            writer.write(example_message.SerializeToString())
        
        print(f'{n_batch} batch generated')
    
    def transform(self, n_cpus = os.cpu_count()):
       
        with mp.Pool(n_cpus) as pool:
            pool.map(self.smi2vox, self.mol_batches)

class CheckVoxel(object):
    def __init__(self, path2batchtensor, shape, 
                 real_char_set = ['C', 'H', 'N', 'O', 'S', 'Cl', 'Br', 'F'],
                 voxel_size = 1.0):
        
        self.real_char_set = real_char_set
        self.voxel_size = voxel_size
        
        self.feature_description = {
            'voxel_atom': tf.io.FixedLenFeature([], tf.string),
            'voxel_descriptor': tf.io.FixedLenFeature([], tf.string),
        }
        self.batch_tensor = self.load_voxel_atom_tensor(path2batchtensor, shape)

    
    def parse_function(self, example_proto):
        return tf.io.parse_single_example(example_proto, self.feature_description)
    
    def load_voxel_atom_tensor(self, tfrecord_path, shape):
        dataset = tf.data.TFRecordDataset([tfrecord_path])
        dataset = dataset.map(self.parse_function)
    
        for parsed_record in dataset:
            voxel_atom = tf.io.parse_tensor(parsed_record['voxel_atom'], out_type=tf.float16)
            voxel_atom = tf.reshape(voxel_atom, shape)
            return voxel_atom.numpy()
        
    def check_index(self, path2sdf, index_mol):
        tensor = self.batch_tensor[index_mol]
        mol = Chem.RWMol()
        atom_coords = []
    
        for x in range(tensor.shape[0]):
            for y in range(tensor.shape[1]):
                for z in range(tensor.shape[2]):
                    for c in range(tensor.shape[3]):
                        if tensor[x, y, z, c] > 0.5:
                            atom_symbol = self.real_char_set[c]
                            idx = mol.AddAtom(Chem.Atom(atom_symbol))
                            coord = Point3D(x * self.voxel_size, y * self.voxel_size, z * self.voxel_size)
                            atom_coords.append((idx, coord))
    
        if not atom_coords:
            print("No atoms found in the tensor")
            return None
    
        conf = Chem.Conformer(len(atom_coords))
        for idx, coord in atom_coords:
            conf.SetAtomPosition(idx, coord)
        mol.AddConformer(conf)
        
        writer = rdmolfiles.SDWriter(path2sdf)
        writer.write(mol)
        writer.close()
        
