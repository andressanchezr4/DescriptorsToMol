"""
@author: andres
"""

import sys
sys.path.append('/path/to/this/repo')

import os
import pandas as pd
from rdkit import Chem
import tensorflow as tf

from Tools import check_predicted_voxels, VoxelTFRecordLoader
from Model_Autoencoder import VoxelEncoder, VoxelDecoder, convAE
from Mol2voxel import RdkitMols2Voxels, CheckVoxel

##################################
#### DATA LOADING & VOXEL GEN ####
##################################
os.environ["CUDA_VISIBLE_DEVICES"] = "-1" 
batch_size = 64
box_size = 24
n_atoms = 8 # ['C', 'H', 'N', 'O', 'S', 'Cl', 'Br', 'F']
path2savetfrvoxels = './voxel_batches/'
batch_shape = (batch_size, box_size, box_size, box_size, n_atoms)

df = pd.read_csv('./data/smiles_example.csv')
df['MOL'] = df.MOLBLOCK.apply(Chem.MolFromMolBlock)

make_voxels = RdkitMols2Voxels(df.MOL.tolist(), box_size, batch_size, path2savetfrvoxels)
# make_vox.Make3D()
make_voxels.transform()

# Check how the voxelization went
tfrecord_path = path2savetfrvoxels + 'batch_1.tfrecord'
path2savesdf = path2savetfrvoxels + 'check_mol_new.sdf'

checking = CheckVoxel(tfrecord_path, batch_shape)
checking.check_index(path2savesdf, 10)

##################################
##### PREPARE MODEL TRAINING #####
##################################
os.environ["CUDA_VISIBLE_DEVICES"] = "1" 
tfr = VoxelTFRecordLoader(path2savetfrvoxels, batch_shape)
train_dataset, val_dataset = tfr.get_dataset_generators()
steps_per_epoch = len(os.listdir(path2savetfrvoxels))

lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=1e-3,
    decay_steps=300,
    decay_rate=0.96,
    staircase=True
)
optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)

encoder_desc = VoxelEncoder(batch_shape)
decoder_atom = VoxelDecoder(batch_size)

autoencoder = convAE(encoder_desc, decoder_atom)
autoencoder.compile(optimizer=optimizer, run_eagerly=True)
autoencoder.fit(train_dataset, epochs=100, validation_data=val_dataset)

# Check the results
check_predicted_voxels(autoencoder, val_dataset, n_atoms, box_size)
