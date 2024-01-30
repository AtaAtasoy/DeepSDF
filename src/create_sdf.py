import sys 
sys.path.append('..')
import warnings
warnings.filterwarnings('ignore')
from backend.sampler import MeshSampler
import trimesh
import numpy as np
import pandas as pd
import shutil
import os
from tqdm import tqdm_notebook as tqdm
import pybullet as pb

def rotate_pointcloud(pointcloud_A, rpy_BA=[np.pi / 2, 0, 0]):
    """
    The default rotation reflects the rotation used for the object during data collection.
    This calculates P_b, where P_b = R_b/a * P_a.
    R_b/a is rotation matrix of a wrt b frame.
    """
    # Rotate object
    rot_Q = pb.getQuaternionFromEuler(rpy_BA)
    rot_M = np.array(pb.getMatrixFromQuaternion(rot_Q)).reshape(3, 3)
    pointcloud_B = np.einsum('ij,kj->ki', rot_M, pointcloud_A)

    return pointcloud_B


def shapenet_rotate(mesh_original):
    '''In Shapenet, the front is the -Z axis with +Y still being the up axis. This function rotates the object to align with the canonical reference frame.
    Args:
        mesh_original: trimesh.Trimesh(), mesh from ShapeNet
    Returns:
        mesh: trimesh.Trimesh(), rotate mesh so that the front is the +X axis and +Y is the up axis.
    '''
    verts_original = np.array(mesh_original.vertices)

    rot_M = pb.getMatrixFromQuaternion(pb.getQuaternionFromEuler([np.pi/2, 0, -np.pi/2]))
    rot_M = np.array(rot_M).reshape(3, 3)
    verts = rotate_pointcloud(verts_original, [np.pi/2, 0, -np.pi/2])

    mesh = trimesh.Trimesh(vertices=verts, faces=mesh_original.faces)

    return mesh

if __name__ == "main":
    DATA_DIR = '../../data/'
    SAVE_DIR = '../../data/'

    categories = os.listdir(DATA_DIR)

    #categories = [category for category in categories if os.path.isdir(os.path.join(DATA_DIR, category))]
    #categories = ['vase', 'bed', 'tv', 'computer', 'lamp', 'pottedplant', 'couch', 'chair', 'desk', 'table']
    #categories = ['bed', 'couch', 'vase', 'pottedplant']
    categories = ['vase']

    for category in categories:
        print(f'Processing {category}')
        category_data_dir = os.path.join(DATA_DIR, category)
        category_save_dir = os.path.join(SAVE_DIR, category)
        
        shapes = [shape for shape in os.listdir(category_data_dir) if os.path.isdir(os.path.join(category_data_dir, shape))]
        for shape in shapes:
            shape_data_dir = os.path.join(category_data_dir, shape)
            shape_save_dir = os.path.join(category_save_dir, shape)
            
            mesh_path = os.path.join(shape_data_dir, 'mesh_simplified.obj')
            print(f'Mesh at {mesh_path}')
            
            mesh = trimesh.load(mesh_path)
            
            rotated_mesh = shapenet_rotate(mesh)
            
            sampler = MeshSampler(mesh)
            
            sampler.compute_visible_faces()
            sampler.sample_points(n_points=100000)
            
            correct_mesh_points, correct_sdf, correct_normals = sampler.compute_sdf(sigma=0.0025)
            
            # Create positive and negative samples according to sdf values with their corresponding points
            # Positive samples are the points that are inside the mesh
            # Negative samples are the points that are outside the mesh
            positive_indices = correct_sdf > 0
            positive_samples = np.concatenate([correct_mesh_points[positive_indices], correct_sdf[positive_indices].reshape(-1, 1)], axis=1)
            
            negative_indices = correct_sdf < 0
            negative_samples = np.concatenate([correct_mesh_points[negative_indices], correct_sdf[negative_indices].reshape(-1, 1)], axis=1)
            
            #print(f'Positive samples shape: {positive_samples.shape}')
            #print(f'Negative samples shape: {negative_samples.shape}')
            
            np.savez(os.path.join(shape_save_dir, 'sdf.npz'), pos=positive_samples, neg=negative_samples)