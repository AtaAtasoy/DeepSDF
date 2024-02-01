import trimesh
import numpy as np
import pandas as pd
import shutil
import os
import argparse
import pybullet as pb
import random
from mesh_to_sdf import sample_sdf_near_surface


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

def scale_and_translate_to_unit_sphere(mesh_original):
    # Find the bounding box of the mesh
    min_coords, max_coords = mesh_original.bounds

    # Calculate the center and size of the bounding box
    center = (min_coords + max_coords) / 2
    size = max(max_coords - min_coords)

    # Scale and translate mesh to fit within the unit sphere
    scaled_mesh = mesh_original.apply_scale(1 / size)
    translated_and_scaled_mesh = scaled_mesh.apply_translation(-center / size / 1.03)
    
    return translated_and_scaled_mesh



def random_rotate(mesh_original):
    R = trimesh.transformations.random_rotation_matrix()
    mesh_rotated = mesh_original.apply_transform(R)
    return mesh_rotated


def random_nonuniform_scale(mesh_original):
    scale_mat = np.eye(4)
    scale_mat[0, 0] = random.uniform(0.7, 1.5)
    scale_mat[1, 1] = random.uniform(0.7, 1.5)
    scale_mat[2, 2] = random.uniform(0.7, 1.5)
    scaled_mesh = mesh_original.apply_transform(scale_mat)
    return scaled_mesh



if __name__ =="__main__":
    parser = argparse.ArgumentParser(description="Argument parser")
    parser.add_argument("--mesh_type", help='Specify a mesh type: "watertight" or "simplified"', required=True) 
    parser.add_argument("--category", help="Specify a category from ['bed', 'couch', 'vase', 'pottedplant']", required=True)
    args = parser.parse_args()
    
    DATA_DIR = '../../data/'
    SAVE_DIR = '../../data/'

    #categories = ['vase', 'bed', 'tv', 'computer', 'lamp', 'pottedplant', 'couch', 'chair', 'desk', 'table']
    #categories = ['bed', 'couch', 'vase', 'pottedplant']
    category = args.category
    mesh_type = args.mesh_type
    
    print(f'Processing {category}')
    category_data_dir = os.path.join(DATA_DIR, category)
    category_save_dir = os.path.join(SAVE_DIR, category)
        
    shapes = [shape for shape in os.listdir(category_data_dir) if os.path.isdir(os.path.join(category_data_dir, shape))]
    for shape in shapes:
        shape_data_dir = os.path.join(category_data_dir, shape)
        shape_save_dir = os.path.join(category_save_dir, shape)
            
        mesh_path = os.path.join(shape_data_dir,f'mesh_{mesh_type}.obj')
        print(f'Mesh at {mesh_path}')
            
        mesh = trimesh.load(mesh_path)
        # Rotate the object to align with the canonical axis
        mesh = shapenet_rotate(mesh)
        # Fit into unit sphere
        scaled_and_translated_mesh = scale_and_translate_to_unit_sphere(mesh)
        scaled_and_translated_mesh.export(os.path.join(shape_save_dir, 'mesh_simplified.obj'), file_type='obj')

        points, sdf = sample_sdf_near_surface(scaled_and_translated_mesh, number_of_points=250000)

        # Create positive and negative samples according to sdf values with their corresponding points
        # Positive samples are the points that are inside the mesh
        # Negative samples are the points that are outside the mesh
        positive_indices = sdf > 0
        positive_samples = np.concatenate([points[positive_indices], sdf[positive_indices].reshape(-1, 1)], axis=1)
            
        negative_indices = sdf < 0
        negative_samples = np.concatenate([points[negative_indices], sdf[negative_indices].reshape(-1, 1)], axis=1)
            
        np.savez(os.path.join(shape_save_dir, 'sdf.npz'), pos=positive_samples, neg=negative_samples)


    # Rotation Augment Loop
    for shape in shapes:
        for i in range(5): # 5 rotation augments
            shape_data_dir = os.path.join(category_data_dir, shape)
            shape_save_dir = os.path.join(category_save_dir, shape)
            # Give it a new name in the form _rot_i
            shape_save_dir = os.path.join(shape_save_dir, '_rot_' + str(i))
            
            mesh_path = os.path.join(shape_data_dir,f'mesh_{mesh_type}.obj')
            print(f'Mesh at {mesh_path}')
                
            mesh = trimesh.load(mesh_path)
            mesh = shapenet_rotate(mesh)
            # Random rotation
            mesh = random_rotate(mesh)
            scaled_and_translated_mesh = scale_and_translate_to_unit_sphere(mesh)
            
            scaled_and_translated_mesh.export(os.path.join(shape_save_dir, 'mesh_simplified.obj'), file_type='obj')

            '''
            sampler = MeshSampler(scaled_and_translated_mesh)
                
            sampler.compute_visible_faces()
            sampler.sample_points(n_points=100000)
                
            correct_mesh_points, correct_sdf, correct_normals = sampler.compute_sdf(sigma=0.0025)
            '''
            
            points, sdf = sample_sdf_near_surface(scaled_and_translated_mesh, number_of_points=250000)

            # Create positive and negative samples according to sdf values with their corresponding points
            # Positive samples are the points that are inside the mesh
            # Negative samples are the points that are outside the mesh
            positive_indices = sdf > 0
            positive_samples = np.concatenate([points[positive_indices], sdf[positive_indices].reshape(-1, 1)], axis=1)
                
            negative_indices = sdf < 0
            negative_samples = np.concatenate([points[negative_indices], sdf[negative_indices].reshape(-1, 1)], axis=1)
                
            np.savez(os.path.join(shape_save_dir, 'sdf.npz'), pos=positive_samples, neg=negative_samples)


    # Scale Augment Loop
    for shape in shapes:
        for i in range(3): # 3 nonuniform scale augments
            shape_data_dir = os.path.join(category_data_dir, shape)
            shape_save_dir = os.path.join(category_save_dir, shape)
            # Give it a new name in the form _scale_i
            shape_save_dir = os.path.join(shape_save_dir, '_scale_' + str(i))
            
            mesh_path = os.path.join(shape_data_dir,f'mesh_{mesh_type}.obj')
            print(f'Mesh at {mesh_path}')
                
            mesh = trimesh.load(mesh_path)
            mesh = shapenet_rotate(mesh)
            # Random nonuniform scale
            mesh = random_nonuniform_scale(mesh)
            scaled_and_translated_mesh = scale_and_translate_to_unit_sphere(mesh)
            
            scaled_and_translated_mesh.export(os.path.join(shape_save_dir, 'mesh_simplified.obj'), file_type='obj')

            '''
            sampler = MeshSampler(scaled_and_translated_mesh)
                
            sampler.compute_visible_faces()
            sampler.sample_points(n_points=100000)
                
            correct_mesh_points, correct_sdf, correct_normals = sampler.compute_sdf(sigma=0.0025)
            '''
            
            points, sdf = sample_sdf_near_surface(scaled_and_translated_mesh, number_of_points=250000)

            # Create positive and negative samples according to sdf values with their corresponding points
            # Positive samples are the points that are inside the mesh
            # Negative samples are the points that are outside the mesh
            positive_indices = sdf > 0
            positive_samples = np.concatenate([points[positive_indices], sdf[positive_indices].reshape(-1, 1)], axis=1)
                
            negative_indices = sdf < 0
            negative_samples = np.concatenate([points[negative_indices], sdf[negative_indices].reshape(-1, 1)], axis=1)
                
            np.savez(os.path.join(shape_save_dir, 'sdf.npz'), pos=positive_samples, neg=negative_samples)
