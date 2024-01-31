import trimesh
import numpy as np
import os
import argparse
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

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Argument parser")
    parser.add_argument("--category", help="Specify a category from ['bed', 'couch', 'vase', 'pottedplant']", required=True)
    args = parser.parse_args()
    
    DATA_DIR = '../../data/'
    SAVE_DIR = '../../data/'

    category = args.category

    category_data_dir = os.path.join(DATA_DIR, category)
    category_save_dir = os.path.join(SAVE_DIR, category)

    shapes = [shape for shape in os.listdir(category_data_dir) if os.path.isdir(os.path.join(category_data_dir, shape))]
    for shape in shapes:
        shape_data_dir = os.path.join(category_data_dir, shape)
        shape_save_dir = os.path.join(category_save_dir, shape)
        
        mesh_path = os.path.join(shape_data_dir, 'mesh_simplified.obj')

        mesh = trimesh.load(mesh_path)
        
        rotated_mesh = shapenet_rotate(mesh)
        rotated_mesh.export(mesh_path)        