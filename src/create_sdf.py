import trimesh
import numpy as np
import os
import argparse
import pybullet as pb
import random
import pickle
from mesh_to_sdf import sample_sdf_near_surface
import point_cloud_utils as pcu


def _as_mesh(scene_or_mesh):
    # Utils function to get a mesh from a trimesh.Trimesh() or trimesh.scene.Scene()
    if isinstance(scene_or_mesh, trimesh.Scene):
        mesh = trimesh.util.concatenate([
            trimesh.Trimesh(vertices=m.vertices, faces=m.faces)
            for m in scene_or_mesh.geometry.values()])
    else:
        mesh = scene_or_mesh
    return mesh

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


def check_watertight(mesh_path):
    try:
        #verts, faces = pcu.load_mesh_vf(mesh_path)  
        mesh = trimesh.load(mesh_path, force='mesh')

        print(f'Checking watertightness of {mesh_path}', mesh.is_watertight)

        if not mesh.is_watertight:
            verts, faces = pcu.make_mesh_watertight(mesh.vertices, mesh.faces, 20_000)
            mesh = trimesh.Trimesh(vertices=verts, faces=faces)

    except Exception as e:
        print(e)
    return mesh


def create_sdf_values(shape_save_dir, mesh, number_of_points=125000):

    points, sdf = sample_sdf_near_surface(mesh, number_of_points=number_of_points)

    # Create positive and negative samples according to sdf values with their corresponding points
    # Positive samples are the points that are inside the mesh
    # Negative samples are the points that are outside the mesh
    positive_indices = sdf > 0
    positive_samples = np.concatenate([points[positive_indices], sdf[positive_indices].reshape(-1, 1)], axis=1)
            
    negative_indices = sdf < 0
    negative_samples = np.concatenate([points[negative_indices], sdf[negative_indices].reshape(-1, 1)], axis=1)
            
    np.savez(os.path.join(shape_save_dir, 'sdf.npz'), pos=positive_samples, neg=negative_samples)


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


def apply_no_augmentation(category_save_dir, category_data_dir, shapes):
    print('No Augment Loop')
    for shape in shapes:
        shape_data_dir = os.path.join(category_data_dir, shape)
        shape_save_dir = os.path.join(category_save_dir, shape)
            
        mesh_path = os.path.join(shape_data_dir,f'mesh.glb')
        print(f'Mesh at {mesh_path}')
        
        # Check if the mesh is watertight
        # If not watertight, make it watertight
        mesh = check_watertight(mesh_path)
        #try:
            #verts, faces = pcu.load_mesh_vf(mesh_path)
            #mesh = _as_mesh(trimesh.load(mesh_path))
            #mesh = trimesh.load(mesh_path, force='mesh')
            #if not mesh.is_watertight:
                #verts, faces = pcu.make_mesh_watertight(mesh.vertices, mesh.faces, 20_000)
                #mesh = trimesh.Trimesh(vertices=verts, faces=faces)
        #except Exception as e:
        #    print(e)
        #mesh = trimesh.load(mesh_path, force='mesh')

        # Fit into unit sphere
        scaled_and_translated_mesh = scale_and_translate_to_unit_sphere(mesh)
        scaled_and_translated_mesh.export(os.path.join(shape_save_dir, 'mesh.glb'), file_type='glb')

        # Rotate the object to align with the canonical axis
        mesh = shapenet_rotate(mesh)
        # Create sdf values and files for the mesh
        create_sdf_values(shape_save_dir = shape_save_dir, mesh = scaled_and_translated_mesh, number_of_points=125000)


def apply_scale_augmentation(category_save_dir, category_data_dir, shapes, num_of_augments=3):
    print('Scale Augment Loop')
    for shape in shapes:
        for i in range(num_of_augments): # 3 nonuniform scale augments
            shape_data_dir = os.path.join(category_data_dir, shape)
            shape_save_dir = os.path.join(category_save_dir, shape)
            # Give it a new name in the form _scale_i
            shape_save_dir = os.path.join(shape_save_dir + '_scale_' + str(i))
            
             # If the directory does not exist, create it
            if not os.path.exists(shape_save_dir):
                os.makedirs(shape_save_dir)
            
            mesh_path = os.path.join(shape_data_dir,f'mesh.glb')
            print(f'Mesh at {mesh_path}')
                
            #Check if the mesh is watertight
            mesh = check_watertight(mesh_path)

            # Random nonuniform scale
            mesh = random_nonuniform_scale(mesh)
            scaled_and_translated_mesh = scale_and_translate_to_unit_sphere(mesh)
            mesh = shapenet_rotate(mesh)
            
            mesh.export(os.path.join(shape_save_dir, 'mesh.glb'), file_type='glb')
            
            # Create sdf values and files for the mesh
            create_sdf_values(shape_save_dir = shape_save_dir, mesh = mesh, number_of_points=125000)
    
           


def apply_rotation_augmentation(category_save_dir, category_data_dir, shapes, num_of_augments=2):
    print('Rotation Augment Loop')
    total_shapes = len(shapes)
    ctr = 0
    for shape in shapes:
        for i in range(num_of_augments): # 5 rotation augments
            shape_data_dir = os.path.join(category_data_dir, shape)
            shape_save_dir = os.path.join(category_save_dir, shape)
            # Give it a new name in the form _rot_i
            shape_save_dir = os.path.join(shape_save_dir + '_rot_' + str(i))
            
            # If the directory does not exist, create it
            if not os.path.exists(shape_save_dir):
                os.makedirs(shape_save_dir)
            
            mesh_path = os.path.join(shape_data_dir,f'mesh.glb')
            #print(f'Mesh at {mesh_path}')
            
            #Check if the mesh is watertight
            mesh = check_watertight(mesh_path)
            
            scaled_and_translated_mesh = scale_and_translate_to_unit_sphere(mesh)
            # Random rotation
            mesh = random_rotate(mesh)
            
            mesh.export(os.path.join(shape_save_dir, 'mesh.glb'), file_type='glb')

            # Create sdf values and files for the mesh
            create_sdf_values(shape_save_dir = shape_save_dir, mesh = mesh, number_of_points=125000)
        ctr += 1

        print(f'{ctr}/{total_shapes} done')
    
            

if __name__ =="__main__":
    os.environ["PYOPENGL_PLATFORM"] = "egl"
    parser = argparse.ArgumentParser(description="Argument parser")
    parser.add_argument("--category", help="Specify a category from ['chair', 'picture_frame_painting', 'sofa', 'rug']", required=True)
    args = parser.parse_args()
    
    DATA_DIR = '/cluster/51/ataatasoy/project/data'
    SAVE_DIR = '/cluster/51/ataatasoy/project/data'

    category = args.category
    
    print(f'Processing {category}')
    category_data_dir = os.path.join(DATA_DIR, category)
    category_save_dir = os.path.join(SAVE_DIR, category)
        
    shapes = [shape for shape in os.listdir(category_data_dir) if os.path.isdir(os.path.join(category_data_dir, shape))]
    
    apply_no_augmentation(category_save_dir,category_data_dir,shapes)
    #apply_rotation_augmentation(category_save_dir,category_data_dir,shapes, num_of_augments=2)
    #apply_scale_augmentation(category_save_dir,category_data_dir,shapes, num_of_augments=3)
    
    
'''
    # Rotation Augment Loop
    print('Rotation Augment Loop')
    for shape in shapes:
        for i in range(2): # 5 rotation augments
            shape_data_dir = os.path.join(category_data_dir, shape)
            shape_save_dir = os.path.join(category_save_dir, shape)
            # Give it a new name in the form _rot_i
            shape_save_dir = os.path.join(shape_save_dir + '_rot_' + str(i))
            
            # If the directory does not exist, create it
            if not os.path.exists(shape_save_dir):
                os.makedirs(shape_save_dir)
            
            mesh_path = os.path.join(shape_data_dir,f'mesh.glb')
            print(f'Mesh at {mesh_path}')

            mesh = trimesh.load(mesh_path)
            mesh = shapenet_rotate(mesh)
                
            # Check if the mesh is watertight
            try:
                verts, faces = pcu.load_mesh_vf(mesh_path)     
                mesh = _as_mesh(trimesh.load(mesh_path))
            
                if not mesh.is_watertight:
                    verts, faces = pcu.make_mesh_watertight(mesh.vertices, mesh.faces, 20_000)
                    mesh = trimesh.Trimesh(vertices=verts, faces=faces)

            except Exception as e:
                print(e)
                continue
            
            
                
            
            # Random rotation
            mesh = random_rotate(mesh)
            scaled_and_translated_mesh = scale_and_translate_to_unit_sphere(mesh)
            
            scaled_and_translated_mesh.export(os.path.join(shape_save_dir, 'mesh_simplified.obj'), file_type='obj')

           
            points, sdf = sample_sdf_near_surface(scaled_and_translated_mesh, number_of_points=125000)

            # Create positive and negative samples according to sdf values with their corresponding points
            # Positive samples are the points that are inside the mesh
            # Negative samples are the points that are outside the mesh
            positive_indices = sdf > 0
            positive_samples = np.concatenate([points[positive_indices], sdf[positive_indices].reshape(-1, 1)], axis=1)
                
            negative_indices = sdf < 0
            negative_samples = np.concatenate([points[negative_indices], sdf[negative_indices].reshape(-1, 1)], axis=1)
                
            np.savez(os.path.join(shape_save_dir, 'sdf.npz'), pos=positive_samples, neg=negative_samples)
    
'''
'''
    # Scale Augment Loop
    print('Scale Augment Loop')
    for shape in shapes:
        for i in range(3): # 3 nonuniform scale augments
            shape_data_dir = os.path.join(category_data_dir, shape)
            shape_save_dir = os.path.join(category_save_dir, shape)
            # Give it a new name in the form _scale_i
            shape_save_dir = os.path.join(shape_save_dir + '_scale_' + str(i))
            
             # If the directory does not exist, create it
            if not os.path.exists(shape_save_dir):
                os.makedirs(shape_save_dir)
            
            mesh_path = os.path.join(shape_data_dir,f'mesh_{mesh_type}.obj')
            print(f'Mesh at {mesh_path}')
                
            mesh = trimesh.load(mesh_path)
            mesh = random_nonuniform_scale(mesh)
            scaled_and_translated_mesh = scale_and_translate_to_unit_sphere(mesh)
            # Random nonuniform scale
            mesh = shapenet_rotate(mesh)
            
            scaled_and_translated_mesh.export(os.path.join(shape_save_dir, 'mesh_simplified.obj'), file_type='obj')
            
            points, sdf = sample_sdf_near_surface(scaled_and_translated_mesh, number_of_points=125000)

            # Create positive and negative samples according to sdf values with their corresponding points
            # Positive samples are the points that are inside the mesh
            # Negative samples are the points that are outside the mesh
            positive_indices = sdf > 0
            positive_samples = np.concatenate([points[positive_indices], sdf[positive_indices].reshape(-1, 1)], axis=1)
                
            negative_indices = sdf < 0
            negative_samples = np.concatenate([points[negative_indices], sdf[negative_indices].reshape(-1, 1)], axis=1)
                
            np.savez(os.path.join(shape_save_dir, 'sdf.npz'), pos=positive_samples, neg=negative_samples)
'''
