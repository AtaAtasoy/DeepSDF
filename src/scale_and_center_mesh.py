import trimesh
import os

# In DeepSDF, the mesh is scaled and translated to fit within the unit sphere
# It also states: "(in practice fit to sphere radius of 1/1.03)"
def scale_and_translate_to_unit_sphere(mesh):
    # Find the bounding box of the mesh
    min_coords, max_coords = mesh.bounds

    # Calculate the center and size of the bounding box
    center = (min_coords + max_coords) / 2
    size = max(max_coords - min_coords)

    # Scale and translate mesh to fit within the unit sphere
    scaled_mesh = mesh.apply_scale(1 / size)
    translated_and_scaled_mesh = scaled_mesh.apply_translation(-center / size / 1.03)
    
    # Save the translated and scaled mesh
    translated_and_scaled_mesh.export()
    
    return translated_and_scaled_mesh


if __name__ == "__main__":
    for category in ['bed', 'couch', 'vase', 'pottedplant']:
        print(f'Processing category {category}')
        # Get the list of shapes for the current category
        shapes = os.listdir(f'./data/{category}')

        # Do not include material files
        shapes = [shape for shape in shapes if shape[-4:] == '.obj']
        
        # Remove the '.obj' extension
        shapes = [shape[:-4] for shape in shapes]

        # Iterate through each shape
        for shape in shapes:
            # Load the mesh
            mesh = trimesh.load(f'./data/{category}/{shape}.obj')

            # Scale and translate the mesh to fit within the unit sphere
            mesh = scale_and_translate_to_unit_sphere(mesh)

            # Save the modified mesh to the same .obj file
            mesh.export(f'./data/{category}/{shape}.obj')