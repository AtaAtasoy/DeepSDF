import glob
import os
import shutil

def move_file(source_path, target_path):
    # Check if the target directory exists, and create it if not
    target_directory = os.path.dirname(target_path)
    if not os.path.exists(target_directory):
        os.makedirs(target_directory)

    # Move the file to the target path
    shutil.copy(source_path, target_path)
    print(f"File copied successfully from '{source_path}' to '{target_path}'")


#Get file names and classes
def generate_dictionary_from_file(file_path):
    result_dict = {}

    with open(file_path, 'r') as file:
        for line in file:
            # Assuming the format is value1, value2, value3, ...
            values = line.strip().split(',')
            
            # Creating a dictionary entry with the first value as key and the rest as values in a list
            key = values[0]
            values = values[1]
            result_dict[key] = values

    return result_dict


def get_name_from_path(file_path):
    # Extract filename from the path
    file_name = os.path.basename(file_path)

    # Extract the desired string from the filename
    desired_string = file_name.split('.')[0]

    return desired_string

# Example usage:
#file_path = "/cluster/51/ataatasoy/amazon-berkeley-objects/metadata/abo_classes_3d.txt"  # Replace with the path to your text file
#print(generated_dict)

# Get files from 3dmodels folders and create a new dataset folder


def count_and_sort_dictionary_values(dictionary):
    value_counts = {}
    
    # Count occurrences of each unique value in the dictionary
    for value in dictionary.values():
        if value in value_counts:
            value_counts[value] += 1
        else:
            value_counts[value] = 1
    
    # Sort the results by the number of occurrences
    sorted_value_counts = sorted(value_counts.items(), key=lambda x: x[1], reverse=True)
    
    # Print the sorted results
    for unique_value, count in sorted_value_counts:
        print(f"Value'{unique_value}' count {count}")

# Example usage

# count_and_sort_dictionary_values(generated_dict)
# search all files inside a specific folder
# *.* means file name with any extension

if __name__ == "__main__":

    dir_path = r'/cluster/51/ataatasoy/abo_dst/3dmodels/original/**/*.glb'
    for file in glob.glob(dir_path, recursive=True):
        file_path = "/cluster/51/ataatasoy/amazon-berkeley-objects/metadata/abo_classes_3d.txt"  # Replace with the path to your text file
        generated_dict = generate_dictionary_from_file(file_path)
        source_path = file    
        result = get_name_from_path(file)
        target_path = f'/cluster/51/ataatasoy/project/data/picture_frame_painting/{result}/mesh.glb'
        category = generated_dict[result]
        print('Source path:', source_path, 'Data name:', result, 'Target path:', target_path, 'Category:', category)
        #if category in ["sofa","chair","rug","picture frame or painting"]:
        if category in  ["picture frame or painting"]:
            move_file(source_path, target_path)