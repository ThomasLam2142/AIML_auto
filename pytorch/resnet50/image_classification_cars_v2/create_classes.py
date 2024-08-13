import os

def list_subfolders(folder):
    return [f.name for f in os.scandir(folder) if f.is_dir()]

def save_subfolders_to_file(folder, output_file):
    subfolders = list_subfolders(folder)
    with open(output_file, 'w') as file:
        for subfolder in subfolders:
            file.write(f"{subfolder}\n")

folder_path = 'imgs'
output_file = 'classes.txt'
save_subfolders_to_file(folder_path, output_file)
