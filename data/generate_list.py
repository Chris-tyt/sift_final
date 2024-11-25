import os

# Get the absolute path of the directory where the script is located
script_dir = os.path.dirname(os.path.abspath(__file__))

# Define the path to the images folder
images_dir = os.path.join(script_dir, 'images')

# Retrieve all image file paths in the folder relative to the script's directory
image_files = [
    os.path.relpath(os.path.join(images_dir, f), script_dir)
    for f in os.listdir(images_dir)
    if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif'))
]

# Check if the folder contains any images
if not image_files:
    print("No images found in the 'images' folder.")
else:
    # Open (or create) list.txt file for writing
    with open(os.path.join(script_dir, 'list.txt'), 'w') as f:
        # Write the first line: the relative path of the first image
        f.write(image_files[0] + '\n')
        
        # Write the second line: the number of images minus 1
        f.write(str(len(image_files) - 1) + '\n')
        
        # Write each subsequent line: the relative paths of all images except the first one
        for image in image_files[1:]:
            f.write(image + '\n')

    print(f"list.txt has been generated successfully with {len(image_files)} images.")
