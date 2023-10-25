import os

# Specify the folder containing your files
folder_path = "C:\\Users\\admin\\Desktop\\DS&ML\\FaceRecog\\data"



# List all the files in the folder
file_list = os.listdir(folder_path)

# Iterate through the files and add the ".jpg" extension
for filename in file_list:
    # Check if the file doesn't already have the ".jpg" extension
    if not filename.endswith(".jpg"):
        new_filename = filename + ".jpg"
        # Rename the file with the ".jpg" extension
        os.rename(os.path.join(folder_path, filename), os.path.join(folder_path, new_filename))

print("File extensions have been updated to .jpg.")
