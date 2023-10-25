import os
import tensorflow as tf

def preprocess_images_in_folder(folder_path, image_size):
    # Get a list of files in the folder
    image_files = [f for f in os.listdir(folder_path) if f.lower().endswith('.jpg')]

    for image_file in image_files:
        image_path = os.path.join(folder_path, image_file)
        
        # Read the image from the file
        image = tf.io.read_file(image_path)

        # Decode the JPEG image
        image = tf.image.decode_jpeg(image, channels=3)

        # Resize the image to the desired size (160x160 pixels)
        image = tf.image.resize(image, image_size)

        # Convert pixel values to the range [0, 1]
        image = tf.image.convert_image_dtype(image, tf.float32)

        # Encode and save the preprocessed image back to the same file
        encoded_image = tf.image.encode_jpeg(tf.cast(image * 255, tf.uint8))
        tf.io.write_file(image_path, encoded_image)

# Example usage:
input_folder_path = "C:\\Users\\admin\\Desktop\\DS\\FaceRecog\\data"
image_size = (160, 160)

preprocess_images_in_folder(input_folder_path, image_size)
