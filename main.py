import cv2

# List of file paths for your three JPG images
image_paths = ['image.jpg', 'image2.jpg', 'image3.jpg']

for image_path in image_paths:
    img = cv2.imread(image_path)
    gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    if img is not None:
        print(f"Shape of {image_path}: {gray_image.shape}")
    else:
        print(f"Failed to read {image_path}")

