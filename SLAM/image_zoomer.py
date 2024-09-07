import cv2
from PIL import Image

def zoom_in_image(image, zoom_factor=2):
    """
    Zoom in on the image by a specified zoom factor.
    
    :param image: The input image to zoom in on.
    :param zoom_factor: The factor by which to zoom in (default is 5 for 500% zoom).
    :return: The zoomed-in image.
    """
    height, width = image.shape[:2]
    
    # Calculate the size of the central region
    new_height = int(height / zoom_factor)
    new_width = int(width / zoom_factor)
    
    # Calculate the coordinates of the central region
    top = (height - new_height) // 2
    left = (width - new_width) // 2
    bottom = top + new_height
    right = left + new_width
    
    # Crop the central region
    cropped_image = image[top:bottom, left:right]
    
    # Resize the cropped region back to the original size
    zoomed_image = cv2.resize(cropped_image, (width, height), interpolation=cv2.INTER_CUBIC)
    
    return zoomed_image

def save_zoomed_image(input_image_path, output_image_path, zoom_factor=5):
    """
    Load an image, zoom in by the specified factor, and save the zoomed image.
    
    :param input_image_path: Path to the input image.
    :param output_image_path: Path to save the zoomed image.
    :param zoom_factor: The factor by which to zoom in (default is 5).
    """
    # Load the image
    image = cv2.imread(input_image_path)

    # Zoom in on the image
    zoomed_image = zoom_in_image(image, zoom_factor=zoom_factor)

    # Save the zoomed image to the specified path
    cv2.imwrite(output_image_path, zoomed_image)
    print(f"Zoomed image saved to: {output_image_path}")

if __name__ == "__main__": 
    input_image_path = '/home/sashitsharma/Desktop/thesis_github/Thesis/SLAM/Results/office_map_image.png' 
    output_image_path = '/home/sashitsharma/Desktop/thesis_github/Thesis/SLAM/Results/office_map_zoomed_image.png' 
    # Call the function to zoom in and save the image
    save_zoomed_image(input_image_path, output_image_path, zoom_factor=5)
