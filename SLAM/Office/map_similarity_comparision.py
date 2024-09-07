from skimage.metrics import structural_similarity as ssim
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt

def zoom_in_image(image, zoom_factor=1.5):
    """
    Zoom in on the image by a specified zoom factor.
    
    :param image: The input image to zoom in on.
    :param zoom_factor: The factor by which to zoom in.
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

def resize_image(image, size=(480, 480)):
    # Resize image to the specified size
    return cv2.resize(image, size, interpolation=cv2.INTER_AREA)

def calculate_mse(imageA, imageB):
    # Mean Squared Error
    err = np.sum((imageA.astype("float") - imageB.astype("float")) ** 2)
    err /= float(imageA.shape[0] * imageA.shape[1])
    return err

def calculate_ssim(imageA, imageB):
    # Structural Similarity Index (SSIM)
    ssim_value, _ = ssim(imageA, imageB, full=True)
    return ssim_value

def calculate_entropy(image):
    # Calculate entropy of the image
    histogram, _ = np.histogram(image.flatten(), bins=256, range=(0, 256))
    histogram = histogram / histogram.sum()
    entropy = -np.sum(histogram * np.log2(histogram + 1e-7))
    return entropy

def compare_maps(static_map_path, dynamic_map_path):
    # Load the images
    static_map = np.array(Image.open(static_map_path).convert('L'))
    dynamic_map = np.array(Image.open(dynamic_map_path).convert('L'))

    # Zoom in on both images by a specified zoom factor
    static_map_zoomed = zoom_in_image(static_map, zoom_factor=1.33)
    dynamic_map_zoomed = zoom_in_image(dynamic_map, zoom_factor=2.5)

    # Resize both images to 480x480 pixels
    static_map_resized = resize_image(static_map_zoomed, size=(640, 640))
    dynamic_map_resized = resize_image(dynamic_map_zoomed, size=(640, 640))

    # Calculate metrics
    mse_value = calculate_mse(static_map_resized, dynamic_map_resized)
    ssim_value = calculate_ssim(static_map_resized, dynamic_map_resized)
    static_entropy = calculate_entropy(static_map_resized)
    dynamic_entropy = calculate_entropy(dynamic_map_resized)

    # Print results
    print(f"Mean Squared Error (MSE): {mse_value}")
    print(f"Structural Similarity Index (SSIM): {ssim_value}")
    print(f"Entropy of Static Map: {static_entropy}")
    print(f"Entropy of Dynamic Map: {dynamic_entropy}")

    # Plot the images and their difference
    plt.figure(figsize=(10, 8))

    plt.subplot(1, 3, 1)
    plt.title('Static Environment Map')
    plt.imshow(static_map_resized, cmap='gray')
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.title('Dynamic Environment Map')
    plt.imshow(dynamic_map_resized, cmap='gray')
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.title('Difference Image')
    diff_image = cv2.absdiff(static_map_resized, dynamic_map_resized)
    plt.imshow(diff_image, cmap='hot')
    plt.axis('off')

    plt.tight_layout()

    output_file_path = '/home/sashitsharma/Desktop/thesis_github/Thesis/SLAM/Office/gmapping/dynamic_static_gmapping_comparision.png'  # Replace with your desired file path
    plt.savefig(output_file_path, bbox_inches='tight', pad_inches=0.1)
    plt.show()

if __name__ == "__main__":
    # Paths to the static and dynamic environment maps
    static_map_path = '/home/sashitsharma/Desktop/thesis_github/Thesis/SLAM/Office/gmapping/gmapping_map_static_image.png'  # Replace with your actual file path
    dynamic_map_path = '/home/sashitsharma/Desktop/thesis_github/Thesis/SLAM/Office/gmapping/gmapping_map_dynamic_image.png'  # Replace with your actual file path

    # Compare the maps
    compare_maps(static_map_path, dynamic_map_path)
