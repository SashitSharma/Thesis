from PIL import Image
import os

def convert_pgm_to_image(pgm_file_path, output_file_path):
    """
    Converts a .pgm image file to a standard image format (.png or .jpeg).
    
    :param pgm_file_path: Path to the .pgm file
    :param output_file_path: Path to save the converted image, including file extension (.png or .jpeg)
    """
    try:
        # Open the .pgm file
        with Image.open(pgm_file_path) as img:
            # Save it as a new image format
            img.save(output_file_path)
        print(f"Converted {pgm_file_path} to {output_file_path}")
    except Exception as e:
        print(f"Failed to convert {pgm_file_path}: {e}")

if __name__ == "__main__":
    # Example file paths
    pgm_file_path = '/home/sashitsharma/Desktop/thesis_github/Thesis/SLAM/Office/karto/karto_map_dynamic.pgm'  # Replace with your .pgm file path
    output_file_path = '/home/sashitsharma/Desktop/thesis_github/Thesis/SLAM/Office/karto/karto_map_static_dynamic_image.png'   # Replace with your desired output path and format (.png or .jpeg)
    
    # Convert the .pgm file to the desired image format
    convert_pgm_to_image(pgm_file_path, output_file_path)
