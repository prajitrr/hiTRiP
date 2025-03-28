import numpy as np
import pandas as pd
import cv2
# STL Imports
import os

# List of supported image formats
image_formats = ['jpg', 'jpeg', 'png', 'bmp', 'tiff', 'tif', 'gif', 'webp', 'tga', 'exr', 'hdr', 'pic', 'jp2', 'j2k', 'j2c', 'jpe', 'jp2', 'jif', 'jfi', 'jfif', 'heic', 'heif']

def order_points(array, g_height, g_width):
    grid_width = 4
    sorted_y_ind = np.argsort(array[:,1])
    sorted_y = np.concatenate((array[sorted_y_ind], sorted_y_ind[..., None]), axis=1)
    for i in range(0, len(sorted_y), grid_width):
        sorted_x_ind = np.argsort(sorted_y[i:i+grid_width,0])
        sorted_y[i:i+grid_width] = sorted_y[i:i+grid_width][sorted_x_ind]
    result = sorted_y[:,2].tolist()
    result = [int(i) for i in result]
    return result

def generate_video(images_path, video_path, fps=20.0):
    """
    Helper function to convert images to video
    @param images_path: Path to the folder containing the images
    @param video_path: Path to the video file to be created
    @param fps: Frames per second of the video
    @return None
    """
    # Filter out non-image files
    sorted_images = [file for file in os.listdir(images_path) if os.path.isfile(os.path.join(images_path, file))]
    # Filter out macOS specific files using Windows-compatible path checks
    sorted_images = [index for index in sorted_images if "__MACOSX" not in index]
    sorted_images = [index for index in sorted_images if ".DS_Store" not in index]
    sorted_images = [index for index in sorted_images if index.split('.')[-1].lower() in image_formats]
    sorted_images.sort()
    
    # Make sure we have images to process
    if not sorted_images:
        raise ValueError(f"No valid image files found in {images_path}")
    
    image = os.path.join(images_path, sorted_images[0])
    height, width = cv2.imread(image, cv2.IMREAD_UNCHANGED).shape[:2]
    
    # Use XVID codec for Windows compatibility instead of mp4v
    if os.name == 'nt':  # Check if running on Windows
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        video_path_win = video_path.replace('.mp4', '.avi')  # Use AVI format on Windows
        video = cv2.VideoWriter(video_path_win, fourcc, float(fps), (width, height))
    else:
        video = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*"mp4v"), float(fps), (width, height))
    
    crop_coords = os.path.join(images_path, "crop.txt")
    
    # Check if crop file exists
    if not os.path.exists(crop_coords):
        raise FileNotFoundError(f"Crop file not found at {crop_coords}")
    
    # Handle different line endings on Windows
    with open(crop_coords, 'r') as f:
        lines = [line.strip() for line in f.readlines() if line.strip()]
    
    # Create a DataFrame from the lines
    coordinates = pd.DataFrame(lines)
    
    # check that the crop file has only one column
    assert coordinates.shape[1] == 1, "Crop file must have only one column"
    # check that the crop file has at least one row
    assert coordinates.shape[0] > 0, "Crop file must have at least one row"
    # check that all rows have 5 elements separated by a single space
    assert all(len(i.split(" ")) == 5 for i in coordinates.iloc[:,0]), "All rows must have 5 elements separated by a single space"
    
    keys = []
    values = []
    for plant in range(len(coordinates)):
        # Get ID (path to each 'cropped' folder)
        plant_ID = coordinates.iloc[plant,0]
        plant_ID = plant_ID.split(" ")[0]  # Split by " "; take the first
        keys.append(plant_ID)  # Add this to the 'keys' list
        
        # Get coordinates
        coords = coordinates.iloc[plant,0]
        coords = coords.split(" ")[1:]  # This must change if using \t sep!!
        coords = [int(i) for i in coords]  # Convert values to integers
        values.append(tuple(coords))  # Append coords as tuple
    
    regions = dict(zip(keys, values))
    
    for i, file in enumerate(sorted_images):
        image = os.path.join(images_path, file)
        frame = cv2.imread(image, cv2.IMREAD_UNCHANGED)
        
        # Skip non-image files that might have slipped through
        if frame is None:
            continue
            
        for plant, coords in regions.items():
            # Get coordinates
            x, y, w, h = coords
            # Add boxes and text to plants
            frame = cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 200), 3)
            frame = cv2.putText(frame, plant, (x - 2, y + 7), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 1, cv2.LINE_AA)
        
        video.write(frame)
    
    cv2.destroyAllWindows()
    video.release()