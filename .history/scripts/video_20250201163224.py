import numpy as np
import pandas as pd
import cv2
from autocrop import order_points

#STL Imports
import os

image_formats = ['jpg', 'jpeg', 'png', 'bmp', 'tiff', 'tif', 'gif', 'webp', 'tga', 'exr', 'hdr', 'pic', 'jp2', 'j2k', 'j2c', 'jpe', 'jp2', 'jif', 'jfi', 'jfif', 'heic', 'heif']


def generate_video(images_path, video_path, fps=20.0):
    #Helper function to convert images to video
    #@param images_path: Path to the folder containing the images
    #@param video_path: Path to the video file to be created
    #@param fps: Frames per second of the video
    #@return None

    #Filter out non-image files
    sorted_images = [file for file in os.listdir(images_path) if os.path.isfile(os.path.join(images_path, file))]
    sorted_images = [index for index in sorted_images if "__MACOSX/" not in index]
    sorted_images = [index for index in sorted_images if ".DS_Store" not in index]
    sorted_images = [index for index in sorted_images if index.split('.')[-1].lower() in image_formats]
    sorted_images.sort()

    image = os.path.join(images_path, sorted_images[0])
    height, width= cv2.imread(image,cv2.IMREAD_UNCHANGED).shape[:2]
    
    video = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*"mp4v"), float(fps), (width, height))

    crop_coords = os.path.join(images_path, "crop.txt")

    coordinates = pd.read_csv(crop_coords, sep='\t', header=None)
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
        plant_ID = coordinates.iloc[plant,0] #
        plant_ID = plant_ID.split(" ")[0] # Split by " "; take the first
        keys.append(plant_ID) # Add this to the 'keys' list
        # print(f"Processing {plant_ID}...")
        # Get coordinates
        coords = coordinates.iloc[plant,0]
        coords = coords.split(" ")[1:]  # This must change if using \t sep!!
        coords = [int(i) for i in coords] # Convert values to integers
        values.append(tuple(coords))  # Append coords as tuple
    

    #Avoided reordering the coordinates
    #coordinate_points = np.array([(coord[0] + coord[2]/2, coord[1] + coord[3]/2) for coord in coords], dtype=np.float16)
    #sorted_values = [values[i] for i in order_points(coordinate_points, g_height=grid_height, g_width=grid_width)]

    regions = dict(zip(keys, values))

    for i, file in enumerate(sorted_images):
        image = os.path.join(images_path, file)
        frame = cv2.imread(image,cv2.IMREAD_UNCHANGED)
        for plant, coords in regions.items():
            # Get coordinates
            x, y, w, h = coords

            # Add boxes and text to plants
            frame = cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 200), 3)
            frame = cv2.putText(frame, plant, (x - 2, y + 7), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
        video.write(frame)
    
    cv2.destroyAllWindows()
    video.release()