from autocrop import autocrop
import pytrip as pt

import argparse
import sys
import time
import os
import zipfile



# Parser
parser = argparse.ArgumentParser(description='Run TRiP on an a folder of images')
parser.add_argument('-d','--img_directory', type=str, required=True, help='Path to images to crop, or to cropped images')
parser.add_argument('-e','--img_extension', type=str, required=False, help='Image extension (e.g. JPG, PNG, TIF)', default="JPG")
parser.add_argument('-c','--combine', type=str, required=False, help='Specify whether to combine numbering of different folders', default="True")
parser.add_argument('-mt','--motion', type=str, required=False, help='Estimate motion', default=None)
parser.add_argument('-m','--model', type=str, required=False, help='Fit model to motion data', default=None)
parser.add_argument('-s','--start_img', type=int, required=False, help='Start image number', default=None)
parser.add_argument('-f','--end_img', type=int, required=False, help='End image number', default=None)
args = parser.parse_args()


# Define the arguments
images_path = args.img_directory # Path to images to crop, or to cropped images
img_extension = str(args.img_extension) # Image extension (e.g. JPG, PNG, TIF)

# check if start_img is int
if args.start_img is not None:
    try:
        start_img = int(args.start_img)
    except ValueError:
        start_img = None
        print("start_img must be an integer")
else:
    start_img = None
# check if end_img is int
if args.end_img is not None:
    try:
        end_img = int(args.end_img)
    except ValueError:
        end_img = None
        print("end_img must be an integer")
else:
    end_img = None


if str(args.motion) == "True":
    motion = True
else:
    motion = False
if str(args.model) == "True":
    model = True
else:
    model = False

if str(args.combine) == "True":
    combine = True
elif str(args.combine) == "False":
    combine = False
else:
    print("Parameter combine must be either True or False")
    sys.exit()

# Run the functions
def TRiP():
    start_all = time.time()
    # Check if images_path exists
    assert os.path.exists(images_path), "I did not find the folder at, " + str(images_path)

    
    start_time = time.time() # start timer

    for root, dirs, files in os.walk(images_path):
        if len(files) != 0:
            break
        dirs.sort()
        previous_plants = 0
        for dir in dirs:
            dir_name = os.path.join(images_path, dir)
            print(dir_name)
            coordinates = autocrop(dir_name, 12, previous_plants, 10, f"../test/{dir_name}_video.mp4")
            crop_coords = os.path.join(dir_name, "crop.txt")
            with open(crop_coords, "w+") as f:
                for object_num, rect in enumerate(coordinates):
                    number = object_num + 1 + previous_plants
                    f.write(f'plant_A{number:02} ')
                    f.write(' '.join(map(str, rect)) + '\n')
            pt.crop_all(dir_name, crop_coords, img_extension, start_img=start_img, end_img=end_img)
            previous_plants = previous_plants + 12


    # img_path = os.path.dirname(os.path.realpath(images_path))

    # coordinates = autocrop(images_path, 12, 0, 45, "../test/out_video.mp4")
    # crop_coords = os.path.join(img_path, "crop.txt")

    # with open(crop_coords, "w+") as f:
    #     for object_num, rect in enumerate(coordinates):
    #         number = object_num + 1
    #         f.write(f'plant_A{number:02} ')
    #         f.write(' '.join(map(str, rect)) + '\n')

    # pt.crop_all(os.path.join(img_path, "Images"), crop_coords, img_extension, start_img=start_img, end_img=end_img)
    # end_time = time.time()  # End timer
    # total_time = round(end_time - start_time,2)
    # print("\nTime to crop: ", total_time, " seconds")
    # print("-----------------------------------\n")

    if motion == True:
        start_time = time.time() # start timer
        pt.estimateAll(indirname="./cropped/", outdirname="./output/motion",img_extension=img_extension) # Estimate motion
        end_time = time.time()  # End timer
        total_time = round(end_time - start_time,2)
        print("\nTime to estimate motion: ", total_time, " seconds")
        print("-----------------------------------\n")
    
    if model == True:
        start_time = time.time()
        pt.ModelFitALL() # Fit model to motion data
        end_time = time.time()  # End timer
        total_time = round(end_time - start_time,2)
        print("\nTime to fit model: ", total_time, " seconds")
        print("-----------------------------------")
    
    end_all = time.time()
    total_time_all = round(end_all - start_all,2)
    print("\nPyTRiP Execution completed!\n")
    print("Total time: ", total_time_all, " seconds\n\n")


if __name__ == "__main__":
    TRiP()

# Example usage:
# python3 TRiP.py -d ../../input/ -c ../crop.txt -mt True -m True
