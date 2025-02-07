from autocrop import autocrop
from video import generate_video
import pytrip as pt

import argparse
import sys
import time
import os
import zipfile


FILE_SEPARATOR = ".........."

# Parser
parser = argparse.ArgumentParser(description='Run accelTRiP on an a folder of images')
parser.add_argument('-d','--img_directory', type=str, required=True, help='Path to images to crop, or to cropped images')

#parser.add_argument('-n','--number_objects', type=str, required=True, help='Number of objects in each image to detect',)

parser.add_argument('-a','--automatic', type=str, required=False, help='Specify whether to perform automatic cropping or not', default="False")
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

if str(args.automatic) == "True":
    automatic = True
elif str(args.automatic) == "False":
    automatic = False
else:  
    print("Parameter automatic must be either True or False")
    sys.exit()

# Run the functions
def TRiP():
    start_all = time.time()
    # Check if images_path exists
    assert os.path.exists(images_path), "I did not find the folder at, " + str(images_path)

    
    start_time = time.time() # start timer

    root, dirs, files = next(os.walk(images_path))
    dirs.sort()
    previous_plants = 0
    num_dirs = len(dirs)
    
    if automatic:
        print(f"---------------------------------------------------------------------")
        print(f"PERFORMING OBJECT DETECTION ON {num_dirs} FOLDERS...")
        for num, dir in enumerate(dirs):
            print(f"Detecting objects in folder {num+1}/{num_dirs}: {dir}")
            dir_name = os.path.join(images_path, dir)
            video = os.path.join(dir_name, f"_{dir}_video.mp4")
            coordinates = autocrop(dir_name, int(args.number_objects), previous_plants, 10, video_name = video)
            crop_coords = os.path.join(dir_name, "crop.txt")
            with open(crop_coords, "w+") as f:
                for object_num, rect in enumerate(coordinates):
                    number = object_num + 1 + previous_plants
                    f.write(f'plant_A{number:02} ')
                    f.write(' '.join(map(str, rect)) + '\n')
            previous_plants += len(coordinates)
        input("Press Enter to continue once you have made all desired changes to the crop.txt files")
    
    elif not automatic:
        print(f"---------------------------------------------------------------------")

        skip_master = False
        master_crop_path_global = os.path.join(os.path.dirname(os.path.abspath(__file__)) + "master_files/master_crop_global.txt")
        if os.path.exists(master_crop_path_global):
            user_input_skip_master = ""
            while (user_input_skip_master != "y" and user_input_skip_master != "n"):
                user_input_skip_master = input(f"Master crop file found. Would you like to use this file to generate individual crop files? (Y/N) \n")
                user_input_skip_master = user_input_skip_master.lower().strip()
            if user_input_skip_master == "y":
                print(f"Using master crop file to generate individual crop files...")
                with open(master_crop_path_global, "r") as f:
                    for line in f:
                        dir_name, crop_coords = line.split(FILE_SEPARATOR)
                        crop_path = os.path.join(images_path, dir_name, "crop.txt")
                        with open(crop_path, "w+") as f2:
                            f2.write(crop_coords)
            else:
                skip_master = True
        elif not os.path.exists(master_crop_path_global) or skip_master:
            print(f"GENERATING crop.txt FILES FOR {num_dirs} FOLDERS. Will not generate crop.txt files if already found...")
            for num, dir in enumerate(dirs):
                print(f"Generating crop.txt file for folder {num+1}/{num_dirs}: {dir}")
                dir_name = os.path.join(images_path, dir)
                crop_coords = os.path.join(dir_name, "crop.txt")
                if not os.path.exists(crop_coords):
                    with open(crop_coords, "w+") as f:
                        f.write("")
                else:
                    print(f"A crop.txt file was already found in folder {dir}. Skipping...")
            input("Press Enter to continue once you have added coordinates to all the crop.txt files")
        while True:
            print(f"---------------------------------------------------------------------")
            print(f"GENERATING VIDEOS FOR {num_dirs} FOLDERS...")
            for num, dir in enumerate(dirs):
                print("Generating video for folder {}/{}: {}".format(num+1, num_dirs, dir))
                dir_name = os.path.join(images_path, dir)
                video = os.path.join(dir_name, f"_{dir}_video.mp4")
                generate_video(dir_name, video)
            usr_input = input("Video was generated. Please view the video. \n" + 
                              "If you are satisfied, type \"next\". Then, press Enter to continue. \n" + 
                              "If you are not satisfied, first edit the crop.txt files, then type \"crop\". Then, press Enter to continue. \n")
            usr_input = usr_input.lower().strip()
            while (usr_input != "next" and usr_input != "crop"):
                usr_input = input("Invalid input. Please choose either \"next\" or \"crop\" as your input. Then, press Enter to continue. \n")
                usr_input = usr_input.lower().strip()
            if (usr_input == "next"):
                break
            elif (usr_input == "crop"):
                continue
            else:
                 raise ValueError("Invalid input") 
        master_crop_generation = ""
        while (master_crop_generation != "y" and master_crop_generation != "n"):
            master_crop_generation = input("Would you like to generate a master crop.txt file? (Y/N) \n")
            master_crop_generation = master_crop_generation.lower().strip()
        
        if master_crop_generation == "y":
            print(f"---------------------------------------------------------------------")
            print(f"GENERATING MASTER crop.txt FILE...")
            master_crop_path = os.path.join(images_path, "master_crop.txt")
            master_crop_path_global = os.path.join(os.path.dirname(os.path.abspath(__file__)) + "master_files/master_crop_global.txt")
            with open(master_crop_path, "w+") as f, open(master_crop_path_global, "w+") as f2:
                for num, dir in enumerate(dirs):
                    dir_name = os.path.join(dir)
                    crop_coords = os.path.join(dir_name, "crop.txt")
                    with open(crop_coords, "r") as crop_file:
                        for line in crop_file:
                            f.write(dir_name + FILE_SEPARATOR + line)
                            f2.write(dir_name + FILE_SEPARATOR + line)
    
    print(f"----------------------------------------------------------------------")
    print(f"CROPPING IMAGES IN {num_dirs} FOLDERS...")
    for num, dir in enumerate(dirs):
        print(f"Cropping images in folder {num+1}/{num_dirs}: {dir}")
        dir_name = os.path.join(images_path, dir)
        crop_coords = os.path.join(dir_name, "crop.txt")
        pt.crop_all(dir_name, crop_coords, img_extension, start_img=start_img, end_img=end_img, out_dir=images_path)
    


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
        combined_csv_name = images_path.split("/")[-1] + "_all_plants"
        pt.estimateAll(indirname=os.path.join(images_path, "cropped/"), outdirname=images_path, all_plants_name=combined_csv_name,img_extension=img_extension) # Estimate motion
        end_time = time.time()  # End timer
        total_time = round(end_time - start_time,2)
        print("\nTime to estimate motion: ", total_time, " seconds")
        print("-----------------------------------\n")
    
    if model == True:
        start_time = time.time()
        pt.ModelFitALL(in_dir=images_path, out_dir=images_path) # Fit model to motion data
        end_time = time.time()  # End timer
        total_time = round(end_time - start_time,2)
        print("\nTime to fit model: ", total_time, " seconds")
        print("-----------------------------------")
    
    end_all = time.time()
    total_time_all = round(end_all - start_all,2)
    print("\naccelTRiP execution completed!\n")
    print("Total time: ", total_time_all, " seconds\n\n")


if __name__ == "__main__":
    TRiP()

# Example usage:
# python3 acceltrip.py -d ../../input/ -c ../crop.txt -mt True -m True
