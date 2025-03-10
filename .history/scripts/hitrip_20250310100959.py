from video import generate_video
import pytrip as pt

import argparse
import sys
import time
import os
import zipfile
import re

def natural_sort(l): 
    str_to_int = lambda item: int(item) if item.isdigit() else item.lower()
    output = lambda input: [str_to_int(char) for char in re.split('([0-9]+)', input)]
    return sorted(l, key=output)

def remove_duplicates(a_list):
    remove_duplicates = set()
    remove_duplicates_add = remove_duplicates.add
    return [x for x in a_list if not (x in remove_duplicates or remove_duplicates_add(x))]

# Parser
parser = argparse.ArgumentParser(description='Run accelTRiP on an a folder of images')
parser.add_argument('-d','--img_directory', type=str, required=True, help='Path to images to crop, or to cropped images')
parser.add_argument('-e','--img_extension', type=str, required=False, help='Image extension (e.g. JPG, PNG, TIF)', default="JPG")
parser.add_argument('-c','--combine', type=str, required=False, help='Specify whether to combine numbering of different folders', default="True")
parser.add_argument('-mt','--motion', type=str, required=False, help='Estimate motion', default=None)
parser.add_argument('-m','--model', type=str, required=False, help='Fit model to motion data', default=None)
parser.add_argument('-s','--start_img', type=int, required=False, help='Start image number', default=None)
parser.add_argument('-f','--end_img', type=int, required=False, help='End image number', default=None)
parser.add_argument('-a','--automatic', type=str, required=False, help='Run automatically', default=None)
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

if args.automatic is not None:
    if str(args.automatic) == "True":
        automatic = True
    elif str(args.automatic) == "False":
        automatic = False
    else:  
        print("Parameter automatic must be either True or False")
        sys.exit()
else:
    automatic = False

# Run the functions
def TRiP():
    start_all = time.time()
    # Check if images_path exists
    assert os.path.exists(images_path), "I did not find the folder at, " + str(images_path)

    
    start_time = time.time() # start timer

    dirs = []
    for item in os.listdir(images_path):
        if os.path.isdir(os.path.join(images_path, item)):
            dirs.append(item)
    
    previous_plants = 0
    num_dirs = len(dirs)
    
    sorted_dirs = natural_sort(dirs)
    use_master_crop = "y"
    if use_master_crop == "y":
        print(f"---------------------------------------------------------------------")

        skip_master = False
        prev_master = False
        # Use os.path.join for Windows compatibility
        master_crop_path_global = os.path.join(os.path.dirname(os.path.abspath(__file__)), "master_files", "master_crop_global.txt")
        # Ensure the master_files directory exists
        os.makedirs(os.path.dirname(master_crop_path_global), exist_ok=True)
        
        if os.path.exists(master_crop_path_global):
            user_input_skip_master = ""
            while (user_input_skip_master != "y" and user_input_skip_master != "n"):
                user_input_skip_master = input(f"Master crop file found. Would you like to use this file to generate individual crop files? (Y/N) \n")
                user_input_skip_master = user_input_skip_master.lower().strip()
            if user_input_skip_master == "y":
                prev_master = True
                print(f"Using master crop file to generate individual crop files...")
                

                with open(master_crop_path_global, "r") as f:
                    master_content = f.readlines()
                    for directory in sorted_dirs:
                        crop_path = os.path.join(images_path, directory, "crop.txt")
                        os.makedirs(os.path.dirname(crop_path), exist_ok=True)
                        with open(crop_path, "w+") as f2:
                            for crop_coords in master_content:
                                if crop_coords == "\n":
                                    break
                                f2.write(crop_coords)
                                                
            else:
                skip_master = True
        if not os.path.exists(master_crop_path_global) or skip_master:
            # Use basename instead of split for Windows compatibility
            master_crop_file_name = os.path.basename(images_path) + "_crop.txt"
            master_crop_new_path = os.path.join(images_path, master_crop_file_name)
            print(f"Previous master crop file not found/not in use. A new master crop file has been created in the experiment folder, {images_path}. Please add coordinates to this file.")
            with open(master_crop_new_path, "w+") as f:
                f.write("")
            input("Press Enter to continue once you have added coordinates to the master crop file")
            

            with open(master_crop_new_path, "r") as f:
                    for directory in sorted_dirs:
                        crop_path = os.path.join(images_path, directory, "crop.txt")
                        os.makedirs(os.path.dirname(crop_path), exist_ok=True)
                        with open(crop_path, "w+") as f2:
                            for crop_coords in f:
                                if crop_coords == "\n":
                                    break
                                f2.write(crop_coords)
        unsatisfied = False
        while True:
            print(f"---------------------------------------------------------------------")
            print(f"GENERATING VIDEOS FOR {num_dirs} FOLDERS...")
            for num, dir in enumerate(sorted_dirs):
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
                unsatisfied = True
                continue
            else:
                raise ValueError("Invalid input") 
        master_crop_save = ""
        if not(not unsatisfied and prev_master):
            while (master_crop_save != "y" and master_crop_save != "n"):
                master_crop_save = input("Would you like to save the master crop.txt file globally? (Y/N) \n")
                master_crop_save = master_crop_save.lower().strip()

        else:
            master_crop_save = "n"        
        
        print(f"---------------------------------------------------------------------")
        print(f"GENERATING MASTER crop.txt FILE...")
        # Use basename instead of split for Windows compatibility
        master_crop_file_name = os.path.basename(images_path) + "_crop.txt"
        master_crop_path = os.path.join(images_path, master_crop_file_name)
        master_crop_path_global = os.path.join(os.path.dirname(os.path.abspath(__file__)), "master_files", "master_crop_global.txt")
        
        # Ensure the master_files directory exists
        os.makedirs(os.path.dirname(master_crop_path_global), exist_ok=True)
        
        with open(master_crop_path, "w+") as f:
            for num, dir in enumerate(sorted_dirs):
                dir_name = dir
                dir_name_full = os.path.join(images_path, dir)
                crop_coords = os.path.join(dir_name_full, "crop.txt")
                with open(crop_coords, "r") as crop_file:
                    for line in crop_file:
                        f.write(line)
                f.write("\n")
            
            # Remove last line break - Windows compatible way
            f.seek(0, os.SEEK_END)
            pos = f.tell() - 1
            if pos >= 0:
                f.seek(pos, os.SEEK_SET)
                if f.read(1) == '\n':
                    f.seek(pos, os.SEEK_SET)
                    f.truncate()
            
        if master_crop_save == "y":    
            with open(master_crop_path_global, "w+") as f:
                for num, dir in enumerate(sorted_dirs):
                    dir_name = dir
                    dir_name_full = os.path.join(images_path, dir)
                    crop_coords = os.path.join(dir_name_full, "crop.txt")
                    with open(crop_coords, "r") as crop_file:
                        for line in crop_file:
                            f.write(line)
                    f.write("\n")

                # Remove last line break - Windows compatible way
                f.seek(0, os.SEEK_END)
                pos = f.tell() - 1
                if pos >= 0:
                    f.seek(pos, os.SEEK_SET)
                    if f.read(1) == '\n':
                        f.seek(pos, os.SEEK_SET)
                        f.truncate()
                        
        
    else:
        print(f"---------------------------------------------------------------------")
        print(f"GENERATING crop.txt FILES FOR {num_dirs} FOLDERS...")
        for num, dir in enumerate(sorted_dirs):
            print(f"Generating crop.txt file for folder {num+1}/{num_dirs}: {dir}")
            dir_name = os.path.join(images_path, dir)
            crop_coords = os.path.join(dir_name, "crop.txt")
            os.makedirs(os.path.dirname(crop_coords), exist_ok=True)
            with open(crop_coords, "w+") as f:
                f.write("")
        input("Press Enter to continue once you have added coordinates to all the crop.txt files")
        while True:
            print(f"---------------------------------------------------------------------")
            print(f"GENERATING VIDEOS FOR {num_dirs} FOLDERS...")
            for num, dir in enumerate(sorted_dirs):
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


    
    print(f"----------------------------------------------------------------------")
    print(f"CROPPING IMAGES IN {num_dirs} FOLDERS...")
    for num, dir in enumerate(sorted_dirs):
        print(f"Cropping images in folder {num+1}/{num_dirs}: {dir}")
        dir_name = os.path.join(images_path, dir)
        crop_coords = os.path.join(dir_name, "crop.txt")
        pt.crop_all(dir_name, crop_coords, img_extension, start_img=start_img, end_img=end_img, out_dir=images_path)
    
    if motion == True:
        start_time = time.time() # start timer
        # Use basename instead of split for Windows compatibility
        combined_csv_name = os.path.basename(images_path) + "_all_plants"
        cropped_dir = os.path.join(images_path, "cropped")
        # Ensure cropped directory exists
        os.makedirs(cropped_dir, exist_ok=True)
        pt.estimateAll(indirname=cropped_dir, outdirname=images_path, all_plants_name=combined_csv_name, img_extension=img_extension) # Estimate motion
        end_time = time.time()  # End timer
        total_time = round(end_time - start_time, 2)
        print("\nTime to estimate motion: ", total_time, " seconds")
        print("-----------------------------------\n")
    
    if model == True:
        start_time = time.time()
        pt.ModelFitALL(in_dir=images_path, out_dir=images_path) # Fit model to motion data
        end_time = time.time()  # End timer
        total_time = round(end_time - start_time, 2)
        print("\nTime to fit model: ", total_time, " seconds")
        print("-----------------------------------")
    
    end_all = time.time()
    total_time_all = round(end_all - start_all, 2)
    print("\nhiTRiP execution completed!\n")
    print("Total time: ", total_time_all, " seconds\n\n")


if __name__ == "__main__":
    TRiP()