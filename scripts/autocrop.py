import numpy as np
import cv2

#STL Imports
import zipfile
from functools import cmp_to_key

def autocrop(file_path, num_objects, tolerance=0, video_name="../out/out_video.mp4"):
    #Currently outputs video with bounding rectangles for demonstrative purposes
  
    #Helper function for autocropping to convert subset of images to video
    #@param file_path: Path to the zip file containing the images
    #@param num_objects Number of objects in frame
    #@param tolerance Integer value specifying tolerance in pixels around each object
    #@return List of tuples containing the bounding rectangles of the objects in the final frame

    #Read in the zip file and select up to 10 evenly spaced frames
    zipped_dir = zipfile.ZipFile(file_path)
    image_list = zipped_dir.infolist()

    with zipped_dir.open(image_list[0].filename) as image:
        height, width= cv2.imdecode(np.frombuffer(image.read(), dtype=np.uint8),cv2.IMREAD_UNCHANGED).shape[:2]
    
    # Uncomment to write video
    video = cv2.VideoWriter(video_name, cv2.VideoWriter_fourcc(*"mp4v"), 20.0, (width, height))
    
    sorted_images = [index.filename for index in image_list]
    sorted_images.sort()

    backSub = cv2.createBackgroundSubtractorMOG2()

    sorted_images = sorted_images[int(0.1*len(sorted_images)):]
    num_frames = len(sorted_images)
    all_rectangles = np.zeros((4, num_frames, num_objects))

    for i, file in enumerate(sorted_images):
        with zipped_dir.open(file) as image:
            frame = cv2.imdecode(np.frombuffer(image.read(), dtype=np.uint8),cv2.IMREAD_UNCHANGED)

            fg_mask = backSub.apply(frame)
            retval, mask_thresh = cv2.threshold( fg_mask, 180, 255, cv2.THRESH_BINARY)
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            mask_eroded = cv2.morphologyEx(mask_thresh, cv2.MORPH_OPEN, kernel)

            contours, hierarchy = cv2.findContours(mask_eroded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            mask = np.zeros(frame.shape[:2],dtype=np.uint8)
            for cnt in contours:
                x, y, w, h = cv2.boundingRect(cnt)
                cv2.rectangle(mask, (x, y), (x+w, y+h), (255), -1)
        
            real_contours,hierarchy=cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
            largest_contours = sorted(real_contours, key=cmp_to_key(compare))[:num_objects]
            
            if len(largest_contours) < num_objects:
                continue

            for j, cnt in enumerate(largest_contours):
                x, y, w, h = cv2.boundingRect(cnt)
                all_rectangles[:,i, j] = cv2.boundingRect(cnt)
    
    final_mask = np.zeros(frame.shape[:2],dtype=np.uint8)
    combined_rectangles = all_rectangles.T.reshape(-1,4)

    for rectangle in combined_rectangles:
        if rectangle[2] == 0 or rectangle[3] == 0:
            continue
        x, y, w, h = tuple(rectangle)
        cv2.rectangle(final_mask, (int(x - tolerance), int(y - tolerance)), (int(x+w+tolerance), int(y+h+tolerance)), (255), -1)
    
    final_contours,hierarchy=cv2.findContours(final_mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

    #Removed [:num_objects] at end of line below
    largest_final_contours = sorted(final_contours, key=cmp_to_key(compare))

    # Uncomment to write video
    for file in sorted_images:
        with zipped_dir.open(file) as image:
            frame = cv2.imdecode(np.frombuffer(image.read(), dtype=np.uint8),cv2.IMREAD_UNCHANGED)
            frame_out = frame.copy()
            for cnt in largest_final_contours:
                x, y, w, h = cv2.boundingRect(cnt)
                frame_out = cv2.rectangle(frame_out, (x - tolerance, y - tolerance), (x+w+tolerance, y+h+tolerance), (0, 0, 200), 3)
                frame_out = cv2.putText(frame_out, 'Fedex', (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 200), 2)
            video.write(frame_out)
    
    # Uncomment to write video
    video.release()
    
    cv2.destroyAllWindows()
    return [cv2.boundingRect(cnt) for cnt in largest_final_contours]

def compare(cnt1, cnt2):
    #Helper function for comparing contours by area
    #@param cnt1: First contour
    #@param cnt2: Second contour
    #@return Difference in area between the two contours
