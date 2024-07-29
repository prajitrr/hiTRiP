import numpy as np
import cv2

#STL Imports
import zipfile
from functools import cmp_to_key

image_formats = ['jpg', 'jpeg', 'png', 'bmp', 'tiff', 'tif', 'gif', 'webp', 'tga', 'exr', 'hdr', 'pic', 'jp2', 'j2k', 'j2c', 'jpe', 'jp2', 'jif', 'jfi', 'jfif', 'heic', 'heif']

def autocrop(file_path, num_objects, prev_objects=0, tolerance=5, grid_height=3, grid_width=4, video_name="../test/out_video.mp4",):
    #Currently outputs video with bounding rectangles for demonstrative purposes
  
    #Helper function for autocropping to convert subset of images to video
    #@param file_path: Path to the zip file containing the images
    #@param num_objects Number of objects in frame
    #@param tolerance Integer value specifying tolerance in pixels around each object
    #@return List of tuples containing the bounding rectangles of the objects in the final frame

    #Read in the zip file and select up to 10 evenly spaced frames
    zipped_dir = zipfile.ZipFile(file_path)
    image_list = zipped_dir.infolist()
    
    #Filter out non-image files
    sorted_images = [index.filename for index in image_list if "__MACOSX/" not in index.filename]
    sorted_images = [index for index in sorted_images if index.split('.')[-1].lower() in image_formats]
    sorted_images.sort()

    with zipped_dir.open(sorted_images[0]) as image:
        height, width= cv2.imdecode(np.frombuffer(image.read(), dtype=np.uint8),cv2.IMREAD_UNCHANGED).shape[:2]
    
    video = cv2.VideoWriter(video_name, cv2.VideoWriter_fourcc(*"mp4v"), 20.0, (width, height))
    backSub = cv2.createBackgroundSubtractorMOG2()

    #Removed -int(0.1*len(sorted_images)) at end of bottom line
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
            largest_contours = sorted(real_contours, key=cmp_to_key(compare_contours))[:num_objects]
            
            if len(largest_contours) < num_objects:
                continue

            for j, cnt in enumerate(largest_contours):
                x, y, w, h = cv2.boundingRect(cnt)
                all_rectangles[:,i, j] = cv2.boundingRect(cnt)
    
    areas = np.multiply(all_rectangles[2,:,:], all_rectangles[3,:,:])
    areas = np.sum(areas, axis=1)
    kept_frames = filter_outliers(areas)
    all_rectangles = all_rectangles[:, kept_frames, :]

    final_mask = np.zeros(frame.shape[:2],dtype=np.uint8)
    combined_rectangles = all_rectangles.T.reshape(-1,4)

    for rectangle in combined_rectangles:
        if rectangle[2] == 0 or rectangle[3] == 0:
            continue
        x, y, w, h = tuple(rectangle)
        cv2.rectangle(final_mask, (int(x - tolerance), int(y - tolerance)), (int(x+w+tolerance), int(y+h+tolerance)), (255), -1)
    
    final_contours,hierarchy=cv2.findContours(final_mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    final_rectangles = [cv2.boundingRect(contour) for contour in final_contours]

    if len(final_rectangles) > num_objects:
        mse_changes = np.zeros(len(final_rectangles), dtype=int)
        with zipped_dir.open(sorted_images[0]) as first_image, zipped_dir.open(sorted_images[-1]) as last_image, zipped_dir.open(sorted_images[int(num_frames/2)]) as middle_image:
            first_frame = cv2.imdecode(np.frombuffer(first_image.read(), dtype=np.uint8),cv2.IMREAD_UNCHANGED)
            middle_frame = cv2.imdecode(np.frombuffer(middle_image.read(), dtype=np.uint8),cv2.IMREAD_UNCHANGED)
            last_frame = cv2.imdecode(np.frombuffer(last_image.read(), dtype=np.uint8),cv2.IMREAD_UNCHANGED)
            for j, rect in enumerate(final_rectangles):
                x, y, w, h = rect
                mse_changes[j] = mse(first_frame[y:y+h, x:x+w, :], last_frame[y:y+h, x:x+w, :]) + mse(first_frame[y:y+h, x:x+w, :], middle_frame[y:y+h, x:x+w, :]) + mse(middle_frame[y:y+h, x:x+w, :], last_frame[y:y+h, x:x+w, :])
        final_rectangles = [rect for _, rect in sorted(zip(mse_changes, final_rectangles))][-num_objects:]
    else:
        pass

    final_points = np.array([(rect[0] + rect[2]/2, rect[1] + rect[3]/2) for rect in final_rectangles], dtype=np.float16)
    sorted_final_rectangles = [final_rectangles[i] for i in order_points(final_points, grid_height=grid_height, grid_width=grid_width)]

    # Uncomment to write video
    for file in sorted_images:
        with zipped_dir.open(file) as image:
            frame = cv2.imdecode(np.frombuffer(image.read(), dtype=np.uint8),cv2.IMREAD_UNCHANGED)
            frame_out = frame.copy()
            for num, rect in enumerate(sorted_final_rectangles):
                number = num + 1
                x, y, w, h = rect
                frame_out = cv2.rectangle(frame_out, (x - tolerance, y - tolerance), (x+w+tolerance, y+h+tolerance), (0, 0, 200), 3)
                frame_out = cv2.putText(frame_out, f'plant_A{number:02}', (x - 2, y + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
            video.write(frame_out)
    
    video.release()
    
    cv2.destroyAllWindows()
    return final_rectangles

def compare_contours(cnt1, cnt2):
    #Helper function for comparing contours by area
    #@param cnt1: First contour
    #@param cnt2: Second contour
    #@return Difference in area between the two contours
    return cv2.contourArea(cnt2) - cv2.contourArea(cnt1)

def filter_outliers(array, m=3.5):
    mad = np.abs(array - np.median(array))
    mdev = np.median(mad)
    s = mad/mdev if mdev else np.zeros(len(mad))
    return np.where(s < m)


def order_points(array, g_height, g_width):
    grid_width = 4
    sorted_y_ind = np.argsort(array[:,1])
    sorted_y = np.concatenate((array[sorted_y_ind], sorted_y_ind[..., None]), axis=1)
    for i in range (0, len(sorted_y), grid_width):
        sorted_x_ind = np.argsort(sorted_y[i:i+grid_width,0])
        sorted_y[i:i+grid_width] = sorted_y[i:i+grid_width][sorted_x_ind]
    result = sorted_y[:,2].tolist()
    result = [int(i) for i in result]
    return result
    
def mse(img1, img2):
    h, w, c = img1.shape
    diff = cv2.subtract(img1, img2)
    err = np.sum(diff**2)
    mse = err/(float(h*w))
    return mse

# def order_points(array, grid_size):
#     remaining_indices = np.arange(len(array))
#     remaining_points = array
#     print(array)
#     #int(-1 * grid_size[1])
#     pts_to_find = -4
#     sorted_indices = []
#     while len(remaining_points) > 0:
#         #top_left = np.argmin(np.sum(remaining_points, axis=1), axis=0)
#         #top_right = (-1 * np.diff(remaining_points, axis=1)).argmax(axis=0)

#         top_left = np.argmin(-1 * np.diff(remaining_points, axis=1), axis=0)
#         top_right = (np.sum(remaining_points, axis=1)).argmax(axis=0)
#         top_left_pt = remaining_points[top_left]
#         top_right_pt = remaining_points[top_right]
#         distances = np.cross(top_right_pt-top_left_pt, top_left_pt-remaining_points)/(np.linalg.norm(top_right_pt-top_left_pt) + 0.0000000001)
        
#         if len(remaining_points) >= np.abs(pts_to_find):
#             found = np.argpartition(distances, pts_to_find)[pts_to_find:]
#         else:
#             found = np.argsort(distances)

#         found_indices = remaining_indices[found.tolist()]
#         found_pts = remaining_points[found.tolist()]
#         found_x = [point[0] for point in found_pts]

#         found_sorted_indices = [index for _, index in sorted(zip(found_x, found_indices.tolist()))]
        
#         remaining_points = np.delete(remaining_points, found.tolist(), axis=0)
#         remaining_indices = np.delete(remaining_indices, found.tolist(), axis=0)
#         sorted_indices = sorted_indices + found_sorted_indices
#         sorted_indices = [int(i) for i in sorted_indices]
#     return sorted_indices

    

if __name__ == "__main__":
    autocrop("../test/trip_test_12.zip", 12, 0, 10, "../test/out_video.mp4")

