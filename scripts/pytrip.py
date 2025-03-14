import numpy as np
import pandas as pd
from PIL import Image
from pathlib import Path
import matplotlib as mpl
import matplotlib.pyplot as plt
import os
from scipy.signal import convolve, convolve2d, detrend
from scipy.optimize import minimize, curve_fit
import cv2
import glob
import sys
import time


#Define blank object for csv formatting
class BlankObj:
    def __repr__(self):
        return ""


# Crop images
def crop_all(images_path, cropfile_path, img_extension, start_img, end_img, out_dir):
    print("\n-----------------------------------")
    print("CROPPING IMAGES...\n")
    img_extension = '*.' + img_extension
    images = sorted([str(p) for p in Path(images_path).glob(img_extension)])
    # Check if images were found
    assert len(images) > 0, f"No images found in {images_path}"

    if start_img is not None and end_img is not None:
        images = images[start_img:end_img]

    # Read crop file
    crop_coords = pd.read_csv(cropfile_path, sep='\t', header=None)
    # check that the crop file has only one column
    assert crop_coords.shape[1] == 1, "Crop file must have only one column"
    # check that the crop file has at least one row
    assert crop_coords.shape[0] > 0, "Crop file must have at least one row"
    # check that all rows have 5 elements separated by a single space
    assert all(len(i.split(" ")) == 5 for i in crop_coords.iloc[:,0]), "All rows must have 5 elements separated by a single space"

    # Create empty lists for ID (key) and value (coordinates)
    keys = []
    values = []

    # Gather IDs and coordinates
    for plant in range(len(crop_coords)):
        # Get ID (path to each 'cropped' folder)
        plant_ID = crop_coords.iloc[plant,0] 
        plant_ID = plant_ID.split(" ")[0] # Split by " "; take the first
        keys.append(plant_ID) # Add this to the 'keys' list
        # Get coordinates
        coords = crop_coords.iloc[plant,0]
        coords = coords.split(" ")[1:]  # This must change if using \t sep!!
        coords = [int(i) for i in coords] # Convert values to integers
        values.append(tuple(coords))  # Append coords as tuple

    # Arrange keys and values in a dictionary
    regions = dict(zip(keys, values))

    ti = len(images)  # total images (ti)
    ci = 0  # current image (ci)

    # Loop thorugh each original image
    for i in images:
        # Current image
        ci = ci+1
        print(f'Processing image {ci}/{ti}: {i}')
        # Read image and convert to array
        image = Image.open(i)
        image = np.array(image)
        # Extract and save the cropped images (assuming imagej's format)
        for plant, coords in regions.items():
            imj_col1 = coords[0] # vertical left
            imj_row1 = coords[1] # hortizontal bottom
            imj_col2 = imj_col1 + coords[2] # vertical right
            imj_row2 = imj_row1 + coords[3] # horizontal top
            # Verify path exists, otherwise create folder
            cropped_path = os.path.join(out_dir, 'cropped', plant)  # path
            if not os.path.exists(cropped_path):  # verify if exist
                os.makedirs(cropped_path)
            # Define cropped image's full path
            # Use os.path.basename to extract file name properly on all platforms
            cropped_path = os.path.join(cropped_path, 'crop_' + os.path.basename(i))
            # Crop image
            cropped_image = image[imj_row1:imj_row2,imj_col1:imj_col2,:] # crop
            sub_image = Image.fromarray(cropped_image)  # Convert to PIL format
            sub_image.save(cropped_path) # Save image

    # Print where images were saved
    print(f"\nAll folders were saved in: {os.path.join(out_dir, 'cropped')}")


# Estimate derivatives
def space_time_deriv(subset_f):
    # N = f.shape[0]    
    # dims = f[0].shape
    N = len(subset_f)
    dims = subset_f[0]['im'].shape
    
    if N == 1:
        # Handle case when N = 1
        fx = np.zeros(dims)
        fy = np.zeros(dims)
        ft = np.zeros(dims)
        return None, None, None
    
    # Define derivative kernels
    if N == 2:
        pre = np.array([0.5, 0.5])
        deriv = np.array([-1, 1])
    elif N == 3:
        pre = np.array([0.223755, 0.552490, 0.223755])
        deriv = np.array([-0.453014, 0.0, 0.453014])
    elif N == 4:
        pre = np.array([0.092645, 0.407355, 0.407355, 0.092645])
        deriv = np.array([-0.236506, -0.267576, 0.267576, 0.236506])
    elif N == 5:
        pre = np.array([0.036420, 0.248972, 0.429217, 0.248972, 0.036420])
        deriv = np.array([-0.108415, -0.280353, 0.0, 0.280353, 0.108415])
    elif N == 6:
        pre = np.array([0.013846, 0.135816, 0.350337, 0.350337, 0.135816, 0.013846])
        deriv = np.array([-0.046266, -0.203121, -0.158152, 0.158152, 0.203121, 0.046266])
    elif N == 7:
        pre = np.array([0.005165, 0.068654, 0.244794, 0.362775, 0.244794, 0.068654, 0.005165])
        deriv = np.array([-0.018855, -0.123711, -0.195900, 0.0, 0.195900, 0.123711, 0.018855])
    else:
        raise Warning(f'No such filter size (N={N})')
        
    pre = [round(element,4) for element in pre]
    deriv = [round(element,4) for element in deriv]
    
    # SPACE/TIME DERIVATIVES
    fdt = np.zeros(dims)
    fpt = np.zeros(dims)
    for i in range(N):
        fpt = fpt + (pre[i] * subset_f[i]['im'])
        fdt = fdt + (deriv[i] * subset_f[i]['im'])
    
    # Reshape the filters to 2D arrays
    pre_2d = np.reshape(pre, (1, -1))
    deriv_2d = np.reshape(deriv, (-1, 1))

    # Perform the convolutions
    fx = convolve2d(fpt, pre_2d.T, mode='same')
    fx = convolve2d(fx, deriv_2d.T, mode='same')
    fy = convolve2d(fpt, pre_2d, mode='same')
    fy = convolve2d(fy, deriv_2d, mode='same')
    ft = convolve2d(fdt, pre_2d.T, mode='same')
    ft = convolve2d(ft, pre_2d, mode='same')
    
    return fx, fy, ft



# Estimate motion
def estimate_motion(dirname, img_extension='JPG'):

    GRADIENT_THRESHOLD = 8

    # load frames
    frames = []
    ext = img_extension.lower()  # Normalize extension to lowercase
    
    # Use os.listdir and filter instead of glob for better cross-platform compatibility
    d = [filename for filename in os.listdir(dirname) 
         if filename.lower().endswith(ext)]
    d = sorted(d)
    N = len(d)
    c = 1
    f = []

    for k in range(1, N+1):
        im = cv2.imread(os.path.join(dirname, d[k-1]))
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        if k == 1:
            scale = round(60 / max(im.shape), 4) # round to match Matlab output
        im = cv2.resize(im, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        f.append({})
        f[c-1]['orig'] = im

        # [0.2989, 0.5870, 0.1140] is a standard basis for RGB to grayscale conversion
        f[c-1]['im'] = np.dot(im[...,:3], [0.2989, 0.5870, 0.1140]).astype(np.float64)

        c += 1
    
    ydim, xdim = f[0]['im'].shape
    
    # Define the number of taps
    taps = 7

    # Define the Gaussian blur kernel and normalize and reshape it
    blur = [1, 6, 15, 20, 15, 6, 1]
    blur = np.array(blur) / np.sum(blur)
    blur = blur.reshape(1, -1)
    
    s = 1 # sub-sample spatially by this amount

    #This is the number of frames that can be used to estimate the motion, where we lose 7 frames
    N = len(f) - (taps-1)

    # // is floor division, does not do anything here since s = 1
    Vx = np.zeros((ydim//s, xdim//s, N))
    Vy = np.zeros((ydim//s, xdim//s, N))
    
    # Loop through the subsets of frames, 7 at a time, computing optical flow
    for k in range(N):
        subset_f = f[k:k+taps]
        if len(subset_f) < 1:
            continue
        
        fx, fy, ft = space_time_deriv(subset_f)

        if any(var is not None for var in [fx,fx,ft]):
            fx2 = convolve2d(convolve2d(fx*fx, blur.T, mode='same'),
                             blur, mode='same')
            fy2 = convolve2d(convolve2d(fy*fy, blur.T, mode='same'),
                             blur, mode='same')
            fxy = convolve2d(convolve2d(fx*fy, blur.T, mode='same'),
                             blur, mode='same')
            fxt = convolve2d(convolve2d(fx*ft, blur.T, mode='same'),
                             blur, mode='same')
            fyt = convolve2d(convolve2d(fy*ft, blur.T, mode='same'),
                             blur, mode='same')

            grad = np.sqrt(np.power(fx, 2) + np.power(fy, 2))
            # Set the specified regions to zero
            grad[:, :5] = 0
            grad[:5, :] = 0
            grad[:, -5:] = 0
            grad[-5:, :] = 0

            # Compute optical flow
            cx = 0
            bad = 0
            for x in range(0, xdim, s):
                cy = 0
                for y in range(0, ydim, s):
                    M = np.array([[fx2[y, x], fxy[y, x]], [fxy[y, x], fy2[y, x]]])
                    b = np.array([fxt[y, x], fyt[y, x]])
                    if np.linalg.cond(M) > 1e2 or grad[y, x] < GRADIENT_THRESHOLD:
                        Vx[cy, cx, k] = 0
                        Vy[cy, cx, k] = 0
                        bad += 1
                    else:
                        v = np.linalg.inv(M) @ b
                        Vx[cy, cx, k] = v[0]
                        Vy[cy, cx, k] = v[1]
                    cy += 1
                cx += 1

            # check if bad / (xdim * ydim) == 1. If so, exit this function 
            if bad / (xdim * ydim) == 1:
                return None, None
    
    
    # visualize motion field
    taps = 13
    blur = np.ones(taps)
    blur = blur / np.sum(blur)
    
    
    c = 0
    motion_x = []
    motion_y = []
    
    eps = 2.2204e-16

    for k in range(N - taps):
        vx = np.zeros(Vx.shape[:2])
        vy = np.zeros(Vy.shape[:2])
        Vx2 = Vx[:, :, k:k+taps]
        Vy2 = Vy[:, :, k:k+taps]

        for j in range(len(blur)):
            vx += blur[j] * Vx2[:, :, j]
            vy += blur[j] * Vy2[:, :, j]

        indx = np.where(np.abs(vx) > eps)
        indy = np.where(np.abs(vy) > eps)

        motion_x.append(1 / scale * np.mean(vx[indx]))
        motion_y.append(-1 / scale * np.mean(vy[indy]))

        c += 1
    
    return motion_x, motion_y


# Estimate motion for all
def estimateAll(indirname, outdirname, all_plants_name, img_extension='JPG'):
    print("\n-----------------------------------")
    print("ESTIMATING MOTION...\n")   

    # Create output directories
    motion_output_dir = os.path.join(outdirname, 'output', 'motion')
    Path(indirname).mkdir(parents=True, exist_ok=True)
    Path(motion_output_dir).mkdir(parents=True, exist_ok=True)

    # Get a list of cropped image subdirectories
    d = [f.path for f in os.scandir(indirname) if f.is_dir()]
    d = sorted(d)

    all_data = pd.DataFrame()
    time_column = pd.DataFrame([], columns=["Time"])
    all_data = pd.concat([all_data, time_column], axis=1)

    for k in d:
        # Get plant ID using os.path.basename for cross-platform compatibility
        plantID = os.path.basename(k).replace('.csv','')
        # Estimate motion
        motion_x, motion_y = estimate_motion(k, img_extension=img_extension)
        # Check if motion was estimated
        if motion_x is None or motion_y is None:
            print(f"ERROR: Could not estimate motion for {plantID}")
            all_data = pd.concat([all_data, pd.DataFrame([BlankObj(),BlankObj()],columns=[f"{plantID}"])], axis=1)
            continue
        
        # Save vertical motion
        motion_y.insert(0, BlankObj())
        motion_y.insert(0, BlankObj())
        df = pd.DataFrame(motion_y, columns=[f"{plantID}"])
        all_data = pd.concat([all_data, df], axis=1)

        # Save CSV file using os.path.join
        output_file = os.path.join(motion_output_dir, f'{plantID}.csv')
        df.to_csv(output_file, index=False, header=True, na_rep='inf')

        print(f"Estimated motion for {k}")
    
    # Save all data using os.path.join
    all_plants_file = os.path.join(motion_output_dir, f'{all_plants_name}.csv')
    all_data.to_csv(all_plants_file, index=False, header=True, na_rep='inf')


# Evaluate model
def evaluateModel(model, N):
    '''
    This function generates a sinusoid of length N 
    for a specified frequency, phase, and amplitude.
    '''
    freq=model[0]; phase=model[1]; amp=model[2]
    t = np.arange(N)
    f = amp * np.cos(freq * 2 * np.pi / len(t) * t + phase)

    return f


# Error function
def errorFunc(model, dat):
    '''
    compute the RMS error between the current model and the data. 
    This is used by the non-linear optimization in modelFit.m
    '''
    N = len(dat)
    f = evaluateModel(model, N)
    err = np.sum((f - dat) ** 2)
    return err


# Model fit
def modelFunc(t, freq, phase, amp):
    return evaluateModel([freq, phase, amp], N=len(t))

# Jacobian
def jacFunc(t, freq, phase, amp):
    dfreq = -amp * np.sin(freq * 2 * np.pi / len(t) * t + phase) * 2 * np.pi / len(t) * t
    dphase = -amp * np.sin(freq * 2 * np.pi / len(t) * t + phase)
    damp = np.cos(freq * 2 * np.pi / len(t) * t + phase)
    return np.column_stack((dfreq, dphase, damp))

# Fit model to motion data
def ModelFitALL(in_dir, out_dir):
    print("\n-----------------------------------")
    print("FITTING MODEL TO MOTION DATA...\n")

    # Use Path for cross-platform glob pattern
    motion_dir = os.path.join(in_dir, "output", "motion")
    model_output_dir = os.path.join(out_dir, 'output', 'model')
    
    # Use pathlib for safer cross-platform globbing
    d = list(Path(motion_dir).glob("*.csv"))
    d = sorted([str(path) for path in d])

    # Create output directory if it doesn't exist
    Path(model_output_dir).mkdir(parents=True, exist_ok=True)

    Path_Array = []
    Period_Array = []
    CTP_Array = []
    rsq_Array = []
    rae_Array = []

    for k in d:
        # Get plant ID using os.path.basename for cross-platform compatibility
        fn = k
        plantID = os.path.basename(k).replace('.csv','')
        
        if "all_plants" in plantID:
            continue
        
        dat = pd.read_csv(fn, header=0)
        # Check if all values are the same
        if len(dat.iloc[:,0].unique()) == 1:
            print(f"All values are the same for {plantID}. No model will be fitted for this plant.")
            continue

        dat.replace([np.inf, -np.inf], np.nan, inplace=True) # replace inf
        dat = dat.fillna(0) # Fill NA with zeros
        dat = dat - dat.mean()
        dat = (dat - detrend(dat, type='linear'))
        dat = np.array(dat.squeeze())
        N = len(dat)

        # compute dominant frequency and phase for starting condition
        D = np.fft.fftshift(np.fft.fft(dat))

        if len(dat) % 2 == 0:
            mid = len(dat) // 2
        else:
            mid = len(dat) // 2

        D = D[mid:mid+11]  # Assumes that the dominant frequency is less than or equal to 10
        ind = np.argmax(np.abs(D))
        freq = ind 
        phase = np.angle(D[ind])  # Starting condition
        amp = np.mean(np.abs(dat))  # Starting condition

        initial_model = [freq, phase, amp]

        # Non-linear fitting of frequency, phase, and amplitude
        model = minimize(errorFunc, initial_model, args=(dat,), method='Nelder-Mead').x

        # Plot results
        f = evaluateModel(model, N)
        plt.figure()
        plt.plot(dat, 'b', linewidth=1)
        plt.plot(f, 'r', linewidth=1)

        if np.count_nonzero(np.logical_not(model)) > 2:
            plt.axis([0, N-1, -1, 1])
        else:
            plt.axis([0, N-1, np.min(dat), np.max(dat)])

        plt.legend(['Data', 'Model'])
        plt.title('Frequency = {}'.format(round(model[0],2)))
        
        # Save the figure using os.path.join for the path
        output_plot = os.path.join(model_output_dir, f'model_{plantID}.png')
        plt.savefig(output_plot, bbox_inches='tight', facecolor='w')
        plt.close()

        freq = model[0]
        Period = N / freq
        Period = Period / 3
        phase = model[1]
        Pjust = 24 / Period
        phi_ang = phase / Pjust
        phi = phi_ang / np.pi
        phi = phi * 12

        if phi < 0:
            CTP = (abs(phi) * 24) / Period
        else:
            CTP = 24 - (phi * 24) / Period

        # define t and model
        t = np.arange(N)   
        model = [freq, phase, amp]

        # Fit the model to the data using curve_fit
        beta, cov = curve_fit(modelFunc, t, dat, p0=model, jac=jacFunc)

        # Calculate the residuals and the coefficient of determination
        fittedData = evaluateModel(beta, N)
        residuals = dat - fittedData
        rsq = 1 - np.sum(residuals ** 2) / np.sum((dat - np.mean(dat)) ** 2)

        # Calculate the confidence interval for frequency (CI_freq)
        ci_freq = np.sqrt(cov[0, 0])
        CI_freq = ci_freq * Period / beta[0]

        # Calculate the confidence interval for the amplitude (CI)
        ci_amp = np.sqrt(cov[2, 2]) / 2
        AMP = beta[2]
        RAE = ci_amp / AMP

        output_values = [Period, CTP, rsq, RAE]
        output_values = [round(num,2) for num in output_values]

        Path_Array.append(plantID)
        Period_Array.append(Period)
        CTP_Array.append(CTP)
        rsq_Array.append(rsq)
        rae_Array.append(RAE)

        print(f"Fitted model for {plantID}")

    Models_data = pd.DataFrame(
        {'ID': Path_Array,
         'Period': Period_Array,
         'CTP': CTP_Array,
         'Rsquared': rsq_Array,
         'RAE': rae_Array,   
        })

    # Save to CSV using os.path.join
    output_file = os.path.join(model_output_dir, 'MODELS_DATA.csv')
    Models_data.to_csv(output_file, index=False)