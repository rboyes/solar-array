import glob
import numpy as np
import pandas as pd
import os
import sys
from PIL import Image
from keras.preprocessing.image import img_to_array, array_to_img
import scipy.ndimage.morphology as morph
from skimage.morphology import remove_small_objects

def imagepair2mask(image_array, mask_array):

    if image_array.shape != mask_array.shape:
        raise ValueError("Input image and mask sizes don't match")    

    # Check if mask is an overlay, otherwise is a binary image
    is_overlay = len(np.unique(np.asarray(mask_array))) > 2
    
    if is_overlay:        
        flat_input_mask = np.mean(mask_array - image_array, axis=-1)
        flat_index = flat_input_mask > 1.0E-2
        flat_input_mask[:] = 0.0
        flat_input_mask[flat_index] = 1.0
        mask = np.expand_dims(morph.binary_fill_holes(flat_input_mask.astype(int)),2)
        mask = remove_small_objects(mask, min_size=16)
        mask = array_to_img(mask)
    else:
        flat_input_mask = np.mean(mask_array, axis=-1)
        mask = array_to_img(np.expand_dims(flat_input_mask, 2))    
    
    return mask

if __name__ == '__main__':

    import argparse
    parser = argparse.ArgumentParser(description="Batch process images to create images and masks", 
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("-i", "--input", help="input folder of images", required = True)
    parser.add_argument("-z", "--imasks", help="input folder of masks", required = True)
    
    parser.add_argument("-o", "--images", help="output folder for converted image files", required = True)
    parser.add_argument("-m", "--masks", help="output folder for converted masks", required = True)
    parser.add_argument("-c", "--csv", help = "output listing of all files", required=True)
    parser.add_argument("-s", "--size", help = "output size of files", required=False, type=int, nargs='+', default=[256,256])
    parser.add_argument("-t", "--step", help="increment between output images", required = False, type=int, default=128)
    
    
    args = parser.parse_args()
    
    image_paths = glob.glob(os.path.join(args.input, "*.png")) 
    image_paths.extend(glob.glob(os.path.join(args.input, "*.jpg")))
    mask_paths = glob.glob(os.path.join(args.imasks, "*.png"))
    mask_paths.extend(glob.glob(os.path.join(args.imasks, "*.jpg")))
    
    name_path_lookup = {}
    for image_path in image_paths:
        name_path_lookup[os.path.basename(os.path.splitext(image_path)[0])] = image_path
    
    
    nrow, ncol = args.size[0], args.size[1]
    step = args.step
    image_data = []
    
    for input_mask_path in mask_paths:
        print("Working on %s" % (input_mask_path))
        input_mask = Image.open(input_mask_path).convert('RGB')
        
        input_mask_name = os.path.basename(os.path.splitext(input_mask_path)[0])
        image_path = name_path_lookup[input_mask_name]
        image = Image.open(image_path).convert('RGB')
        image_name = os.path.basename(os.path.splitext(image_path)[0])
        
        if image_name != input_mask_name:
            raise ValueError("Input image and mask names don't match, %s %s" % (image_name, input_mask_name))
        
        if image.size != input_mask.size:
            raise ValueError("Input image and mask sizes don't match")
        input_mask_array = img_to_array(input_mask)/255.0
        image_array = img_to_array(image)/255.0
        
        input_mask_array = input_mask_array[:,:,:3]
        image_array = image_array[:,:,:3]      
        
        mask = imagepair2mask(image_array, input_mask_array)
                        
        index = 0
        image_width, image_height = image.size
        
        row_range = np.arange(0, image_height - nrow, step)        
        if row_range[-1] < (image_height - nrow):        
            row_range = np.append(row_range, image_height - nrow)
            
        col_range = np.arange(0, image_width - ncol, step)        
        if col_range[-1] < (image_width - ncol):
            col_range = np.append(col_range, image_width - ncol)
        
        for row in row_range:
            for col in col_range:
                                
                image_path = os.path.abspath(os.path.join(args.images, image_name + "_%06d.jpg" % index))
                mask_path = os.path.abspath(os.path.join(args.masks, image_name + "_%06d.png" % index))
                
                cropped_image = image.crop((col, row, col + ncol, row + nrow))
                cropped_mask = mask.crop((col, row, col + ncol, row + nrow))
                
                cropped_image.save(image_path, "JPEG", quality = 85)
                cropped_mask.save(mask_path, "PNG")
                
                segmented = int(np.max(np.array(cropped_mask)) > 0)
                image_data.append([image_path,mask_path,segmented])
                
                index += 1

    df = pd.DataFrame(image_data)
    df.to_csv(args.csv, index=False, header=False)