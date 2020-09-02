import numpy as np
import pandas as pd
from PIL import Image

from extract_image_masks import imagepair2mask
from train import overlap, overlap_loss, bce_overlap_loss
from train import preprocess_image, get_model_type
from mask_image import mask_image, tta, flip_tta

from keras.preprocessing.image import load_img, img_to_array, array_to_img
from keras.models import load_model

import skimage.segmentation
from skimage.measure import label

def confusion_terms(true_labels, eval_labels, threshold = 0.2):
    
    ntrue = len(np.unique(true_labels))
    neval = len(np.unique(eval_labels))
    intersection = np.histogram2d(true_labels.flatten(), eval_labels.flatten(), bins=(ntrue, neval))[0]
    area_true = np.histogram(true_labels, bins = ntrue)[0]
    area_eval = np.histogram(eval_labels, bins = neval)[0]
    area_true = np.expand_dims(area_true, -1)
    area_eval = np.expand_dims(area_eval, 0)
    union = area_true + area_eval - intersection
    intersection = intersection[1:,1:]
    union = union[1:,1:]
    union[union == 0] = 1e-9
    iou = intersection / union
    matches = iou > threshold
    true_positives = np.sum(matches, axis=1) == 1   # Correct objects
    false_positives = np.sum(matches, axis=0) == 0  # Missed objects
    false_negatives = np.sum(matches, axis=1) == 0  # Extra objects
    tp, fp, fn = np.sum(true_positives), np.sum(false_positives), np.sum(false_negatives)
    
    return tp, fp, fn
    
if __name__ == '__main__':

    import argparse
    parser = argparse.ArgumentParser(description="Batch process images to create images and masks", 
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("-m", "--model", help="input model", required = True)
    parser.add_argument("-v", "--validation", help="list of input images/masks", required = True)
    parser.add_argument("-t", "--threshold", help="IOU threshold", required = False,  type=float, default=0.5)
    parser.add_argument("--tta", help="test time augmentation", required = False,  action = 'store_true', default=False)
    parser.add_argument("--ftta", help="flip test time augmentation", required = False,  action = 'store_true', default=False)    

    args = parser.parse_args()
    
    validation_df = pd.read_csv(args.validation, header = None)
    image_paths, mask_paths = validation_df[0].tolist(), validation_df[1].tolist()
    
    model = load_model(args.model, custom_objects={'overlap_loss' : overlap_loss,
                                                   'overlap' : overlap,
                                                   'bce_overlap_loss' : bce_overlap_loss})
    ttp, tfp, tfn = 0, 0, 0
    
    for image_path, mask_path in zip(image_paths, mask_paths):
        image_array = img_to_array(load_img(image_path))
        mask_array = img_to_array(Image.open(mask_path).convert('RGB'))/255.0
        
        mask_array = mask_array[:,:,:3]
        image_array = image_array[:,:,:3]
        
        true_mask = np.squeeze(img_to_array(imagepair2mask(image_array/255.0, mask_array)).astype(np.uint8))
        
        image_array = preprocess_image(image_array, get_model_type(model))

        if args.tta:
            output_mask = tta(image_array, model)
        elif args.ftta:
            output_mask = flip_tta(image_array, model)
        else:
            output_mask = mask_image(image_array, model)

        eval_mask = np.squeeze(output_mask.astype(np.uint8))
        
        true_labels = label(true_mask)
        eval_labels = label(eval_mask)
        
        print("Number of labels in image %s = %d" % (mask_path, len(np.unique(true_labels)) - 1))
        
        tp, fp, fn = confusion_terms(true_labels, eval_labels)
        ttp += tp
        tfp += fp
        tfn += fn

    print("\nTP\tFP\tFN\tPrec")
    print("{}\t{}\t{}\t{:1.3f}".format(ttp, tfp, tfn,  ttp / (ttp + tfp + tfn)))
