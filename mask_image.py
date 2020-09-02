import io
import math
import numpy as np
import os
import pandas as pd
import geohash2
import itertools

from train import overlap, overlap_loss, bce_overlap_loss, preprocess_image, get_model_type

from keras.preprocessing.image import load_img, img_to_array, array_to_img
from keras.models import load_model

from scipy.ndimage import label
from scipy.ndimage.measurements import center_of_mass

from PIL import Image

import random

import time

def getStaticMap(latlong, zoom, size, image_format='png32', use_proxy = False, api_key='AIzaSyAb8ehOA6s65DORxE6JqQ2ZMU5jMSxXOyE'):
    import requests
    import urllib.parse
    api_keys = [api_key]
    
    urlVars = {
            'center': str(latlong[0]) + ',' + str(latlong[1]),
            'zoom' : str(zoom),
            'size' : str(size[0]) + 'x' + str(size[1]),
            'key' : random.choice(api_keys),
            'maptype' : 'satellite',
            'format' : image_format
            }
    url = 'https://maps.googleapis.com/maps/api/staticmap?'
    url += urllib.parse.urlencode(urlVars)
    
    if use_proxy:
        proxy_server = "http://proxy.fjgslbdmz.uk.centricaplc.com:80"
        proxy = {"http": proxy_server, "https":proxy_server}
        delay = 0.2
        while True:
            try:
                time.sleep(delay)                            
                map_data = requests.get(url, verify=False, timeout = 12, proxies = proxy).content
                break
            except requests.exceptions.ProxyError as ex:
                print("Warning - proxy error " + str(ex))
                delay += 60.0
                continue
    else:
        map_data = requests.get(url).content

    map_image = Image.open(io.BytesIO(map_data))

    map_array = img_to_array(map_image)
    map_array = map_array[:,:,:3]
        
    return map_array

def calcDegreesPerPixel(latlong, zoom, size):
    parallelMultiplier = math.cos(latlong[0] * math.pi / 180)
    degreesPerPixelX = 360 / math.pow(2, zoom + 8)
    degreesPerPixelY = 360 / math.pow(2, zoom + 8) * parallelMultiplier  
    
    return (degreesPerPixelX, degreesPerPixelY)
    
def convToLatLong(x, y, latlong, zoom, size):

    (degreesPerPixelX, degreesPerPixelY) = calcDegreesPerPixel(latlong, zoom, size)
    pointLat = latlong[0] - degreesPerPixelY * ( y - size[0] / 2)
    pointLng = latlong[1] + degreesPerPixelX * ( x  - size[1] / 2)

    return (pointLat, pointLng)

def convToPixel(pointLat, pointLng, latlong, zoom, size):
    
    (degreesPerPixelX, degreesPerPixelY) = calcDegreesPerPixel(latlong, zoom, size)
    y = round((latlong[0] - pointLat)/degreesPerPixelY + size[0]/2)
    x = round((pointLng - latlong[1])/degreesPerPixelX + size[1]/2)
    
    return (x, y)

def convToArea(latlong, zoom, size):
    latlong1 = convToLatLong(0, 0, latlong, zoom, size)
    latlong2 = convToLatLong(size[1] - 1, size[0] - 1, latlong, zoom, size)

    r2 = 6373000.0**2
    lat2 = math.radians(latlong2[0])
    lat1 = math.radians(latlong1[0])
     
    area = r2*math.pi*(math.cos(lat2) - math.cos(lat1))*(latlong2[1] - latlong1[1])/180.0
        
    return area

def pixelArea(latlong, zoom, size):
    return convToArea(latlong, zoom, size)/(size[0]*size[1])

def latLongDistance(latlong1, latlong2):
    lat1 = math.radians(latlong1[0])
    lat2 = math.radians(latlong2[0])
    lon1 = math.radians(latlong1[1])
    lon2 = math.radians(latlong2[1])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    
#    a = (math.sin(dlat/2))**2 + math.cos(lat1) * math.cos(lat2) * (math.sin(dlon/2))**2 
#    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1.0 - a))
    a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
    c = 2 * math.asin(math.sqrt(a))     
    r = 6373000.0
    return r*c

def precisionArea(precision):
    geohash = geohash2.encode(0, 0, precision=15)
    lat, long, latErr, longErr = geohash2.decode_exactly(geohash[:precision])
    
    latDist = latLongDistance((0, 0), (lat, 0))
    longDist = latLongDistance((0, 0), (0, long))
    return latDist*longDist    

def maskToLatLong(mask, latlong, zoom, size, area_threshold = 1.5):
    
    labelled_image, num_labels = label(mask)
    unique, counts = np.unique(labelled_image, return_counts = True)
    label_counts = dict(zip(unique, counts))
    
    pixel_centres = center_of_mass(mask, labelled_image,range(1,num_labels+1))
    image_area = convToArea(latlong, zoom, size)    
    latlong_centres = []
    latlong_areas = []
    
    for lidx, pc in enumerate(pixel_centres, start = 1):
        pixel_area = image_area*label_counts[lidx]/(size[0]*size[1])
        
        if pixel_area > area_threshold:        
            latlong_areas.append(pixel_area)
            lc = convToLatLong(pc[1], pc[0], latlong, zoom, (size[1], size[0]))
            latlong_centres.append(lc)
    
    return latlong_centres, latlong_areas

def maskToLatLongPoints(mask, latlong, zoom):

    mask = np.squeeze(mask)
    indexes = np.argwhere(mask > 0)
    size = (mask.shape[0], mask.shape[1])
    return [convToLatLong(index[1], index[0], latlong, zoom, (size[0], size[1])) for index in indexes]


def maskToClusters(mask, latlong, zoom, size, area_threshold = 1.5):
    labelled_image, num_labels = label(mask)
    
    unique, counts = np.unique(labelled_image, return_counts = True)
    
    label_counts = dict(zip(unique, counts))
    label_areas = dict()
    image_area = convToArea(latlong, zoom, size)    
    
    for l, count in label_counts.items():
        label_areas[l] = image_area * label_counts[l]/(size[0]*size[1])

    label_points = dict()
    for l, count in label_counts.items():
        
        if l == 0:
            continue
        if label_areas[l] < area_threshold:
            continue
        
        lp = []
        for r, c in zip(*np.where(np.squeeze(labelled_image == l))):
            lc = convToLatLong(c, r, latlong, zoom, (size[1], size[0]))
            lp.append(lc)
        label_points[l] = lp
        
    return label_points, label_areas
        
def flip_tta(image, model, binarize = True):
    mask = mask_image(image, model, binarize = False)
    mask += np.fliplr(mask_image(np.fliplr(image), model, binarize = False))    
    mask = mask/2.0
    if binarize:
        mask = np.where(mask > 0.5, 1.0, 0.0)    
    return mask
    
def tta(image, model, binarize = True):
    
    mask = mask_image(image, model, binarize = False)
    mask += np.rot90(mask_image(np.rot90(image, 1), model, binarize = False),3)
    mask += np.rot90(mask_image(np.rot90(image, 2), model, binarize = False),2)
    mask += np.rot90(mask_image(np.rot90(image, 3), model, binarize = False),1)    
    
    imagelr = np.fliplr(image)
    masklr = mask_image(imagelr, model, binarize = False)
    masklr += np.rot90(mask_image(np.rot90(imagelr, 1), model, binarize = False),3)
    masklr += np.rot90(mask_image(np.rot90(imagelr, 2), model, binarize = False),2)
    masklr += np.rot90(mask_image(np.rot90(imagelr, 3), model, binarize = False),1)
    
    masklr = np.fliplr(masklr)
    
    mask += masklr
    
    mask = mask / 8.0
        
    if binarize:
        mask = np.where(mask > 0.5, 1.0, 0.0)
    
    return mask        

def mask_image(image, model, binarize = True):

    model_size = model.layers[0].input_shape[1:4]
    image_size = image.shape[0:2]

    step = min(model_size[0:2]) // 4

    starts0 = np.arange(0, image_size[0] - model_size[0], step)        
    if starts0[-1] < (image_size[0] - model_size[0]):
        starts0 = np.append(starts0, image_size[0] - model_size[0])
            
    starts1 = np.arange(0, image_size[1] - model_size[1], step)        
    if starts1[-1] < (image_size[1] - model_size[1]):        
        starts1 = np.append(starts1, image_size[1] - model_size[1])

    starts = [start for start in itertools.product(starts0, starts1)]

    num_patches = len(starts)
    patches = np.zeros((num_patches, model_size[0], model_size[1], model_size[2]))    
    count_image = np.zeros((image_size[0], image_size[1], 1))
    for patch_index, start in enumerate(starts):
        slice0  = slice(start[0],start[0]+model_size[0])
        slice1 = slice(start[1],start[1]+model_size[1])
        patches[patch_index] = image[slice0, slice1, :]
        count_image[slice0, slice1, 0] += 1.0

    patch_masks = model.predict(patches, batch_size = min(num_patches, 4))
    mask = np.zeros((image_size[0], image_size[1], 1))
    for patch_index, start in enumerate(starts):
        patch_mask = patch_masks[patch_index]
        slice0  = slice(start[0],start[0]+model_size[0])
        slice1 = slice(start[1],start[1]+model_size[1])
        mask[slice0, slice1, 0] += patch_mask[:,:,0]
    mask = mask / count_image
    
    if binarize:
        mask = np.where(mask > 0.5, 1.0, 0.0)
    
    return mask
            
def compute_mask(input_image, tta, ftta, binarize = False):
    if tta:
        return tta(input_image, model, binarize)
    elif ftta:
        return flip_tta(input_image, model, binarize)
    else:
        return mask_image(input_image, model, binarize)

if __name__ == '__main__':
    import argparse
    import sys
    from tqdm import tqdm
    
    parser = argparse.ArgumentParser(description="Test deeplab segmenter", 
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("-i", "--image", help="input image ", required = False)
    parser.add_argument("-d", "--directory", help="input folder contaoning images", required=False)
    
    parser.add_argument("-l", "--latlong", help="input latitude/longitude", nargs='+', required = False, type=float, default=[51.521038, -3.264662])
    parser.add_argument("-m", "--model", help="input model", required = True)
    parser.add_argument("-o", "--output", help="output image or latlong coordinates or folder", required = True)
    parser.add_argument("-s", "--size", help="size of image, only required for static map download", nargs = '+', required = False, type=int, default=[640,640])
    parser.add_argument("-z", "--zoom", help="zoom of image, only required for static map download", required = False, type=int, default=19)
    parser.add_argument("-t", "--tta", help="test time augmentation", required = False,  action = 'store_true', default=False)
    parser.add_argument("-f", "--ftta", help="flip test time augmentation", required = False,  action = 'store_true', default=False)
    
    
    args = parser.parse_args()
    
    model = load_model(args.model, custom_objects={'overlap_loss' : overlap_loss,
                                                   'overlap' : overlap,
                                                   'bce_overlap_loss' : bce_overlap_loss}) 
        
    if args.directory is not None and os.path.isdir(args.directory):
        
        if not os.path.isdir(args.output):
            raise ValueError("Error: output should be a directory")
            
        paths = [os.path.join(args.directory, f) for f in os.listdir(args.directory) if os.path.isfile(os.path.join(args.directory, f))]

        for path in tqdm(paths):
            filename, extension = os.path.splitext(path)
            if extension != '.jpg':
                continue
            
            input_image = img_to_array(load_img(path))
            output_mask = compute_mask(input_image, args.tta, args.ftta, binarize=True)
            
            if np.sum(output_mask) == 0:
                continue
            
            output_path = filename + '.png'
            output = array_to_img(output_mask)
            output.save(output_path)                
    
        sys.exit(0)

    if args.image is not None:
        input_image = img_to_array(load_img(args.image))      
    else:
        input_image = getStaticMap(tuple(args.latlong), args.zoom, tuple(args.size))
       
    input_image = preprocess_image(input_image, get_model_type(model))
    output_mask = compute_mask(input_image, args.tta, args.ftta)
    
    if os.path.splitext(args.output)[1] == '.csv':
        latlong_centres, areas = maskToLatLong(output_mask, tuple(args.latlong), args.zoom, tuple(args.size))
        print("Area = %f" % (areas[0]))
        df = pd.DataFrame(latlong_centres)
        df.to_csv(args.output,header=False, index=False)        
    else:
        output = array_to_img(output_mask)
        output.save(args.output)
