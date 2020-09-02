import os

import numpy as np

from keras.preprocessing.image import array_to_img, load_img, img_to_array

import glob
import geohash2
import random
import scipy.ndimage.morphology as morph

from mask_image import getStaticMap

from extract_image_masks import imagepair2mask

def bbox1(mask):
    
    a = np.where(np.squeeze(mask) != 0)
    if len(a) == 4:
        bbox = np.min(a[0]), np.max(a[0]), np.min(a[1]), np.max(a[1])
    else:
        bbox = [0,0,0,0]
    return bbox

def ncc(image1, image2, mask = None, dilations=2):
    if image1.shape != image2.shape:
        raise ValueError("Input images are different shapes")
        
    if len(image1.shape) != 3:
        raise ValueError("Inputs need to be multichanneled images")
    a = np.mean(image1, axis=2)
    v = np.mean(image2, axis=2)        

    a = np.ndarray.flatten(a)
    v = np.ndarray.flatten(v)
    
    if mask is not None:
        dmask = morph.binary_dilation(mask, iterations=dilations)
        fm = np.ndarray.flatten(dmask)
        maskidx = np.where(fm > 0)
        if len(maskidx) > 10:
            a = a[np.where(fm > 0)]
            v = v[np.where(fm > 0)]
    
    a = (a - np.mean(a)) / (np.std(a) * len(a))
    v = (v - np.mean(v)) /  np.std(v)
    
    x = np.correlate(a, v)
    
    return x[0]    

def montecarlo(image, mask, latlng, llrange, nlevel = 5, nrand = 50, zoom = 19, threshold = 0.75):
    
    best_ncc = -1000.0
    use_mask_level = 2
    best_geohash = ""
    best_image = image.copy()
    for level in range(0, nlevel):
        shrinker = 2**(-level)
        latlng1 = (latlng[0] - llrange*shrinker, latlng[1] - llrange*shrinker)
        latlng2 = (latlng[0] + llrange*shrinker, latlng[1] + llrange*shrinker)
        if level < use_mask_level:
            best_ncc, best_geohash, best_image = find_optimum(image, None, latlng1, latlng2, nrand, zoom, best_ncc, best_geohash, best_image)
            continue
        elif level == use_mask_level:
            best_ncc = -1000.0
        best_ncc, best_geohash, best_image = find_optimum(image, mask, latlng1, latlng2, nrand, zoom, best_ncc, best_geohash, best_image)
            
        print("Best NCC at level %d = %f, geohash = %s" % (level + 1, best_ncc, best_geohash))
        if best_ncc > threshold:
            break
        latlng = geohash2.decode_exactly(best_geohash)
        latlng = latlng[0], latlng[1]
    return best_ncc, best_geohash, best_image

def find_optimum(image, mask, latlng1, latlng2, nrand, zoom, best_ncc, best_geohash, best_image):
    
    latlngs = [(random.uniform(latlng1[0], latlng2[0]), random.uniform(latlng1[1], latlng2[1])) for i in range(nrand)]
    
    for latlng in latlngs:
        dimage = getStaticMap(latlng, zoom, image.shape[0:2], image_format='jpg', use_proxy=args.proxy)
        nxc = ncc(image, dimage, mask)
        if nxc > best_ncc:
            best_ncc = nxc
            best_geohash = geohash2.encode(latlng[0], latlng[1], precision=32)
            best_image = dimage
    
    return best_ncc, best_geohash, best_image

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="Test deeplab array panel area", 
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("-i", "--input", help="Use images in folder, geoencoded file names", required = True)
    parser.add_argument("-m", "--masks", help="Use masks in folder", required = True)
    
    parser.add_argument("-o", "--output", help="Output images in folder, same file names", required = True)
    parser.add_argument("-n", "--omasks", help="Output masks to folder, same file names", required = True)
    parser.add_argument("--proxy", help="use the proxy to download", required = False,  action = 'store_true', default=False)

    parser.add_argument("-r", "--range", help="lat/long range to search", required = False, type=float, default=0.00008)
        
    parser.add_argument("-z", "--zoom", help="zoom of image downloaded from maps", required = False, type=int, default=19)
    parser.add_argument("-t", "--threshold", help="NCC threshold, if downloaded image greater than this it is a duplicate", required = False, type=float, default=0.85)


    args = parser.parse_args()
    
    image_paths = []
    mask_paths = []
    geohashes = []
    for imgpath, maskpath in zip(glob.glob(os.path.join(args.input, "*.*")) , glob.glob(os.path.join(args.masks, "*.*"))):
        fname = os.path.basename(imgpath)
        mname = os.path.basename(maskpath)
        
        
        image = img_to_array(load_img(imgpath))
        overlay = img_to_array(load_img(maskpath))
        
        if image.shape != overlay.shape:
            print("Warning: image %s and mask %s are different shapes, skipping")
            continue
        
        igeohash, _ = os.path.splitext(fname)
        mgeohash, _ = os.path.splitext(mname)
        
        if igeohash != mgeohash:
            raise ValueError("Mismatched image and mask paths")
            
        try:
            latlng = geohash2.decode_exactly(igeohash)
            image_paths.append(imgpath)
            mask_paths.append(maskpath)
            geohashes.append(igeohash)
        except KeyError as ex:
            print("Key error in filename %s, skipping" % (fname))
            continue            
            
    for imgpath, maskpath, geohash in zip(image_paths, mask_paths, geohashes):
        fname = os.path.basename(imgpath)
        geohash, _ = os.path.splitext(fname)
        
        latlng = geohash2.decode_exactly(geohash)
        latlng = (latlng[0], latlng[1])

        image = img_to_array(load_img(imgpath))
        overlay = img_to_array(load_img(maskpath))/255.0

        mask = imagepair2mask(image/255.0, overlay)
        
        timage = getStaticMap(latlng, args.zoom, image.shape[0:2], image_format='jpg', use_proxy=args.proxy)
        if ncc(image, timage) > args.threshold:
            print("Skipping image %s" % (fname))
            continue            
        
        mask_array = img_to_array(mask)
        if len(np.unique(mask_array)) == 1:
            nlevel, nrand = 2, 30
        else:
            nlevel, nrand = 5, 40
            
        best_ncc, best_geohash, best_image = montecarlo(image, mask_array, latlng, args.range, zoom = args.zoom, threshold=args.threshold, nlevel=nlevel, nrand=nrand)
        if best_ncc > args.threshold:
            print("Skipping image %s" % (fname))
            continue
        
        array_to_img(best_image).save(os.path.join(args.output, best_geohash + ".jpg"))
        mask.save(os.path.join(args.omasks, best_geohash + ".png"))
                
