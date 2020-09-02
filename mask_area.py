import os
import time

import numpy as np
import pandas as pd
from collections import defaultdict
from train import overlap, overlap_loss, bce_overlap_loss

from sklearn.cluster import DBSCAN

from keras.preprocessing.image import array_to_img, load_img, img_to_array
from keras.models import load_model
from keras.applications import imagenet_utils

import glob
import geohash2
from train import preprocess_image, get_model_type


from mask_image import mask_image, tta, flip_tta, maskToLatLong, pixelArea, precisionArea, maskToLatLongPoints
from mask_image import getStaticMap, maskToClusters, convToLatLong, latLongDistance
       
def mergeClusters(clusters, precision=11):
    
    hashedClusters = []
    for i, cluster in enumerate(clusters):
        hashes = set()
        for point in cluster:
            hashes.add(geohash2.encode(point[0], point[1], precision=precision))         
        hashedClusters.append(hashes)
        
    newClusters1 = mergeClusters3(hashedClusters)
    newClusters2 = mergeClusters3(newClusters1)
    while (len(newClusters2) - len(newClusters1)) > 0:
        newClusters1 = newClusters2.copy()
        newClusters2 = mergeClusters3(newClusters1)        

    pointClusters = []
    for i, hashedCluster in enumerate(newClusters2):
        pointCluster = []
        for geohash in hashedCluster:
            point = geohash2.decode_exactly(geohash)
            pointCluster.append((point[0], point[1]))
        pointClusters.append(pointCluster)
        
    return pointClusters

def findjoin(cluster, clusters):
    
    for label1, cluster1 in clusters.items():
        if not cluster.isdisjoint(cluster1):
            return True, label1
    return False, -1

def mergeClusters3(hashedClusters):
    assignedClusters = defaultdict(set)
    unassignedClusters = defaultdict(set)
    
    for i, cluster in enumerate(hashedClusters):
        unassignedClusters[i] = hashedClusters[i].copy()
    
    unassignedMergeLabel, unassignedCluster = unassignedClusters.popitem()
    assignedClusters[unassignedMergeLabel] = unassignedCluster
        
    while unassignedClusters:
                
        unassignedMergeLabel, unassignedCluster = unassignedClusters.popitem()
        isjoined, assignedMergeLabel = findjoin(unassignedCluster, assignedClusters)
        
        if isjoined:
            assignedClusters[assignedMergeLabel].update(unassignedCluster)         
        else:
            assignedClusters[unassignedMergeLabel] = unassignedCluster
            
        print("Number of unassigned clusters = %d" % (len(unassignedClusters)))
            
    mergedClusters = []
    for ckey, clist in assignedClusters.items():
        mergedClusters.append(clist)
    return mergedClusters
    
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="Test deeplab array panel area", 
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("-m", "--models", help="input model", required = True, nargs='+')
    
    parser.add_argument("-o", "--output", help="latlong coordinates", required = True)
    parser.add_argument("-d", "--folder", help="Use images in downloads folder, geoencoded file names", required = False)
    parser.add_argument("-a", "--tta", help="test time augmentation", required = False,  action = 'store_true', default=False)
    parser.add_argument("-z", "--zoom", help="zoom of image downloaded from maps", required = False, type=int, default=19)

    args = parser.parse_args()

    filepaths = glob.glob(os.path.join(args.folder, "*.jpg"))


    if len(args.models) > 1:
        masks = []    
        for modelPath in args.models:
            model = load_model(modelPath, custom_objects={'overlap_loss' : overlap_loss,
                                                          'overlap' : overlap,
                                                          'bce_overlap_loss' : bce_overlap_loss})
        
            model_type = get_model_type(model)
            for findex, fpath in enumerate(filepaths):
                
                print("Processing mask %d of %d: %s, model %s" % (findex + 1, len(filepaths), fpath, modelPath))
                
                fname = os.path.basename(fpath)
                geohash, _ = os.path.splitext(fname)
                exact_latlong = geohash2.decode_exactly(geohash)
                latlong = (exact_latlong[0], exact_latlong[1])
                image = img_to_array(load_img(fpath))
                
                image = preprocess_image(image, model_type)
                    
                if args.tta:
                    mask = tta(image, model, binarize=False)
                else:
                    mask = mask_image(image, model, binarize=False)
                    
                if len(masks) == len(filepaths):
                    masks[findex] += mask
                else:
                    masks.append(mask)
                    
        for mindex, mask in enumerate(masks):
            mask /= float(len(args.models))
            mask = np.where(mask > 0.5, 1.0, 0.0)
            masks[mindex] = mask
    else:
        model = load_model(args.models[0], custom_objects={'overlap_loss' : overlap_loss,
                           'overlap' : overlap,   
                           'bce_overlap_loss' : bce_overlap_loss})
    
        model_type = get_model_type(model)
        
    clusters = []
    areas = []
    mask_points = []

    for findex, fpath in enumerate(filepaths):
        fname = os.path.basename(fpath)
        geohash, _ = os.path.splitext(fname)
        exact_latlong = geohash2.decode_exactly(geohash)
        latlong = (exact_latlong[0], exact_latlong[1])        
        image = img_to_array(load_img(fpath))
        image = preprocess_image(image, model_type)
            
        if(len(args.models) > 1):
            mask = masks[findex]
        else:
            if args.tta:
                mask = tta(image, model, binarize=True)
            else:
                mask = mask_image(image, model, binarize=True)

        mask = np.squeeze(mask)
        mask_latlongs = maskToLatLongPoints(mask, latlong, args.zoom)
        if len(mask_latlongs) > 0:
            mask_points.append(mask_latlongs)

    mask_points = np.array(mask_points)

    clusterer = DBSCAN(eps=1.0E-6, min_samples=25)
    clusterer.fit(mask_points)

    cluster_centres = []
    cluster_sizes = []
    for label in np.unique(clusterer.labels_):
        label_indexes = np.argwhere(clusterer._labels == label)
        label_points = mask_points[label_indexes, :]
        cluster_centres.append(np.mean(label_points, axis=0))
        cluster_sizes.append(len(label_points))


    print("Number of clusters = %d" % (len(clusters)))
    startTime = time.time()
    mergePrecision = 10
    mergedClusters = mergeClusters(clusters, precision=mergePrecision)
    precArea = precisionArea(mergePrecision)
    print("Number of merged clusters = %d, time taken = %f" % (len(mergedClusters), time.time() - startTime))
    
    clusterData = []
    for cluster in mergedClusters:
        data = list(np.average(np.array(cluster), axis = 0))        
        data.append(len(cluster)*precArea)
        clusterData.append(data)
    
    centres_df = pd.DataFrame(clusterData, columns=['latitude', 'longitude', 'area m2'])
    centres_df.to_excel(args.output, index = True)
