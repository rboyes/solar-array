import numpy as np
import os
import itertools

from keras.preprocessing.image import array_to_img
import geohash2

import pandas as pd


from mask_image import getStaticMap, convToLatLong, latLongDistance

if __name__ == '__main__':
    
    area_choices = {"radyr" : [(51.515038, -3.264662), (51.521038, -3.253662)],
                    "cardiff-nw" : [(51.513032 -3.278080), (51.529047 -3.220842)],
                    "truro" : [(50.255276, -5.126830),(50.280242, -5.030567)],
                    "cardiff" : [(51.453775, -3.285230),(51.549318, -3.081602)],
					 "exeter" : [(50.682542, -3.566623),(50.746967, -3.458233)],
                    "staines" : [(51.419570, -0.584109),(51.439256, -0.491873)],
                    "reading" : [(51.413069, -1.062613),(51.492414, -0.884075)],
                    "bakersfield" : [(35.270983, -119.176717), (35.428574, -118.930458)]}
    
    
    import argparse
    parser = argparse.ArgumentParser(description="Download a map area, geoencoded file names", 
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("-f", "--llatlong", help="From latitutde/longitude for radyr", nargs='+', required = False, type=float, default=[51.515038, -3.264662])
    parser.add_argument("-t", "--ulatlong", help="To latitutde/longitude for radyr", nargs='+', required = False, type=float, default=[51.521038, -3.253662])
    parser.add_argument("-c", "--choice", help="Choose an area to download", required = False, choices = set(area_choices.keys()))

    parser.add_argument("-s", "--size", help="size of image downloaded from maps", nargs = '+', required = False, type=int, default=[640,640])
    parser.add_argument("-z", "--zoom", help="zoom of image downloaded from maps", required = False, type=int, default=19)
    parser.add_argument("-d", "--folder", help="Save downloaded images to this folder, geoencoded file names", required = True)
    
    parser.add_argument("--csv", help = "Use a set of lat/long coordinates from a csv file", required = False)
    parser.add_argument("--proxy", help="use the proxy to download", required = False,  action = 'store_true', default=False)
    parser.add_argument("--apikey", help="Specify the google static maps api key", required = False)

    args = parser.parse_args()
    
    if args.csv is not None:
        latlong_df = pd.read_csv(args.csv)
        subset = latlong_df[['latitude', 'longitude']]
        latlongs = [tuple(x) for x in subset.values]      
    else:
        if args.choice is not None:
            llatlong = area_choices[args.choice][0]
            ulatlong = area_choices[args.choice][1]
        else:
            llatlong = args.llatlong
            ulatlong = args.ulatlong
            
        latlong1 = convToLatLong(0, 0, llatlong, args.zoom, tuple(args.size))
        latlong2 = convToLatLong(args.size[1] - 1, args.size[0] - 1, llatlong, args.zoom, tuple(args.size))
        
        print("Single map distance on diagonal = %3.2f metres" % (latLongDistance(latlong1, latlong2)))
        print("Range of distance on diagonal = %3.2f metres" % (latLongDistance(llatlong, ulatlong)))
        
        latStep = abs(latlong1[0] - latlong2[0])*0.85
        longStep = abs(latlong1[1] - latlong2[1])*0.85
    
        latRange = np.arange(min(llatlong[0], ulatlong[0]), max(llatlong[0], ulatlong[0]), latStep)
        longRange = np.arange(min(llatlong[1], ulatlong[1]), max(llatlong[1], ulatlong[1]), longStep)
        
        
        latlongs = [latlong for latlong in itertools.product(latRange, longRange)]
        
    ndownloads = len(latlongs)
        
    for latlong in latlongs:
        geohash = geohash2.encode(latlong[0], latlong[1], precision=32)
        decoded_hash = geohash2.decode_exactly(geohash)

        ndownloads = ndownloads - 1
    
        output_path = os.path.join(args.folder, geohash + ".jpg")
        if os.path.exists(output_path):
            print("Skipping image located at (%f,%f) as filename %s as it already exists" % (latlong[0], latlong[1], geohash))            
            continue
        else:
            print("Saving image located at (%f,%f) as filename %s, decoded as (%f,%f), number to go = %d" % (latlong[0], latlong[1], geohash, decoded_hash[0], decoded_hash[1], ndownloads))
                
        image = getStaticMap(latlong, args.zoom, tuple(args.size), use_proxy=args.proxy, api_key=args.apikey)/255.0
        
        if args.folder and os.path.isdir(args.folder):
                array_to_img(image).save(output_path)        