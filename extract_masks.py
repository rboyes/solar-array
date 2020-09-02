import glob
import json
import numpy as np
import pandas as pd
import os
from PIL import Image
from PIL import ImageDraw

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="Batch process tif images with polygons to create numpy images and masks", 
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("-g", "--geojson", help="input geojson that contains image names and polygons", required = True)
    parser.add_argument("-i", "--input", help="input folder of tif images", required = True)
    parser.add_argument("-o", "--images", help="output folder for converted image files", required = True)
    parser.add_argument("-m", "--masks", help="output folder for converted masks", required = True)
    parser.add_argument("-c", "--csv", help = "output listing of all files", required=True)
    parser.add_argument("-s", "--size", help = "output size of files", required=False, type=int, nargs='+', default=[256,256])
    parser.add_argument("-t", "--step", help="increment between output images", required = False, type=int, default=128)
    parser.add_argument("-z", "--zoom", help="resample the images to zoom in", required = False, type=float, default=1.0)
    
    
    args = parser.parse_args()
    
    # Load the geojson file
    with open(args.geojson) as gjf:
        geometry = json.load(gjf)
    
    # Extract the geojson polygons with their associated image ids
    # Create a dictionary of {image_id, [polygons, [(x1,y1),(x2,y2)]]}
    polygon_lookup = {}
    for polygon in geometry['features']:
        properties = polygon['properties']
        image_name = properties['image_name']
        vertices = properties['polygon_vertices_pixels']
        
        if image_name not in polygon_lookup:
            polygon_lookup[image_name] = []
        polygon_lookup[image_name].append(vertices)
    
    tiff_paths = glob.glob(os.path.join(args.input, "*.tif"))
    
    nrow, ncol = args.size[0], args.size[1]
    step = args.step
    image_data = []
    
    for tindex, tiff_path in enumerate(tiff_paths):
        print("Reading tiff %d of %d, %s" % (tindex + 1, len(tiff_paths), tiff_path))        
        image = Image.open(tiff_path).convert('RGB')
        image_name = os.path.basename(os.path.splitext(tiff_path)[0])
        
        mask = Image.new('1', image.size, color = 0)
        
        if image_name in polygon_lookup:
            for polygon in polygon_lookup[image_name]:
                
                np_polygon = np.array(polygon)
                flat_polygon = np_polygon.flatten()
                if len(flat_polygon) < 6:
                    print("Warning: non-polygon shape found (< 3 vertices), skipping")
                    continue
                ImageDraw.Draw(mask).polygon(flat_polygon.tolist(), outline = 1, fill = 1)
        
        if abs(1.0 - args.zoom) > 0.0:
            newsize = round(image.size[0]*args.zoom), round(image.size[1]*args.zoom)
            image = image.resize(newsize, Image.LANCZOS)
            mask = mask.resize(newsize, Image.NEAREST)
        
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
    df.to_csv(args.csv, index=False, header=False, mode='a')