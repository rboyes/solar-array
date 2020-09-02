import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

import random

import tensorflow as tf

from keras import backend as K

from keras.preprocessing.image import load_img, img_to_array, array_to_img
from keras.preprocessing.image import random_channel_shift, random_brightness
from keras.applications import imagenet_utils
from keras.layers import Input, merge, Conv2D, MaxPooling2D, Conv2DTranspose
from keras.layers import Dropout, BatchNormalization, Activation, SpatialDropout2D
from keras.layers import add, UpSampling2D
from keras.layers.merge import concatenate
from keras.callbacks import ModelCheckpoint, CSVLogger
from keras.models import Model, load_model
from keras.optimizers import Adam
from keras.losses import binary_crossentropy
from keras.applications import VGG16

from PIL import ImageOps

from resnet50 import ResNet50
import inception_resnet_v2
from inception_resnet_v2 import InceptionResNetV2

def get_model_type(model):
    if model.layers[5].name == 'res2a_branch2a':
        return "resnet"    
    elif len(model.layers) > 200 and model.layers[260] == 'block35_10_ac':
        return "inception_resnet"
    else:
        return "other"

def preprocess_image(image, model_type):
    if model_type == "resnet":
        return imagenet_utils.preprocess_input(image)
    elif model_type == "inception_resnet":
        return inception_resnet_v2.preprocess_input(image)
    else:
        return image/255.0
    
def conv3Block(inputs, size):

    convolved = Conv2D(size, (3, 3), padding='same')(inputs)
    convolved = BatchNormalization(axis = -1)(convolved)
    convolved = Activation('relu')(convolved)

    convolved = Conv2D(size, (3, 3), padding='same')(convolved)
    convolved = BatchNormalization(axis = -1)(convolved)
    convolved = Activation('relu')(convolved)
    
    convolved = Conv2D(size, (3, 3), padding='same')(convolved)
    convolved = BatchNormalization(axis = -1)(convolved)
    convolved = Activation('relu')(convolved)
    return convolved

def convBlock(inputs, size):

    convolved = Conv2D(size, (3, 3), padding='same')(inputs)
    convolved = BatchNormalization(axis = -1)(convolved)
    convolved = Activation('relu')(convolved)

    convolved = Conv2D(size, (3, 3), padding='same')(convolved)
    convolved = BatchNormalization(axis = -1)(convolved)
    convolved = Activation('relu')(convolved)

    return convolved

def mergeBlock(lores, hires, size):
    up = Conv2DTranspose(size,(2,2), strides=(2,2), padding='same')(lores)
    merged = concatenate([up, hires]) 
    conv = convBlock(merged, size)
    return conv

def merge3Block(lores, hires, size):
    up = Conv2DTranspose(size,(2,2), strides=(2,2), padding='same')(lores)
    merged = concatenate([up, hires]) 
    conv = conv3Block(merged, size)
    return conv

def poolBlock(inputs, size):
    conv = convBlock(inputs, size)
    pool = MaxPooling2D(pool_size=(2, 2))(conv)
    return conv, pool 

def endBlock(x):
    #x = SpatialDropout2D(0.1)(x)
    x = Conv2D(1, (1, 1))(x)
    x = BatchNormalization(axis = -1)(x)
    x = Activation('sigmoid')(x)
    
    return x

def filterSize(size0, layer):
    return size0*(2**layer)

def createUnet4(input_shape, size = 32):

    inputs = Input((input_shape[0], input_shape[1], input_shape[2])) 
    (conv0down, pool0down) = poolBlock(inputs,    filterSize(size, 0))
    (conv1down, pool1down) = poolBlock(pool0down, filterSize(size, 1))
    (conv2down, pool2down) = poolBlock(pool1down, filterSize(size, 2))
    (conv3down, pool3down) = poolBlock(pool2down, filterSize(size, 3))
    (conv4down, pool4down) = poolBlock(pool3down, filterSize(size, 4))

    centre = conv3Block(pool4down, filterSize(size, 4))

    conv4up = mergeBlock(centre,  conv4down, filterSize(size, 4))
    conv3up = mergeBlock(conv4up, conv3down, filterSize(size, 3))
    conv2up = mergeBlock(conv3up, conv2down, filterSize(size, 2))    
    conv1up = mergeBlock(conv2up, conv1down, filterSize(size, 1))
    conv0up = mergeBlock(conv1up, conv0down, filterSize(size, 0))
    
    end = endBlock(conv0up)
    
    model4u = Model(input=inputs, output=end)
        
    return model4u

def get_unet_resnet(input_shape):
    resnet_base = ResNet50(input_shape=input_shape, include_top=False)

    for l in resnet_base.layers:
        l.trainable = True
    conv1 = resnet_base.get_layer("activation_1").output
    conv2 = resnet_base.get_layer("activation_10").output
    conv3 = resnet_base.get_layer("activation_22").output
    conv4 = resnet_base.get_layer("activation_40").output
    conv5 = resnet_base.get_layer("activation_49").output

    up6 = concatenate([UpSampling2D()(conv5), conv4], axis=-1)
    conv6 = conv_block_simple(up6, 256, "conv6_1")
    conv6 = conv_block_simple(conv6, 256, "conv6_2")

    up7 = concatenate([UpSampling2D()(conv6), conv3], axis=-1)
    conv7 = conv_block_simple(up7, 192, "conv7_1")
    conv7 = conv_block_simple(conv7, 192, "conv7_2")

    up8 = concatenate([UpSampling2D()(conv7), conv2], axis=-1)
    conv8 = conv_block_simple(up8, 128, "conv8_1")
    conv8 = conv_block_simple(conv8, 128, "conv8_2")

    up9 = concatenate([UpSampling2D()(conv8), conv1], axis=-1)
    conv9 = conv_block_simple(up9, 64, "conv9_1")
    conv9 = conv_block_simple(conv9, 64, "conv9_2")

    vgg = VGG16(input_shape=input_shape, input_tensor=resnet_base.input, include_top=False)
    for l in vgg.layers:
        l.trainable = False
    vgg_first_conv = vgg.get_layer("block1_conv2").output
    up10 = concatenate([UpSampling2D()(conv9), resnet_base.input, vgg_first_conv], axis=-1)
    conv10 = conv_block_simple(up10, 32, "conv10_1")
    conv10 = conv_block_simple(conv10, 32, "conv10_2")
    conv10 = SpatialDropout2D(0.5)(conv10)
    x = Conv2D(1, (1, 1), activation="sigmoid", name="prediction")(conv10)
    model = Model(resnet_base.input, x)
    return model, "resnet"

def get_unet_inception_resnet_v2(input_shape):
    base_model = InceptionResNetV2(include_top=False, input_shape=input_shape)
    conv1 = base_model.get_layer('activation_3').output
    conv2 = base_model.get_layer('activation_5').output
    conv3 = base_model.get_layer('block35_10_ac').output
    conv4 = base_model.get_layer('block17_20_ac').output
    conv5 = base_model.get_layer('conv_7b_ac').output
    up6 = concatenate([UpSampling2D()(conv5), conv4], axis=-1)
    conv6 = conv_block_simple(up6, 256, "conv6_1")
    conv6 = conv_block_simple(conv6, 256, "conv6_2")

    up7 = concatenate([UpSampling2D()(conv6), conv3], axis=-1)
    conv7 = conv_block_simple(up7, 256, "conv7_1")
    conv7 = conv_block_simple(conv7, 256, "conv7_2")

    up8 = concatenate([UpSampling2D()(conv7), conv2], axis=-1)
    conv8 = conv_block_simple(up8, 128, "conv8_1")
    conv8 = conv_block_simple(conv8, 128, "conv8_2")

    up9 = concatenate([UpSampling2D()(conv8), conv1], axis=-1)
    conv9 = conv_block_simple(up9, 64, "conv9_1")
    conv9 = conv_block_simple(conv9, 64, "conv9_2")

    up10 = concatenate([UpSampling2D()(conv9), base_model.input], axis=-1)
    conv10 = conv_block_simple(up10, 48, "conv10_1")
    conv10 = conv_block_simple(conv10, 32, "conv10_2")
    conv10 = SpatialDropout2D(0.25)(conv10)
    x = Conv2D(1, (1, 1), activation="sigmoid", name="prediction")(conv10)
    model = Model(base_model.input, x)
    return model, "incres"

def conv_block_simple(prevlayer, filters, prefix, strides=(1, 1)):
    conv = Conv2D(filters, (3, 3), padding="same", kernel_initializer="he_normal", strides=strides, name=prefix + "_conv")(prevlayer)
    conv = BatchNormalization(name=prefix + "_bn")(conv)
    conv = Activation('relu', name=prefix + "_activation")(conv)
    return conv

def fuzzy_overlap(yTrue, yPredicted):
    smooth = 1.0E-4
    yTrueFlat = K.flatten(yTrue)
    yPredictedFlat = K.flatten(yPredicted)
    intersection = K.minimum(yTrueFlat, yPredictedFlat)
    union = K.maximum(yTrueFlat, yPredictedFlat)
    overlap = (K.sum(intersection)+smooth)/(K.sum(union) + smooth)
    return overlap

def overlap_loss(yTrue, yPredicted):
    return 1.0 - overlap(yTrue, yPredicted)

def bce_overlap_loss(yTrue, yPredicted):
    return overlap_loss(yTrue,yPredicted) + binary_crossentropy(yTrue, yPredicted)

def tversky_loss(y_true, y_predicted):
    actual = K.flatten(y_true)
    predicted = K.flatten(y_predicted)
    intersection = K.sum(K.minimum(actual,predicted))
    
    inverted_actual = 1.0 - actual
    inverted_predicted = 1.0 - predicted
    
    alpha_term = 0.7*K.sum(K.minimum(predicted,inverted_actual))
    beta_term = 0.3*K.sum(K.minimum(actual,inverted_predicted))
    
    return 1.0 - intersection/(intersection + alpha_term + beta_term)
        
def overlap(yTrue, yPredicted):
    smooth = 1.0E-4
    yTrueFlat = K.flatten(yTrue)
    yPredictedFlat = K.flatten(yPredicted)
    intersection = K.sum(yTrueFlat*yPredictedFlat)
    overlap = (2.0*intersection + smooth)/(K.sum(yTrueFlat) + K.sum(yPredictedFlat) + smooth)
    return overlap

def augment_brightness(image):
    return random_brightness(image, (0.8, 1.2))

def augment_channel(image):
    return random_channel_shift(image, 25.0, channel_axis=2)

def augment_autocontrast(image):
    x = array_to_img(image)
    x = ImageOps.autocontrast(x, cutoff = 0.001)
    return img_to_array(x)

def augment_adapthist(image):
    from skimage import exposure
    return 255.0*exposure.equalize_adapthist(image.astype(np.uint8), clip_limit=0.02)

def augment_image(image, mask):
    
    if random.choice([False, True]):
        image = np.fliplr(image)
        mask = np.fliplr(mask)
    
    rot = np.random.choice([0, 1, 2, 3])
    image = np.rot90(image, rot)
    mask = np.rot90(mask, rot)
                    
    augmentations = [augment_brightness, augment_channel, augment_autocontrast]
    random.shuffle(augmentations)
    
    for augmentation in augmentations:
        if random.random() < 0.25:
            image = augmentation(image)
    return image, mask

def weighted_generator(image_paths, mask_paths, weights, batch_size=16, augment=True, imagenet_preprocess="resnet"):
    
    if len(image_paths) != len(mask_paths):
        raise ValueError("Mismatched array lengths")
        
    if len(image_paths) != len(weights):
        raise ValueError("Mismatched array lengths")
    
    nrow, ncol, nchannel = get_shape(image_paths[0])
    
    while True:
        images = np.ndarray((batch_size, nrow, ncol, nchannel), dtype = np.float32)
        masks = np.ndarray((batch_size, nrow, ncol, 1), dtype = np.float32)
        batch_index = 0
        while batch_index < batch_size:
            image_index = random.randint(0, len(image_paths) - 1)
            if random.random() > weights[image_index]:
                continue
            image = img_to_array(load_img(image_paths[image_index]))
            mask = img_to_array(load_img(mask_paths[image_index], grayscale = True))
            if augment:
                image, mask = augment_image(image, mask)
            images[batch_index] = preprocess_image(image, imagenet_preprocess)
            masks[batch_index] = preprocess_image(mask, "other")
            batch_index += 1
        yield images, masks

def get_paths(filename):
    df = pd.read_csv(filename, header = None)
    return df[0].tolist(), df[1].tolist(), df[2].tolist()  

def get_shape(image_path):
    test_image = img_to_array(load_img(image_path))
        
    return test_image.shape[0], test_image.shape[1], test_image.shape[2]    

def shuffle_lists(a, b, c):
    d = list(zip(a, b, c))
    random.shuffle(d)
    a, b, c = zip(*d)
    return a, b, c

def basic_generator(image_paths, mask_paths, batch_size = 16, imagenet_preprocess = "resnet"):
    while True:
        np_image_paths = np.array(image_paths)
        np_mask_paths = np.array(mask_paths)
        
        nrow, ncol, nchannel = get_shape(image_paths[0])
        
        num_images = len(image_paths)        
        image_index = np.random.permutation(num_images)
        
        for index in range(0, num_images, batch_size):
            end_index = min(index + batch_size, num_images)
            
            index_bs = end_index - index
            batch_image_paths = np_image_paths[image_index[index:end_index]].tolist()
            batch_mask_paths = np_mask_paths[image_index[index:end_index]].tolist()
            images = np.ndarray((index_bs, nrow, ncol, nchannel), dtype = np.float32)
            masks = np.ndarray((index_bs, nrow, ncol, 1), dtype = np.float32)
               
            for batch_index in range(index_bs):
                image = img_to_array(load_img(batch_image_paths[batch_index]))
                mask = img_to_array(load_img(batch_mask_paths[batch_index], grayscale = True))
                                
                images[batch_index] = image
                masks[batch_index] = mask/255.0
                
            images = preprocess_image(images, imagenet_preprocess)

            yield images, masks
            
            
def generator(image_paths, mask_paths, contains_mask, batch_size = 16, augment = True, split=0.2, imagenet_preprocess = "resnet"):
    
    
    if split < 0.0:
        split = contains_mask.count(True)/len(contains_mask)
        
    while True:
        np_image_paths = np.array(image_paths)
        np_mask_paths = np.array(mask_paths)
        
        mask_index = [i for i, e in enumerate(contains_mask) if e]
        no_mask_index = [i for i, e in enumerate(contains_mask) if not e]

        random.shuffle(mask_index)
        
        mask_bs = int(batch_size*split)
        
        nrow, ncol, nchannel = get_shape(image_paths[0])
        
        for index in range(0, len(mask_index), mask_bs):
            end_index = min(index + mask_bs, len(mask_index))
            
            batch_image_paths = np_image_paths[mask_index[index:end_index]].tolist()
            batch_mask_paths = np_mask_paths[mask_index[index:end_index]].tolist()
            
            no_mask_bs = batch_size - len(batch_image_paths)
            
            no_mask_selection = random.sample(no_mask_index, no_mask_bs)
            batch_image_paths.extend(np_image_paths[no_mask_selection].tolist())
            batch_mask_paths.extend(np_mask_paths[no_mask_selection].tolist())
        
            batch_contains_mask = [True] * len(batch_image_paths)
            batch_contains_mask.extend([False] * no_mask_bs)
        
            batch_image_paths, batch_mask_paths, batch_contains_mask = shuffle_lists(batch_image_paths, batch_mask_paths, batch_contains_mask)
            
            images = np.ndarray((batch_size, nrow, ncol, nchannel), dtype = np.float32)
            masks = np.ndarray((batch_size, nrow, ncol, 1), dtype = np.float32)              

            for batch_index in range(batch_size):
                image = img_to_array(load_img(batch_image_paths[batch_index]))
                mask = img_to_array(load_img(batch_mask_paths[batch_index], grayscale = True))
                
                if augment:
                    if np.random.choice([False, True]):
                        image = np.fliplr(image)
                        mask = np.fliplr(mask)
                    rotateChoice = np.random.choice([0, 0, 1, 2, 3])
                    image = np.rot90(image, rotateChoice)
                    mask = np.rot90(mask, rotateChoice)
                    
                    if np.random.choice([False, False, True, True, True]):
                        image = random_channel_shift(image, 25.0, channel_axis=2)
                                    
                images[batch_index] = image
                masks[batch_index] = mask/255.0
                                
            images = preprocess_image(images, imagenet_preprocess)
                
            yield images, masks
            
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="Train deeplab segmenter", 
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("-t", "--train", help="list of input images/masks", required = True)
    parser.add_argument("-v", "--validation", help="list of input images/masks", required = True)
    
    parser.add_argument("-m", "--model", help="output trained model", required = True)
    parser.add_argument("-l", "--log", help="output log", required = True)
    parser.add_argument("-b", "--batchsize", help="batch size", type=int,required = False, default=8)
    parser.add_argument("-r", "--rate", help = "Learning rate", type=float, required=False, default=1.0E-3)
    parser.add_argument("-e", "--epochs", help = "number of epochs", type=int, required=False, default=50)
    parser.add_argument("--resnet", help="use a resnet unet", required = False,  action = 'store_true', default=False)
    parser.add_argument("--incres", help="use an inception_resnet unet", required = False,  action = 'store_true', default=False)
    parser.add_argument("--bcedice", help="use a bce + dice loss function", required = False,  action = 'store_true', default=True)
    parser.add_argument("--split", help="use this much masked/unmasked data in training", required = False,  type=float, default=0.625)
    parser.add_argument("--nbatches", help="Number of batches", type=int, required = False, default=1000)

    args = parser.parse_args()
    
    random.seed(7)
    np.random.seed(7)
    train_image_paths, train_mask_paths, train_weights = get_paths(args.train)
    validation_image_paths, validation_mask_paths, validation_weights = get_paths(args.validation)

    csv_logger = CSVLogger(args.log, append=True)
    checkpointer = ModelCheckpoint(filepath=args.model, verbose = 1, save_best_only = True)
        
    if args.resnet:
        model, model_type = get_unet_resnet(get_shape(train_image_paths[0]))
    elif args.incres:
        model, model_type = get_unet_inception_resnet_v2(get_shape(train_image_paths[0]))
    else:
        raise NotImplementedError("Has been removed")
        
    print(model.summary())
    print("number of layers = %d" % (len(model.layers)))
    
    if args.bcedice:
        model.compile(optimizer=Adam(lr=args.rate), loss=bce_overlap_loss, metrics=[overlap, binary_crossentropy])
    else:
        model.compile(optimizer=Adam(lr=args.rate), loss=overlap_loss, metrics=[overlap, binary_crossentropy])


    training_generator = weighted_generator(train_image_paths, train_mask_paths, train_weights,
                                   batch_size=args.batchsize, augment = True, imagenet_preprocess=model_type)
    validation_generator = basic_generator(validation_image_paths, validation_mask_paths,
                                           batch_size=args.batchsize, imagenet_preprocess=model_type)

    model.fit_generator(training_generator, steps_per_epoch = args.nbatches, 
                        nb_epoch=args.epochs, verbose=1,
                        callbacks = [checkpointer, csv_logger], validation_data = validation_generator,
                        validation_steps = len(validation_image_paths)//args.batchsize)