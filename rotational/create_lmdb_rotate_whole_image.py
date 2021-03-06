import numpy as np
import os
import random
import shutil
import time
from random import randint

import cv2
import lmdb
from caffe.proto import caffe_pb2

import caffe_image as ci
import caffe_lmdb


def create_all_lmdbs(args):
    filedata, lmdb_dir, images_per_angle = args
    create_lmdbs (filedata, lmdb_dir, images_per_angle,False)

def create_lmdbs(filedata, lmdb_dir, images_per_angle, create_val_set=True, create_files=False):
    start_time = time.time()
    print 'Creating lmdb for:' , lmdb_dir
    img_dir = '/home/pkrush/img-files'
    max_images = 99999999
    crop_size = 28
    before_rotate_size = 56
    classes = 360
    mask = None
    #radii = [28, 42, 64, 96, 146, 224]
    #file_radius = 224

    # For Dates:
    radii = [28]
    file_radius = 28

    if os.path.exists(lmdb_dir):
        shutil.rmtree(lmdb_dir)

    if not os.path.exists(lmdb_dir):
        os.makedirs(lmdb_dir)

    if create_files:
        if os.path.exists(img_dir):
            shutil.rmtree(img_dir)
        if not os.path.exists(img_dir):
            os.makedirs(img_dir)
        for x in range(0, classes):
            img_dir_class = img_dir + '/' + str(1000 + x)
            if not os.path.exists(img_dir_class):
                os.makedirs(img_dir_class)

    # create DBs
    train_image_db = lmdb.open(os.path.join(lmdb_dir, 'train_db'), map_async=True, max_dbs=0)
    val_image_db = lmdb.open(os.path.join(lmdb_dir, 'val_db'), map_async=True, max_dbs=0)

    # add up all images to later create mean image
    image_sum = np.zeros((1, crop_size, crop_size), 'float64')

    # arrays for image and label batch writing
    train_image_batch = []
    val_image_batch = []
    id = -1
    crops = []

    # for filename in glob.iglob('/home/pkrush/copper/test/*.png'):
    for image_id, filename, angle_offset in filedata:
        id += 1
        if id > max_images - 1:
            break
        cap = cv2.imread(filename)
        if cap is None:
            print 'Image is bad'
            continue
        gray = cv2.cvtColor(cap, cv2.COLOR_BGR2GRAY)

        rows, cols = gray.shape
        if not rows == cols == file_radius * 2:
            print 'This is hard coded to use 448x448 images'

        sized_crops = []
        for radius in radii:
            # crop = gray[144:304, 144:304]
            # graycopy = gray.copy()
            # if radius== 224:
            #  x_jitter = 0
            #   y_jitter = 0
            # else:
            # pixels_to_jitter = 10
            #  x_jitter = (random.random() * pixels_to_jitter * 2) - pixels_to_jitter
            #   y_jitter = (random.random() * pixels_to_jitter * 2) - pixels_to_jitter

            #crop = gray[(file_radius - radius)+x_jitter :(file_radius + radius)+x_jitter, (file_radius - radius)+y_jitter:(file_radius + radius)+y_jitter]
            crop = gray[file_radius - radius:file_radius + radius, file_radius - radius:file_radius + radius]
            crop = cv2.resize(crop, (before_rotate_size, before_rotate_size), interpolation=cv2.INTER_AREA)
            sized_crops.append(crop)
        crops.append(sized_crops)

        # mask = ci.get_circle_mask(crop_size)

        phase = 'train'
        if create_val_set:
            train_vs_val = randint(1, 4)
            if train_vs_val != 4:
                phase = 'train'
            if train_vs_val == 4:
                phase = 'val'
    key = 0

    number_of_angles = int(images_per_angle * 360)
    if not create_val_set:
        # You have to more test images if this is the test set:
        number_of_angles *= len(crops)
    angles = ci.get_angle_sequence(number_of_angles, -1)
    #For Dates: Run with the old testID of 0 to get -29 to 29 degrees:
    #angles = ci.get_angle_sequence(number_of_angles, 0)

    print 'len(angles)',len(angles)
    for random_float, angle, class_angle in angles:
        #The rotate(80us), the encode(40us), and drive write(80ms) take up the most amount of time here.
        loop_time = time.time()
        random_index = random.randint(0, len(crops) - 1)
        random_sized_scale_index = random.randint(0, len(radii) - 1)
        crop = crops[random_index][random_sized_scale_index]
        image_id = filedata[random_index][0]
        angle_offset = filedata[random_index][2]
        angle_to_rotate = angle + angle_offset
        scale = (crop_size / 2) / float(radii[random_sized_scale_index])
        rot_image = ci.get_whole_rotated_image(crop, mask, angle_to_rotate, crop_size, before_rotate_size, scale)

        if create_files:
            cv2.imwrite(img_dir + '/' + str(class_angle + 1000) + '/' + str(id) + str(angle).zfill(5) + '.png',
                        rot_image)

        datum = caffe_pb2.Datum()
        datum.data = cv2.imencode('.png', rot_image)[1].tostring()
        # datum.data = cv2.imencode('.png', rot_image).tostring()
        datum.label = int(class_angle)
        datum.encoded = 1

        rot_image = rot_image.reshape(1, crop_size, crop_size)
        image_sum += rot_image
        # datum = caffe.io.array_to_datum(rot_image, class_angle)

        # key_string = '{:08}'.format((id * 100000) +  count)
        # key = '{:08}'.format(angle)
        # str_id = str(randint(0, 9999999)) + ',' + str(image_id) + ',' + str(class_angle)
        # str_id = '{:03}'.format(image_id % 1000) + '{:05}'.format(key)
        # 00000000_123 is the key digits makes, but I still! don't know if I can change that.
        str_id = '{:05}'.format(key) + ',' + '{:05}'.format(image_id) + ',' + str(class_angle)

        key += 1

        if create_val_set:
            train_vs_val = randint(1, 4)
            if train_vs_val != 4:
                phase = 'train'
            if train_vs_val == 4:
                phase = 'val'

        if phase == 'train':
            # train_image_batch.append([str(image_id) + "," + str(angle + 1000), datum])
            train_image_batch.append([str_id.encode('ascii'), datum])

        if phase == 'val':
            # val_image_batch.append([str(image_id) + "," + str(angle + 1000), datum])
            val_image_batch.append([str_id.encode('ascii'), datum])
        if len(train_image_batch) > 10000:
            loop_time = time.time()
            caffe_lmdb.write_batch_to_lmdb(train_image_db, train_image_batch)
            caffe_lmdb.write_batch_to_lmdb(val_image_db, val_image_batch)
            #print 'Write batch after %s micro seconds' % ((time.time() - loop_time)*1000000,)
            train_image_batch = []
            val_image_batch = []


    if len(train_image_batch) > 0:
        caffe_lmdb.write_batch_to_lmdb(train_image_db, train_image_batch)
        caffe_lmdb.write_batch_to_lmdb(val_image_db, val_image_batch)

    # label_batch = []

    train_image_db.close()
    val_image_db.close()
    # label_db.close()

    # save mean
    mean_image = (image_sum / (id + 1) * images_per_angle * 360).astype('uint8')
    ci.save_mean(mean_image, os.path.join(lmdb_dir, 'mean.binaryproto'))
    print lmdb_dir, 'Done after %s seconds' % (time.time() - start_time,)

    return
