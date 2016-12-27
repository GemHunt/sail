import os
import random
import shutil
import sys
import time
from random import randint

import cv2

import caffe_image as ci
import caffe_lmdb

sys.path.append('/home/pkrush/caffe/python')
sys.path.append('/home/pkrush/digits')

# Find the best implementation available
try:
    from cStringIO import StringIO
except ImportError:
    from StringIO import StringIO

import lmdb
import numpy as np

if __name__ == '__main__':
    dirname = os.path.dirname(os.path.realpath(__file__))
    sys.path.insert(0, os.path.join(dirname, '..', '..'))

# Import digits.config first to set the path to Caffe
from caffe.proto import caffe_pb2


def create_lmdbs(filedata, lmdb_dir, images_per_angle, test_id, create_val_set=True, create_files=False):
    start_time = time.time()

    max_images = 99999999
    crop_size = 28
    before_rotate_size = 56
    classes = 360
    if os.path.exists(lmdb_dir):
        shutil.rmtree(lmdb_dir)

    if not os.path.exists(lmdb_dir):
        os.makedirs(lmdb_dir)

    if create_files:
        img_dir = '/home/pkrush/img-files'

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

    # for filename in glob.iglob('/home/pkrush/copper/test/*.jpg'):
    for image_id, filename, angle_offset in filedata:
        print image_id
        # imageid = filename[-9:]
        # imageid = imageid[:5]
        id += 1
        if id > max_images - 1:
            break

        # crop = cv2.imread('/home/pkrush/copper/test.jpg')
        crop = cv2.imread(filename)
        if crop is None:
            continue

        # images are 256 x 256, Center is 128,128, crop 56x56 from center:
        # crop = crop[100:156,100:156]

        # images are 256 x 256, Center is 128,128, crop 160x160 from center:
        crop = crop[48:208, 48:208]

        crop = cv2.resize(crop, (before_rotate_size, before_rotate_size), interpolation=cv2.INTER_AREA)
        crop = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)

        crops.append(crop)
        mask = ci.get_circle_mask(crop_size)

        phase = 'train'
        if create_val_set:
            train_vs_val = randint(1, 4)
            if train_vs_val != 4:
                phase = 'train'
            if train_vs_val == 4:
                phase = 'val'

    key = 0

    number_of_angles = images_per_angle * 360
    if not create_val_set:
        # You have to more test images if this is the test set:
        number_of_angles *= len(crops)
    angles = ci.get_angle_sequence(number_of_angles, test_id)

    for random_float, angle, class_angle in angles:
        random_index = random.randint(0, len(crops) - 1)
        crop = crops[random_index]
        image_id = filedata[random_index][0]
        angle_offset = filedata[random_index][2]

        angle_to_rotate = angle + angle_offset
        if angle_to_rotate > 360:
            angle_to_rotate - 360

        rot_image = ci.get_whole_rotated_image(crop, mask, angle_to_rotate, crop_size)

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

    caffe_lmdb.write_batch_to_lmdb(train_image_db, train_image_batch)
    caffe_lmdb.write_batch_to_lmdb(val_image_db, val_image_batch)
    train_image_batch = []
    val_image_batch = []

    # label_batch = []

    train_image_db.close()
    val_image_db.close()
    # label_db.close()

    # save mean
    mean_image = (image_sum / (id + 1) * images_per_angle * 360).astype('uint8')
    ci.save_mean(mean_image, os.path.join(lmdb_dir, 'mean.binaryproto'))
    print 'Done after %s seconds' % (time.time() - start_time,)

    return
