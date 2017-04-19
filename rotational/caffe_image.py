"""This module is a group of functions for dealing with images in general and how they relate to Caffe."""
import PIL.Image
import math
import numpy as np
import random

import cv2
from caffe.proto import caffe_pb2


def get_filename(coin_id, image_id,crop_dir):
    dir = crop_dir + str(coin_id / 100) + '/'
    filename = dir + str(coin_id).zfill(5) + str(image_id).zfill(2) + '.png'
    return filename

def get_filename_from(file_number,crop_dir):
    coin_id = file_number / 100
    image_id = file_number % 100
    return get_filename(coin_id, image_id,crop_dir)


def get_whole_rotated_image(crop, mask, angle, crop_size, before_rotate_size, scale):
    #Better for larger:
    #pixels_to_jitter = 35 * scale
    #For Dates:
    pixels_to_jitter = 4 #Old Way

    center_x = before_rotate_size / 2 + (random.random() * pixels_to_jitter * 2) - pixels_to_jitter
    center_y = before_rotate_size / 2 + (random.random() * pixels_to_jitter * 2) - pixels_to_jitter

    rot_image = crop.copy()
    rot_image = rotate(rot_image, angle, center_x, center_y, before_rotate_size, before_rotate_size)
    # This is hard coded for 28x28.
    rot_image = cv2.resize(rot_image, (41, 41), interpolation=cv2.INTER_AREA)
    rot_image = rot_image[6:34, 6:34]

    # rot_image = rot_image * mask
    return rot_image

def get_circle_mask(crop_size):
    mask = np.zeros((crop_size, crop_size), dtype=np.uint8)
    cv2.circle(mask, (crop_size / 2, crop_size / 2), (crop_size / 2), 1, cv2.cv.CV_FILLED, lineType=8, shift=0)
    return mask


def center_rotate(img, angle):
    rows, cols = img.shape
    m = cv2.getRotationMatrix2D((cols / 2, rows / 2), angle, 1)
    cv2.warpAffine(img, m, (cols, rows), img, cv2.INTER_CUBIC)
    return img


def rotate(img, angle, center_x, center_y, rows, cols):
    m = cv2.getRotationMatrix2D((center_x, center_y), angle, 1)
    cv2.warpAffine(img, m, (cols, rows), img, cv2.INTER_CUBIC)
    return img


def get_rotated_crop(crop_dir, crop_id, crop_size, angle):
    filename = get_filename_from(crop_id,crop_dir)
    crop = cv2.imread(filename)
    if crop == None:
        print crop_id, 'None'
        return None

    crop = cv2.resize(crop, (crop_size, crop_size), interpolation=cv2.INTER_AREA)
    m = cv2.getRotationMatrix2D((crop_size / 2, crop_size / 2), angle, 1)
    cv2.warpAffine(crop, m, (crop_size, crop_size), crop, cv2.INTER_CUBIC)
    return crop


def rotate_point(angle, center_x, center_y, point_x, point_y):
    rotated_x = ((point_x - center_x) * math.cos(angle)) - ((point_y - center_y) * math.sin(angle)) + center_x
    rotated_y = ((point_x - center_x) * math.sin(angle)) + ((point_y - center_y) * math.cos(angle)) + center_y
    return rotated_x, rotated_y


def rotate_matrix(angle, center_x, center_y, mat):
    rotated = mat.copy()
    # OMG I need to learny my matrix math in Python!
    for num in range(0, 4):
        rotated_x, rotated_y = rotate_point(math.radians(angle), center_x, center_y, mat[num, 0], mat[num, 1])
        rotated[num, 0] = rotated_x
        rotated[num, 1] = rotated_y
    return rotated


def save_image(image, filename):
    # converting from BGR to RGB
    image = image[[2, 1, 0], ...]  # channel swap
    # convert to (height, width, channels)
    image = image.astype('uint8').transpose((1, 2, 0))
    image = PIL.Image.fromarray(image)
    image.save(filename)


def save_mean(mean, filename):
    """
    Saves mean to file

    Arguments:
    mean -- the mean as an np.ndarray
    filename -- the location to save the image
    """
    if filename.endswith('.binaryproto'):
        blob = caffe_pb2.BlobProto()
        blob.num = 1
        blob.channels = mean.shape[0]
        blob.height = mean.shape[1]
        blob.width = mean.shape[2]
        blob.data.extend(mean.astype(float).flat)
        with open(filename, 'wb') as outfile:
            outfile.write(blob.SerializeToString())

    elif filename.endswith(('.jpg', '.jpeg', '.png')):
        save_image(mean, filename)
    else:
        raise ValueError('unrecognized file extension')


def get_caffe_image(crop, crop_size):
    # this is how you get the image from file:
    # coinImage = [caffe.io.load_image("some file", color=False)]

    caffe_image = cv2.resize(crop, (crop_size, crop_size), interpolation=cv2.INTER_AREA)
    caffe_image = caffe_image.astype(np.float32) / 255
    caffe_image = np.array(caffe_image).reshape(crop_size, crop_size, 1)
    # Caffe wants a list so []:
    caffe_images = [caffe_image]
    return caffe_images


def get_angle_sequence(length, test_id):
    random.seed(test_id)
    angles = []
    for count in range(0, length):
        angle = float(count) / (length / 360)
        class_angle = int(round(angle))
        if class_angle == 360:
            class_angle = 0

        if test_id == -1:
            angles.append([random.random(), angle, class_angle])

        if test_id == 0:
            if 0 <= class_angle < 30:
                angles.append([random.random(), angle, class_angle])
            if 330 <= class_angle < 360:
                angles.append([random.random(), angle, class_angle])
        if class_angle == 50:
            pass

        if test_id == 1:
            if 30 <= class_angle < 60:
                angles.append([random.random(), angle, class_angle])
            if 300 <= class_angle < 330:
                angles.append([random.random(), angle, class_angle])

        if test_id == 2:
            if 60 <= class_angle < 90:
                angles.append([random.random(), angle, class_angle])
            if 270 <= class_angle < 300:
                angles.append([random.random(), angle, class_angle])

        if test_id == 3:
            if 90 <= class_angle < 120:
                angles.append([random.random(), angle, class_angle])
            if 240 <= class_angle < 270:
                angles.append([random.random(), angle, class_angle])

        if test_id == 4:
            if 120 <= class_angle < 150:
                angles.append([random.random(), angle, class_angle])
            if 210 <= class_angle < 240:
                angles.append([random.random(), angle, class_angle])

        if test_id == 5:
            if 150 <= class_angle < 210:
                angles.append([random.random(), angle, class_angle])

    angles.sort()
    return angles


def get_composite_image(images, rows,cols):
    crop_rows, crop_cols, channels = images[0].shape
    composite_rows = crop_rows * rows
    composite_cols = crop_cols * cols
    composite_image = np.zeros((composite_rows, composite_cols, 3), np.uint8)
    key = 0
    for x in range(0, rows):
        for y in range(0, cols):
            key += 1
            if len(images) <= key:
                break
            if images[key] is not None:
                composite_image[x * crop_rows:((x + 1) * crop_rows), y * crop_cols:((y + 1) * crop_cols)] = images[key]
    return composite_image


def get_formated_angle(angle):
    angle %= 360
    if angle < -179:
        angle += 360
    if angle > 180:
        angle -= 360
    return angle
