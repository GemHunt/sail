"""
A set of functions to test out self supervised rotated coin image models
"""

import cPickle as pickle
import glob
import numpy as np
import operator
import os
import random
import shutil
import subprocess
import sys
import time
from itertools import islice
from subprocess import Popen
from multiprocessing import Pool

import create_lmdb_rotate_whole_image
import image_set
import summarize_rotated_crops

home_dir = '/home/pkrush/cent-models/'
crop_dir = '/home/pkrush/cents-test/'

# For Dates:
# home_dir = '/home/pkrush/cent-date-models/'
# crop_dir = '/home/pkrush/cent-dates/'

data_dir = home_dir + 'metadata/'
train_dir = home_dir + 'train/'
test_dir = home_dir + 'test/'

def init_dir():
    directories = [home_dir, data_dir, crop_dir, train_dir, test_dir]
    make_dir(directories)


def make_dir(directories):
    for path_name in directories:
        if not os.path.exists(path_name):
            os.makedirs(path_name)

def create_new_index(count, total_images, index_name):
    # This assumes there are no gaps in the image_ids
    seed_image_ids = [random.randint(0, total_images) for x in range(count)]
    #seed_image_ids = range(0, 100)
    #seed_image_ids = [0, 100]
    pickle.dump(seed_image_ids, open(data_dir + index_name + '.pickle', "wb"))

def create_seed_and_test_random(factor, start_id):
    # Only use 1/factor of the crop images
    # for example there are 10000 crops and a factor of 100
    #then only 100 of them would be the random seed and test images.
    # A factor of 0 would be 100%
    # This should be changed to percent!
    crops = []
    image_ids = []
    for filename in glob.iglob(crop_dir + '*.png'):
        crops.append(filename)

    for filename in crops:
        renamed = filename.replace("_", "")
        image_id = int(renamed.replace('.png', '').replace('/home/pkrush/cents/', ''))
        if image_id < start_id:
            continue
        renamed = crop_dir + str(image_id) + '.png'
        os.rename(filename, renamed)
        rand_int = random.randint(0, factor)
        if rand_int == 0:
            image_ids.append(image_id)
    pickle.dump(image_ids, open(data_dir + 'seed_image_ids.pickle', "wb"))
    pickle.dump(image_ids, open(data_dir + 'test_image_ids.pickle', "wb"))


def create_test_ids_from_multi_point(size_limit=-1):
    save_multi_point_ids()
    test_image_ids = []
    seed_image_ids = pickle.load(open(data_dir + 'multi_point_ids.pickle', "rb"))

    for coin_id, image_ids in seed_image_ids.iteritems():
        if size_limit > 0 and size_limit <= coin_id:
            continue
        # for only one random test image:
        # test_image_id = image_ids[random.randint(0, len(image_ids) - 1)]
        # test_image_ids.append(coin_id * 100 + test_image_id)
        for image_id in image_ids:
            test_image_ids.append(coin_id * 100 + image_id)
    pickle.dump(test_image_ids, open(data_dir + 'test_image_ids.pickle', "wb"))

def get_filename(coin_id, image_id):
    dir = crop_dir + str(coin_id / 100) + '/'
    filename = dir + str(coin_id).zfill(5) + str(image_id).zfill(2) + '.png'
    return filename

def get_filename_from(file_number):
    coin_id = file_number / 100
    image_id = file_number % 100
    return get_filename(coin_id, image_id)


def get_seed_image_ids():
    seed_image_ids = pickle.load(open(data_dir + 'seed_image_ids.pickle', "rb"))
    return sorted(set(seed_image_ids) - set(image_set.widened_seeds))

    # test_image_ids = pickle.load(open(data_dir + 'test_image_ids.pickle', "rb"))
    # seed_image_ids = seed_image_ids + test_image_ids[0:180]
    # seed_image_ids = seed_image_ids + image_set.widened_seeds()
    # pickle.dump(seed_image_ids, open(data_dir + 'seed_image_ids.pickle', "wb"))


def get_test_image_ids():
    test_image_ids = pickle.load(open(data_dir + 'test_image_ids.pickle', "rb"))
    return sorted(set(test_image_ids) - set(image_set.widened_seeds))

    # test_image_ids += get_seed_image_ids()
    # test_image_ids += image_set.widened_seeds()
    # test_image_ids = list(set(test_image_ids))
    # pickle.dump(test_image_ids, open(data_dir + 'test_image_ids.pickle', "wb"))

def rename_crops():
    crops = []
    for filename in glob.iglob(crop_dir + '*.png'):
        crops.append([random.random(), filename])
    crops.sort()
    pickle.dump(crops, open(data_dir + 'copper_crops.p', "wb"))
    key = 0
    for rand, filename in crops:
        key += 1
        os.rename(filename, crop_dir + str(key) + '.png')


def save_multi_point_ids():
    multi_point_ids = {}
    for root, dirnames, filenames in os.walk(crop_dir):
        for filename in filenames:
            if filename.endswith('.png'):
                file_number = int(filename.replace('.png', ''))
                coin_id = int(file_number / 100)
                image_id = file_number % 100
                if coin_id not in multi_point_ids.iterkeys():
                    multi_point_ids[coin_id] = []
                multi_point_ids[coin_id].append(image_id)
    pickle.dump(multi_point_ids, open(data_dir + 'multi_point_ids.pickle', "wb"))


def rename_multi_point_crops():
    crops = []
    test_image_ids = []
    for filename in glob.iglob(crop_dir + '*.png'):
        crops.append(filename)
    crops.sort()
    for filename in crops:
        renamed = filename.replace("_","")
        image_id = int(renamed.replace('.png', '').replace(crop_dir, ''))
        renamed = crop_dir + str(image_id) + '.png'
        os.rename(filename,renamed)
        test_image_ids.append(image_id)
    pickle.dump(test_image_ids, open(data_dir + 'test_image_ids.pickle', "wb"))


def save_multi_point_crops():
    crops = []
    test_image_ids = []
    for filename in glob.iglob(crop_dir + '*.png'):
        crops.append(filename)
    crops.sort()
    for filename in crops:
        renamed = filename.replace("_", "")
        image_id = int(renamed.replace('.png', '').replace(crop_dir, ''))
        renamed = crop_dir + str(image_id) + '.png'
        os.rename(filename, renamed)
        test_image_ids.append(image_id)
    pickle.dump(test_image_ids, open(data_dir + 'test_image_ids.pickle', "wb"))


def copy_file(filename, path_name):
    with open(filename, 'r') as myfile:
        data = myfile.read().replace('replace_dir_name_', path_name)
    with open(path_name + filename, 'w') as file_:
        file_.write(data)

def copy_train_files(lmdb_dir,multi_image_training = False):
    if multi_image_training:
        copy_file('solver-multi.prototxt', lmdb_dir)
    else:
        copy_file('solver.prototxt', lmdb_dir)
    copy_file('train_val.prototxt', lmdb_dir)
    copy_file('deploy.prototxt', lmdb_dir)
    copy_file('labels.txt', lmdb_dir)


def create_train_script(lmdb_dir, weight_filename,multi_image_training):
    log_filename = 'caffe_output.log'
    shell_script = 'cd ' + lmdb_dir + '\n'
    shell_script += '/home/pkrush/caffe/build/tools/caffe '
    shell_script += 'train '
    if multi_image_training:
        shell_script += '-solver ' + lmdb_dir + 'solver-multi.prototxt '
    else:
        shell_script += '-solver ' + lmdb_dir + 'solver.prototxt '
    shell_script += '-weights ' + weight_filename + ' '
    shell_script += '2> ' + lmdb_dir + log_filename + ' \n'
    shell_script += 'grep accu ' + log_filename + ' \n'
    shell_filename = lmdb_dir + 'train-single-coin-lmdbs.sh'
    create_shell_script(shell_filename, shell_script)
    return shell_filename


def get_single_lmdb_filedata(seed_id, max_value_cutoff):
    seeds = pickle.load(open(data_dir + 'seed_data.pickle', "rb"))
    filedata = []
    values = seeds[seed_id]

    # this is handy for large groups (heads,tails)
    # values.sort(key=lambda x: x[0], reverse=True)
    # best_results_by_angle_group = {}
    # for max_value, angle, image_id in values:
    # rounded_angle = int(round(angle / 5) * 5)
    # if not rounded_angle in best_results_by_angle_group.keys():
    # best_results_by_angle_group[rounded_angle] = [max_value, angle, image_id]
    # else:
    # if max_value > best_results_by_angle_group[rounded_angle][0]:
    # best_results_by_angle_group[rounded_angle] = [max_value, angle, image_id]
    # values = best_results_by_angle_group.values()

    filedata.append([seed_id, crop_dir + str(seed_id) + '.png', 0])

    for image_id, test_values in values.iteritems():
        max_value, angle = test_values
        if max_value > max_value_cutoff:
            filedata.append([image_id, crop_dir + str(image_id) + '.png', angle])

    return filedata


def get_single_lmdb_multi_point_filedata(seed_id, max_value_cutoff, multi_point_error_test_image_ids):
    # multi_point_error_test_image_ids really should be passed in per seed_id
    seeds = pickle.load(open(data_dir + 'seed_data.pickle', "rb"))
    filedata = []
    expanded_filedata = []
    coin_ids = []
    values = seeds[seed_id]

    filedata.append([seed_id, crop_dir + str(seed_id) + '.png', 0])

    for image_id, test_values in values.iteritems():
        max_value, angle = test_values
        if max_value > max_value_cutoff:
            if image_id not in multi_point_error_test_image_ids:
                filedata.append([image_id, crop_dir + str(image_id) + '.png', angle])

    multi_point_ids = pickle.load(open(data_dir + 'multi_point_ids.pickle', "rb"))
    for test_image_id, image_path, angle in filedata:
        coin_id = int(test_image_id / 100)
        if coin_id in coin_ids:
            continue
        coin_ids.append(coin_id)
        for crop_id in multi_point_ids[coin_id]:
            test_image_id = coin_id * 100 + crop_id
            if test_image_id not in multi_point_error_test_image_ids:
                # the angles should not vary for the same coin id:
                expanded_filedata.append([test_image_id, crop_dir + str(test_image_id) + '.png', angle])
    return expanded_filedata


# delete this function:
def create_single_lmdbs(seed_image_ids):
    weight_filename = 'starting-weights.caffemodel'
    shutil.copyfile(weight_filename, train_dir + weight_filename)
    shell_filenames = []
    for image_id in seed_image_ids:
        print 'Creating single lmdb for ' + str(image_id)
        filedata = [[image_id, crop_dir + str(image_id) + '.png', 0]]
        lmdb_dir = train_dir + str(image_id) + '/'
        create_lmdb_rotate_whole_image.create_lmdbs(filedata, lmdb_dir, 100, True, False)
        copy_train_files(lmdb_dir)
        shell_filename = create_train_script(lmdb_dir, train_dir + weight_filename, False)
        shell_filenames.append(shell_filename)
    create_script_calling_script(train_dir + 'train_all.sh', shell_filenames)


def create_single_lmdb(seed_image_id, filedata, test_id, multi_image_training=False, images_per_angle=500,
                       retraining=False):
    start_time = time.time()
    print 'create_single_lmdb for ' + str(seed_image_id)

    if retraining:
        weight_filename = 'snapshot_iter_16880.caffemodel'
        shutil.copyfile(train_dir + str(seed_image_id) + '/' + weight_filename, train_dir + weight_filename)
    else:
        weight_filename = 'starting-weights.caffemodel'
        shutil.copyfile(weight_filename, train_dir + weight_filename)

    lmdb_dir = train_dir + str(seed_image_id) + '/'

    create_lmdb_rotate_whole_image.create_lmdbs(filedata, lmdb_dir, images_per_angle, True, False)
    copy_train_files(lmdb_dir, multi_image_training)
    create_train_script(lmdb_dir, train_dir + weight_filename, multi_image_training)
    print 'Done in %s seconds' % (time.time() - start_time,)

def create_test_script(image_id,test_id,multi_image_training = False ):
    shell_script = 'cd ' + train_dir + str(image_id) + '/\n'
    shell_script += '/home/pkrush/caffe/.build_release/examples/cpp_classification/classification.bin '
    shell_script += 'deploy.prototxt '
    if multi_image_training:
        shell_script += 'snapshot_iter_16880.caffemodel '
    else:
        shell_script += 'snapshot_iter_844.caffemodel '
    shell_script += 'mean.binaryproto '
    shell_script += 'labels.txt '
    shell_script += test_dir + str(test_id) + '/train_db/data.mdb '
    shell_script += '> ' + test_dir + str(test_id) + '/' + str(image_id) + '.dat\n'
    shell_filename = test_dir + str(test_id) + '/test-' + str(image_id) + '.sh'
    create_shell_script(shell_filename, shell_script)
    return  shell_filename


def create_shell_script(filename, shell_script):
    shell_script = '#!/bin/bash\n' + 'echo Entered ' + filename + '\n' + shell_script
    shell_script = shell_script + 'echo Exited ' + filename + '\n'

    with open(filename, 'w') as file_:
        file_.write(shell_script)
    fd = os.open(filename, os.O_RDONLY)
    os.fchmod(fd, 0755)
    os.close(fd)


def create_script_calling_script(filename, shell_filenames):
    shell_script = ''
    for shell_filename in shell_filenames:
        shell_script += shell_filename + '\n'
    create_shell_script(filename, shell_script)


def read_test(test_batch_ids,image_ids):
    all_results_filename = data_dir + 'all_results.pickle'
    all_results = []
    new_all_results = []
    if os.path.exists(all_results_filename):
        all_results = pickle.load(open(data_dir + 'all_results.pickle', "rb"))

    # If only one image is being read remove the old image, else output all new results
    if len(image_ids) == 1:
        for results in all_results:
            if len(results) == 0:
                print('No test results, the network or test image was bad on ' + str(image_ids[0]) + '\n')
                continue

            if results[0] != image_ids[0]:
                new_all_results.append(results)

    for test_batch_id in test_batch_ids:
        for image_id in image_ids:
            filename = test_dir + str(test_batch_id) + '/' + str(image_id) + '.dat'

            if not os.path.isfile(filename):
                continue
            results = summarize_rotated_crops.get_results(filename, image_id)
            # results = summarize_whole_rotated_model_results.summarize_whole_rotated_model_results(
            #                                                                  filename, image_id,low_angle,high_angle)
            new_all_results.append(results)
    pickle.dump(new_all_results, open(data_dir + 'all_results.pickle', "wb"))


def widen_model(seed_image_id, max_test_id, max_value_cutoff):
    for test_id in range(0, max_test_id + 1):
        for x in range(0, 2):
            filedata = get_single_lmdb_filedata(seed_image_id, max_value_cutoff)
            run_train_test(seed_image_id, filedata, max_value_cutoff, test_id, True)
    image_set.create_composite_images(crop_dir, data_dir, 120, 8, 20)

def run_train_test(seed_image_id, filedata,max_value_cutoff, test_id, multi_image_training = False):
    create_single_lmdb(seed_image_id, filedata, test_id, multi_image_training)
    run_script(train_dir + str(seed_image_id) + '/train-single-coin-lmdbs.sh')
    for test_id in range(0, test_id + 1):
        create_test_script(seed_image_id, test_id, multi_image_training)
        run_script(test_dir + str(test_id) + '/test-' + str(seed_image_id) + '.sh')

def run_test(seed_image_id, max_value_cutoff, test_id):
    for test_id in range(0, test_id + 1):
        run_script(test_dir + str(test_id) + '/test-' + str(seed_image_id) + '.sh')
    read_test([seed_image_id], test_id)
    image_set.read_results(max_value_cutoff, data_dir, [seed_image_id])

def test_all(seed_image_ids):
    for seed_image_id in seed_image_ids:
        for test_id in range(0, 6):
            run_script(test_dir + str(test_id) + '/test-' + str(seed_image_id) + '.sh')

    read_test(seed_image_ids, 5)
    read_all_results(0, seed_image_ids)

def run_script(filename):
    print "Running " + filename
    subprocess.call(filename)


def run_scripts(filenames, max_workers):
    # This results in a 3.14x speed up using 6 workers to classify because the classification tool is CPU heavy
    start_time = time.time()
    processes = (Popen(cmd, shell=True) for cmd in filenames)
    run_processes(processes, max_workers)

def run_processes(processes, max_workers):
    running_processes = list(islice(processes, max_workers))  # start new processes
    while running_processes:
        time.sleep(.001)  # Sleep for a ms so this loop does not waste CPU
        for i, process in enumerate(running_processes):
            if process.poll() is not None:  # the process has finished
                running_processes[i] = next(processes, None)  # start new process
                if running_processes[i] is None:  # no new processes
                    del running_processes[i]
                    break
    print 'All processes run in %s seconds' % (time.time() - start_time,)

def create_new_indexes(total_new_seed_imgs, total_new_test_imgs):
    seeds = pickle.load(open(data_dir + 'seed_data.pickle', "rb"))
    seed_image_ids = []
    test_image_ids = []
    count = 0

    for seed_image_id, values in seeds.iteritems():
        values.sort(key=lambda x: x[0], reverse=False)
        # seed_image_ids.append(values[0:total_new_seed_imgs][2])
        # test_image_ids.append(values[total_new_seed_imgs:total_new_seed_imgs+total_new_test_imgs][2])

        for max_value, angle, image_id in values:
            count += 1
            if count < total_new_seed_imgs:
                seed_image_ids.append(image_id)
            else:
                if count < total_new_seed_imgs + total_new_test_imgs:
                    test_image_ids.append(image_id)
        count = 0
    pickle.dump(seed_image_ids, open(data_dir + 'seed_image_ids.pickle', "wb"))
    pickle.dump(test_image_ids, open(data_dir + 'test_image_ids.pickle', "wb"))


def save_graph():
    image_set.set_angles_postive()
    nodes = image_set.get_nodes()
    edges = image_set.get_edges()
    pickle.dump(nodes, open(data_dir + 'nodes.pickle', "wb"))
    pickle.dump(edges, open(data_dir + 'edges.pickle', "wb"))


def read_all_results(cut_off=0, seed_image_ids=None, seeds_share_test_images=True, remove_widened_seeds=False):
    image_set.read_results(cut_off, data_dir, seed_image_ids, seeds_share_test_images, remove_widened_seeds)
    # image_set.create_composite_images(crop_dir, data_dir, 140, 150, 10)


def retrain_widened_seed(seed_image_id, cut_off):
    # This did not seem to add much to the model
    # usage: retrain_widened_seed(7855,27)
    max_value_cutoff = 10
    read_all_results(max_value_cutoff, seeds_share_test_images=False, remove_widened_seeds=True)
    filedata = get_single_lmdb_filedata(seed_image_id, cut_off)
    run_train_test(seed_image_id, filedata, max_value_cutoff, test_id=5,multi_image_training =  True)
    read_all_results(max_value_cutoff, seeds_share_test_images=False, remove_widened_seeds=True)
    image_set.create_composite_images(crop_dir, data_dir, crop_size=140, rows=50, cols=10)


# Manually run and edit functions below:   ****************************************
def build_init_rotational_networks():
    # This function is meant to be edited and run manually for now.
    # This starts from scratch: Only square images need to exist in the crop dir
    # init_dir()
    #create_new_index(20, 13926, 'seed_image_ids')
    #create_new_index(500, 13926, 'test_image_ids')
    # create_seed_and_test_random(25)
    # seeds = get_seed_image_ids()
    #create_single_lmdbs(seeds)
    create_test_lmdbs(0)
    # create_all_test_lmdbs()
    #run_script(train_dir + 'train_all.sh')
    run_script(test_dir + 'test_all.sh')
    read_test(get_seed_image_ids(), 0)
    read_all_results(10, seeds_share_test_images=False, remove_widened_seeds=True)


def link_seed_by_graph(seed_id, cut_off, min_connections, max_depth):
    image_set.read_results(cut_off, data_dir, seeds_share_test_images=False, remove_widened_seeds=True)
    image_set.save_widened_seeds(data_dir, seed_id, cut_off)
    image_set.read_results(cut_off, data_dir, seeds_share_test_images=True, remove_widened_seeds=True)
    save_graph()
    most_connected_seeds = image_set.find_most_connected_seeds(data_dir, seed_id, min_connections, max_depth)
    filedata = []
    if len(most_connected_seeds) != 0:
        # image_set.create_composite_image(crop_dir, data_dir, 130, 30, 10, most_connected_seeds.iterkeys())
        for seed_image_id, values in most_connected_seeds.iteritems():
            print values
            filedata.append([seed_image_id, crop_dir + str(seed_image_id) + '.png', values[2]])
    print 'Count of images linked by graph:', len(most_connected_seeds)
    image_set.create_composite_image_from_filedata(crop_dir, data_dir, 140, rows=150, cols=10, filedata=filedata)
    if len(filedata) > 9:
        run_train_test(seed_id, filedata, cut_off, test_id=5, multi_image_training=True)
        run_test(seed_id, cut_off, test_id=5)
        read_all_results(cut_off)
    else:
        print 'Not enough seeds found'


def get_errors_and_angles(min_good_images_per_seed= 30, angle_tolerance=6):
    # Yes this function is doing too much!
    # Find all test_image_ids that don't match the major class
    # Find all test_image_ids that the angle is off where the major class is correct
    # Find the average good angles


    seeds = pickle.load(open(data_dir + 'seed_data.pickle', "rb"))
    coin_results = {}
    error_test_image_ids = []
    all_error_test_image_ids = []
    average_coin_angles = {}

    for seed_image_id, images in seeds.iteritems():
        for test_image_id, values in images.iteritems():
            max_value, angle = values
            coin_id = int(test_image_id / 100)
            if coin_id not in coin_results.iterkeys():
                coin_results[coin_id] = []
            coin_results[coin_id].append([test_image_id, seed_image_id, max_value, angle])

    bad_angle_grand_total = 0
    bad_seed_grand_total = 0
    bad_coin_id_grand_total = 0

    for coin_id, values in coin_results.iteritems():
        bad_angle_total = 0
        bad_seed_total = 0

        seed_image_id_counts = {}
        for test_values in values:
            seed_image_id = test_values[1]
            result = test_values[2]
            if seed_image_id not in seed_image_id_counts.iterkeys():
                seed_image_id_counts[seed_image_id] = 0
            seed_image_id_counts[seed_image_id] += 1

        major_seed_image_id = max(seed_image_id_counts.iteritems(), key=operator.itemgetter(1))[0]

        correct_values = []
        angles = []
        for test_values in values:
            test_image_id = test_values[0]
            seed_image_id = test_values[1]
            test_angle = test_values[3]
            if seed_image_id != major_seed_image_id:
                bad_seed_total += 1
                bad_seed_grand_total +=1
                error_test_image_ids.append(test_image_id)
                continue
            correct_values.append(test_values)
            angles.append(test_angle)

        median_angle = np.median(angles)
        angle_difference_from_median_total = 0
        count = 0
        for test_values in correct_values:
            test_image_id = test_values[0]
            test_angle = test_values[3]
            angle_difference_from_median = abs(median_angle - test_angle)
            if angle_difference_from_median > 359 - angle_tolerance:
                angle_difference_from_median = abs(angle_difference_from_median - 360)
            if angle_difference_from_median > angle_tolerance:
                bad_angle_total += 1
                bad_angle_grand_total += 1
                error_test_image_ids.append(test_image_id)
            else:
                angle_difference_from_median_total += angle_difference_from_median
                count +=1
        good = len(values) - (bad_angle_total + bad_seed_total)
        if count != 0:
            average_angle_difference_from_median = angle_difference_from_median_total / count
            average_coin_angles[coin_id] = get_normal_angle (median_angle + average_angle_difference_from_median)
            print 'Average Angle',  average_coin_angles[coin_id]

        if min_good_images_per_seed > good:
            bad_coin_id_grand_total += 1
            for test_values in values:
                test_image_id = test_values[0]
                error_test_image_ids.append(test_image_id)
        all_error_test_image_ids.extend(error_test_image_ids)


        print coin_id, len(values), 'Good:', good, 'Bad Angle:', bad_angle_total, 'Bad Seed:', bad_seed_total
    print 'bad_angle_grand_total:', bad_angle_grand_total, 'bad_seed_grand_total:', bad_seed_grand_total, 'bad_coin_id_grand_total:' , bad_coin_id_grand_total
    return sorted(list(set(all_error_test_image_ids))), average_coin_angles

def get_normal_angle(angle):
    if angle > 0 and angle < 360:
        return angle
    if angle < 0:
        return angle + 360
    if angle >= 360:
        return angle - 360

def create_test_lmdb_batches(test_image_ids,seed_image_ids,images_per_angle):
    #todo:This needs to create the lmdbs files in order
    start_time = time.time()
    print 'Starting create_test_lmdb_batches'
    #Create test lmdbs by 10 coin_ids  (10 x 57 = 570 images)
    test_batch_filedata = {}

    for test_image_id in test_image_ids:
        test_batch_id = test_image_id / 1000
        if test_batch_id not in test_batch_filedata.iterkeys():
            test_batch_filedata[test_batch_id] = []
            lmdb_dir = test_dir + str(test_batch_id) + '/'
            if not os.path.exists(lmdb_dir):
                os.makedirs(lmdb_dir)
        test_batch_filedata[test_batch_id].append([test_image_id, get_filename_from(test_image_id), 0])

    shell_filenames = []
    calling_args = []
    for test_batch_id,filedata in test_batch_filedata.iteritems():
        lmdb_dir = test_dir + str(test_batch_id) + '/'
        calling_args.append([filedata, lmdb_dir, images_per_angle])
        #create_lmdb_rotate_whole_image.create_lmdbs(filedata, lmdb_dir, images_per_angle, False, False)
        for image_id in seed_image_ids:
            shell_filenames.append(create_test_script(image_id,test_batch_id))

    create_script_calling_script(test_dir + 'test_all.sh', shell_filenames)

    pool = Pool(8)
    pool.map(create_lmdb_rotate_whole_image.create_all_lmdbs, calling_args)
    pool.close()
    pool.join()
    print 'create_test_lmdb_batches', 'Done after %s seconds' % (time.time() - start_time,)


# Multi-Point Works awesome ************************************************************************************
init_dir()
start_time = time.time()
test_image_ids = []
new_test_image_ids = []
new_seed_image_ids = []
count = 0
seed_image_ids = pickle.load(open(data_dir + 'seed_image_ids_all.pickle', "rb"))
for coin_id in seed_image_ids:
    if (count < 99999999) and (coin_id % 2 == 0):
        new_seed_image_ids.append(coin_id * 100)
        new_seed_image_ids.append((coin_id +3) * 100)
        for image_id in range(0,57):
            new_test_image_ids.append(coin_id * 100 + image_id)
            new_test_image_ids.append((coin_id +3) * 100 + image_id)
        count += 2
test_image_ids = sorted(new_test_image_ids)

#seed_image_ids = sorted(new_seed_image_ids)
#pickle.dump(seed_image_ids, open(data_dir + 'seed_image_ids.pickle', "wb"))
pickle.dump(test_image_ids, open(data_dir + 'test_image_ids.pickle', "wb"))
#save_multi_point_ids()

seed_image_data = pickle.load(open(data_dir + 'multi_point_ids.pickle', "rb"))
seed_image_ids = (200,1100)
scripts_to_run = []
images_per_angle = 200
#seed_image_ids = pickle.load(open(data_dir + 'seed_image_ids.pickle', "rb"))
#test_image_ids = pickle.load(open(data_dir + 'seed_image_ids.pickle', "rb"))

#create_test_lmdb_batches(test_image_ids,seed_image_ids,1)
test_batch_ids = []
for test_image_id in test_image_ids:
    test_batch_id = test_image_id / 1000
    if test_batch_id not in test_batch_ids:
        test_batch_ids.append(test_batch_id)

for seed_image_id in seed_image_ids:
    pass
    # filedata = []
    # seed_images = seed_image_data[int(seed_image_id / 100)]
    # for image_id in seed_images:
    #     test_image_id = seed_image_id + image_id
    #     filename = get_filename_from(test_image_id)
    #     filedata.append([test_image_id, filename, 0])
    #create_single_lmdb(seed_image_id, filedata, 0, True, images_per_angle)
    #run_script(train_dir + str(seed_image_id) + '/train-single-coin-lmdbs.sh')
#     for test_batch_id in test_batch_ids:
#         filename = test_dir + str(test_batch_id) + '/' + str(seed_image_id) + '.dat'
#         if os.path.isfile(filename):
#             file_size = os.path.getsize(filename)
#             if file_size > 0:
#                 print 'Exists:', filename
#                 continue
#         create_test_script(seed_image_id,test_batch_id,True)
#         scripts_to_run.append(test_dir + str(test_batch_id) + '/test-' + str(seed_image_id) + '.sh')
#
# run_scripts(scripts_to_run,max_workers=6)
read_test(test_batch_ids,seed_image_ids)

# ********
# Step 2:
# Then widen the seed to include all crops in all results for each seed:
# Check out the results in the png
# Note the cutoff
# This should be changed to include the step 3 double check

# for seed_image_id in widen_seed_image_ids:
#     cutoff = 13
#     filedata = get_single_lmdb_multi_point_filedata(seed_image_id, cutoff, multi_point_error_test_image_ids)
#     create_single_lmdb(seed_image_id, filedata, 0, True, 2800, retraining=True)
#     run_script(train_dir + str(seed_image_id) + '/train-single-coin-lmdbs.sh')
#     run_script(test_dir + str(0) + '/test-' + str(seed_image_id) + '.sh')
# read_test(seed_image_ids, 360)



# # image_set.read_results(0, data_dir, seeds_share_test_images=False, bad_coin_ids=bad_coin_ids, ground_truth=ground_truth)
# image_set.read_results(0, data_dir, seeds_share_test_images=False)
# multi_point_error_test_image_ids = get_multi_point_error_test_image_ids()
# print 'The following test_image_ids where taking out of the image:'
# print multi_point_error_test_image_ids
# print 'multi_point_error_test_image_ids length:' + str(len(multi_point_error_test_image_ids))
# image_set.create_composite_images(crop_dir, data_dir, 125, 40, 10, None, multi_point_error_test_image_ids, True)
# #image_set.create_composite_images(crop_dir, data_dir, 125, 40, 10)
#Dates  ************************************************************************************
image_set.read_results(test_batch_ids, data_dir, seeds_share_test_images=False)
multi_point_error_test_image_ids, coin_angles = get_errors_and_angles()
# Create a composite image for dates:
image_set.create_date_composite_image(crop_dir, data_dir, 1100, 2000, multi_point_error_test_image_ids,coin_angles)
