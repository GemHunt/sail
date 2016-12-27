"""
A set of functions to test out self supervised rotated coin image models
"""

import cPickle as pickle
import glob
import os
import random
import shutil
import subprocess
import time

import create_lmdb_rotate_whole_image
import image_set
import summarize_rotated_crops

home_dir = '/home/pkrush/lmdb-files/'
data_dir = home_dir + 'metadata/'
crop_dir = home_dir + 'crops/'
train_dir = home_dir + 'train/'
test_dir = home_dir + 'test/'
test_angles = {0: (30, 330), 1: (60, 300), 2: (90, 270), 3: (120, 240), 4: (150, 210), 5: (180, 180)}
wide_image_ids = {11458, 12004}


def init_dir():
    directories = [home_dir, data_dir, crop_dir, train_dir, test_dir]
    for test_id in range(0, 6):
        directories.append(test_dir + str(test_id) + '/')
    make_dir(directories)


def make_dir(directories):
    for path_name in directories:
        if not os.path.exists(path_name):
            os.makedirs(path_name)


def create_new_seed_index():
    seed_image_ids = [random.randint(4000, 13828) for x in range(250)]
    pickle.dump(seed_image_ids, open(data_dir + 'seed_image_ids.pickle', "wb"))



def get_seed_image_ids():
    return get_test_image_ids()

    # seed_image_ids = pickle.load(open(data_dir + 'seed_image_ids.pickle', "rb"))
    # return sorted(set(seed_image_ids))

    # test_image_ids = pickle.load(open(data_dir + 'test_image_ids.pickle', "rb"))
    # seed_image_ids = seed_image_ids + test_image_ids[0:180]
    # seed_image_ids = seed_image_ids + wide_image_ids()
    # pickle.dump(seed_image_ids, open(data_dir + 'seed_image_ids.pickle', "wb"))


def get_test_image_ids():
    test_image_ids = pickle.load(open(data_dir + 'test_image_ids.pickle', "rb"))
    return sorted(set(test_image_ids))

    # test_image_ids += get_seed_image_ids()
    # test_image_ids += wide_image_ids()
    # test_image_ids = list(set(test_image_ids))
    # pickle.dump(test_image_ids, open(data_dir + 'test_image_ids.pickle', "wb"))


def rename_crops():
    crops = []
    for filename in glob.iglob(crop_dir + '*.jpg'):
        crops.append([random.random(), filename])
    crops.sort()
    pickle.dump(crops, open(data_dir + 'copper_crops.p', "wb"))
    key = 0
    for rand, filename in crops:
        key += 1
        os.rename(filename, crop_dir + str(key) + '.jpg')


def copy_file(filename, path_name):
    with open(filename, 'r') as myfile:
        data = myfile.read().replace('replace_dir_name_', path_name)
    with open(path_name + filename, 'w') as file_:
        file_.write(data)


def create_single_lmdbs(seed_image_ids):
    weight_filename = 'starting-weights.caffemodel'
    shutil.copyfile(weight_filename, train_dir + weight_filename)
    shell_filenames = []
    for image_id in seed_image_ids:
        filedata = [[image_id, crop_dir + str(image_id) + '.jpg', 0]]
        lmdb_dir = train_dir + str(image_id) + '/'
        create_lmdb_rotate_whole_image.create_lmdbs(filedata, lmdb_dir, 100, -1, True, False)
        print 'Creating single lmdb for ' + str(image_id)
        copy_train_files(lmdb_dir)
        shell_filename = create_train_script(lmdb_dir, train_dir + weight_filename)
        shell_filenames.append(shell_filename)
    create_script_calling_script(train_dir + 'train_all.sh', shell_filenames)


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

    filedata.append([seed_id, crop_dir + str(seed_id) + '.jpg', 0])
    for image_id, test_values in values.iteritems():
        max_value, angle = test_values
        if max_value > max_value_cutoff:
            filedata.append([image_id, crop_dir + str(image_id) + '.jpg', angle])
    return filedata

def create_single_lmdb(seed_image_id, filedata, test_id=0,multi_image_training = False ):
    start_time = time.time()
    print 'create_single_lmdb for ' + str(seed_image_id)

    weight_filename = train_dir + str(seed_image_id) + '/' + 'starting-weights.caffemodel'
    weight_filename_copy = train_dir + 'starting-weights.caffemodel'
    shutil.copyfile(weight_filename_copy, weight_filename)

    lmdb_dir = train_dir + str(seed_image_id) + '/'

    #create_lmdb_rotate_whole_image.create_lmdbs(filedata, lmdb_dir, int(100 + (0.5 * test_id)), -1, True, False)
    create_lmdb_rotate_whole_image.create_lmdbs(filedata, lmdb_dir, int(200 + (2 * test_id)), -1, True, False)
    copy_train_files(lmdb_dir,multi_image_training)
    create_train_script(lmdb_dir, weight_filename_copy,multi_image_training)
    print 'Done in %s seconds' % (time.time() - start_time,)


def create_test_lmdbs(test_id):
    # test_image_ids = [x for x in range(13927)]
    test_image_ids = get_test_image_ids()
    filedata = []
    lmdb_dir = test_dir + str(test_id) + '/'
    for image_id in test_image_ids:
        filedata.append([image_id, crop_dir + str(image_id) + '.jpg', 0])

    create_lmdb_rotate_whole_image.create_lmdbs(filedata, lmdb_dir, 3, test_id, False, False)

    shell_filenames = []
    seed_image_ids = get_seed_image_ids()

    for image_id in seed_image_ids:
        shell_filenames.append( create_test_script(image_id,test_id))
    create_script_calling_script(test_dir + 'test_all.sh', shell_filenames)

def create_test_script(image_id,test_id,multi_image_training = False ):
    shell_script = 'cd ' + train_dir + str(image_id) + '/\n'
    shell_script += '/home/pkrush/caffe/.build_release/examples/cpp_classification/classification.bin '
    shell_script += 'deploy.prototxt '
    if multi_image_training:
        shell_script += 'snapshot_iter_8440.caffemodel '
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


def read_test(image_ids, max_test_id):
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

    low_angle, high_angle = test_angles[max_test_id]
    for test_id in range(0, max_test_id + 1):
        for image_id in image_ids:
            filename = test_dir + str(test_id) + '/' + str(image_id) + '.dat'

            if not os.path.isfile(filename):
                continue
            results = summarize_rotated_crops.get_results(filename, image_id, low_angle, high_angle)
            # results = summarize_whole_rotated_model_results.summarize_whole_rotated_model_results(
            #                                                                  filename, image_id,low_angle,high_angle)
            new_all_results.append(results)
    pickle.dump(new_all_results, open(data_dir + 'all_results.pickle', "wb"))


def widen_model(seed_image_id, max_test_id, max_value_cutoff):
    for test_id in range(0, max_test_id + 1):
        for x in range(0, 2):
            filedata = get_single_lmdb_filedata(seed_image_id, max_value_cutoff)
            run_train_test(seed_image_id, filedata, max_value_cutoff, test_id)
    image_set.create_composite_images(crop_dir, data_dir, 120, 8, 20)


def run_train_test(seed_image_id, filedata,max_value_cutoff, test_id, multi_image_training = False):
    create_single_lmdb(seed_image_id, filedata, test_id, multi_image_training)
    for test_id in range(0, test_id + 1):
        create_test_script(seed_image_id, test_id, multi_image_training)
    run_script(train_dir + str(seed_image_id) + '/train-single-coin-lmdbs.sh')
    run_script(test_dir + str(test_id) + '/test-' + str(seed_image_id) + '.sh')
    read_test([seed_image_id], test_id)
    # in the metadata dir rm *.png
    image_set.read_results(max_value_cutoff, data_dir, [seed_image_id])


def run_test(seed_image_id, max_value_cutoff, test_id):
    for test_id in range(0, test_id + 1):
        run_script(test_dir + str(test_id) + '/test-' + str(seed_image_id) + '.sh')
    read_test([seed_image_id], test_id)
    image_set.read_results(max_value_cutoff, data_dir, [seed_image_id])


def create_all_test_lmdbs():
    for test_id in range(1, 6):
        create_test_lmdbs(test_id)


def test_all(seed_image_ids):
    for seed_image_id in seed_image_ids:
        for test_id in range(0, 6):
            run_script(test_dir + str(test_id) + '/test-' + str(seed_image_id) + '.sh')

    read_test(seed_image_ids, 5)
    read_all_results(0, seed_image_ids)


def run_script(filename):
    print "Running " + filename
    subprocess.call(filename)


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
    #image_set.create_composite_images(crop_dir, data_dir, 140,10,10)


def link_seed_by_graph(seed_image_id, min_connections, max_depth):
    filedata = []
    read_all_results(10, seeds_share_test_images=True, remove_widened_seeds=True)
    save_graph()
    # read_all_results(5,[4866],seeds_share_test_images=False,remove_widened_seeds=True)
    most_connected_seeds = image_set.find_most_connected_seeds(data_dir, seed_image_id, min_connections, max_depth)
    if len(most_connected_seeds) != 0:
        image_set.create_composite_image(crop_dir, data_dir, 130, 10, 30, most_connected_seeds.iterkeys())
        for seed_image_id, values in most_connected_seeds.iteritems():
            print values
            filedata.append([seed_image_id, crop_dir + str(seed_image_id) + '.jpg', values[2]])
    print 'Count of images linked by graph:', len(most_connected_seeds)
    return filedata


def retrain_widened_seed(seed_image_id, cut_off):
    max_value_cutoff = 10
    read_all_results(max_value_cutoff, seeds_share_test_images=False, remove_widened_seeds=True)
    filedata = get_single_lmdb_filedata(seed_image_id, cut_off)
    run_train_test(seed_image_id, filedata, max_value_cutoff, test_id=5,multi_image_training =  True)
    read_all_results(max_value_cutoff, seeds_share_test_images=False, remove_widened_seeds=True)
    image_set.create_composite_images(crop_dir, data_dir,crop_size = 140,rows=10,cols=50)


retrain_widened_seed(7855,27)


#read_all_results(10, seeds_share_test_images=False, remove_widened_seeds=True)
#image_set.create_composite_images(crop_dir, data_dir, 120,10,50)
#image_set.save_widened_seeds(data_dir, 8058,32)
#image_set.save_widened_seeds(data_dir, 7855,19)


'''
# Good ones to link: 8058,7855
seed_image_id = 8058
filedata = link_seed_by_graph(seed_image_id,min_connections=10, max_depth=18)
image_set.create_composite_image_from_filedata(crop_dir, data_dir, 140, rows=10, cols=50, filedata=filedata)

if 1==0:
    if len(filedata) > 5:
        max_value_cutoff = 10
        run_train_test(seed_image_id, filedata, max_value_cutoff, test_id=5,multi_image_training =  True)
        run_test(seed_image_id, max_value_cutoff, test_id=5)
        read_all_results(15)
        image_set.create_composite_images(crop_dir, data_dir,crop_size = 140,rows=10,cols=50)
    else:
        print 'Not enough seeds found'
'''



# Instructions from scratch:
# create_new_seed_index()
# seeds = get_seed_image_ids()- wide_image_ids()
# create_single_lmdbs(seeds)
# create_test_lmdbs(0)
# run_script(train_dir + 'train_all.sh')
# run_script(test_dir + 'test_all.sh')

# read_test(get_seed_image_ids(),0)
# read_all_results(10)


# Second Try Script:
# I renamed lmdb-files to lmdbfiles100
# I also copied the crops and 2 pickles for seeds and test IDs
# I cropped 56x56 from center, dropped using the mask, and dropped the resize.
# init_dir()
# seeds = get_seed_image_ids()
# create_single_lmdbs(seeds)
# create_test_lmdbs(0)
# run_script(train_dir + 'train_all.sh')
# run_script(test_dir + 'test_all.sh')
# read_test(get_seed_image_ids(),0)

# read_all_results(16)

# Pick top seed with the most image results over 20 and highest of those results:
# widen_model(9813,5,23)
# create_all_test_lmdbs()
# Shrink the results to the widened seeds:
# read_all_results(0,[11458,12004])
# create_all_test_lmdbs()  #Raise the number of test images
# test_all(seed_image_ids)
# Check out the test set results and choose the number of seeds(60) and training images(1000).
# Test on new test set and make 30 new seeds low performers of each set.
# Create test sets from the 500 lowest performers of each set.
# create_new_indexes(30, 500)
