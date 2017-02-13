import cPickle as pickle
import glob
import math
import os
import shutil
from collections import namedtuple

import cv2
import networkx as nx
import pandas as pd

import caffe_image as ci

results_dict = {}
Image = namedtuple('Image', 'seed_image_id image_id angle max_value')
Group = namedtuple('Group', 'group_id starting_seed_id images')
seed_groups = []
#widened_seeds = [3893, 5107, 6280, 9813, 4152,8924]
#remove_results_of_these_seeds = [8058,7855]
# widened_seeds = [178, 137]
widened_seeds = []
remove_results_of_these_seeds = []


def read_results(cut_off, data_dir, seed_image_ids=None, seeds_share_test_images=True, remove_widened_seeds=False):
    all_results = pickle.load(open(data_dir + 'all_results.pickle', "rb"))
    # columns = ['seed_image_id', 'image_id', 'angle', 'max_value']
    image_ids_with_highest_max_value = {}
    results_dict.clear()
    seeds_to_remove = []

    if remove_widened_seeds:
        seeds_to_remove = widened_seeds
        for seed_id in remove_results_of_these_seeds:
            seeds_to_remove += pickle.load(open(data_dir + str(seed_id) + '.pickle', "rb"))

    # This fills image_ids_with_highest_max_value:
    grand_total_max_value = 0
    grand_total_count = 0
    average_max_values = {}
    for results in all_results:
        total_max_value = 0
        count = 0
        for seed_image_id, image_id, angle, max_value in results:
            total_max_value += max_value
            count += 1
        average_max_value = total_max_value / count
        print seed_image_id, count, total_max_value, average_max_value
        average_max_values[seed_image_id] = average_max_value
        grand_total_max_value += total_max_value
        grand_total_count += count
    overall_average_max_value = grand_total_max_value / grand_total_count
    print 'overall_average_max_value', overall_average_max_value

    for results in all_results:
        for seed_image_id, image_id, angle, max_value in results:
            # This is cheating. Some models are more confident than others and they score better if they are balanced.
            if seed_image_id not in (0, 100):
                adjustment = overall_average_max_value / average_max_values[seed_image_id]
                max_value = max_value * adjustment * .7

            # if seed_image_id in (14300, 23500, 22000, 15800, 23700):
            #     max_value = max_value * .6
            #
            if seed_image_id in (18200, 16300, 26300, 19300, 14300):
                max_value = max_value * 1.4


            # Well, we know this was a match already:
            if seed_image_id == image_id:
                continue
            # This optionally filters the results smaller:
            if seed_image_ids is not None:
                if seed_image_id not in seed_image_ids:
                    continue
            # This optionally filters only the best results:
            if max_value < cut_off:
                continue

            # This optionally filters out widened seeds
            if remove_widened_seeds:
                if seed_image_id in seeds_to_remove:
                    continue

            if image_id in image_ids_with_highest_max_value:
                if image_ids_with_highest_max_value[image_id][2] < max_value:
                    image_ids_with_highest_max_value[image_id] = [seed_image_id, angle, max_value]
            else:
                image_ids_with_highest_max_value[image_id] = [seed_image_id, angle, max_value]

            if not seed_image_id in results_dict:
                results_dict[seed_image_id] = {}

            if not image_id in results_dict[seed_image_id]:
                results_dict[seed_image_id][image_id] = [max_value, angle]

            if max_value > results_dict[seed_image_id][image_id][0]:
                results_dict[seed_image_id][image_id] = [max_value, angle]


    if not seeds_share_test_images:
        results_dict.clear()
        for image_id, values in image_ids_with_highest_max_value.iteritems():
            seed_image_id, angle, max_value = values
            if not seed_image_id in results_dict:
                results_dict[seed_image_id] = {}

            if not image_id in results_dict[seed_image_id]:
                results_dict[seed_image_id][image_id] = [max_value, angle]

            if max_value > results_dict[seed_image_id][image_id][0]:
                results_dict[seed_image_id][image_id] = [max_value, angle]

    pickle.dump(results_dict, open(data_dir + 'seed_data.pickle', "wb"))


def get_results_list(seed_id_filter=-1):
    results_list = []
    for seed_image_id, seed_values in results_dict.iteritems():
        if seed_id_filter == -1:
            pass
        else:
            if seed_image_id != seed_id_filter:
                continue
        for image_id, values in seed_values.iteritems():
            max_value, angle = values
            Image = namedtuple('Image', 'seed_image_id image_id angle max_value')
            results_list.append(Image(seed_image_id, image_id, angle, max_value))
    return results_list


def set_angles_postive():
    Image = namedtuple('Image', 'seed_image_id image_id angle max_value')
    results_list = get_results_list()
    new_results_list = []

    for image in results_list:
        # Flip the seeds and test image_ids on all 331-359 angles so all angles are 0-29.
        if image.angle < 30:
            new_results_list.append(image)
        else:
            seed_image_id = image.image_id
            image_id = image.seed_image_id
            angle = 360 - image.angle
            max_value = image.max_value
            new_results_list.append(Image(seed_image_id, image_id, angle, max_value))

    results_list = new_results_list
    for image in results_list:
        if not image.seed_image_id in results_dict:
            results_dict[image.seed_image_id] = {}

        if not image.image_id in results_dict[image.seed_image_id]:
            results_dict[image.seed_image_id][image.image_id] = [image.max_value, image.angle]
        else:
            existing_max_value = results_dict[image.seed_image_id][image.image_id][0]
            existing_angle = results_dict[image.seed_image_id][image.image_id][1]
            if abs(image.angle - existing_angle) > 2:
                print 'Angles off by more than 3: ', image, existing_angle

            if image.max_value > existing_max_value:
                results_dict[image.seed_image_id][image.image_id] = [image.max_value, image.angle]


def create_group():
    # Set the first top-link for the starting seed will be the test image with the most points.
    # Keep going until I get back to the starting seed.
    pass


def set_starting_seed():
    results_left = results_dict.copy()
    df = pd.DataFrame(get_results_list())
    grouped = df.groupby(by=['seed_image_id'])['max_value'].sum()
    starting_seed_id = grouped.idxmax()

    links = pd.DataFrame(get_results_list(starting_seed_id))
    grouped = df.groupby(by=['seed_image_id'])['max_value'].sum()
    test_id = grouped.idxmax()
    del results_left[starting_seed_id][test_id]

    # The seed left with the most points(sum of max_value for all links) is the starting seed.
    # Starting seed is 0 and 360.
    pass


def get_nodes():
    nodes = set(results_dict.keys())
    for seed_image_id, seed_values in results_dict.iteritems():
        for image_id, values in seed_values.iteritems():
            if not image_id in nodes:
                nodes.add(image_id)
    return nodes


def get_edges():
    # this is currently ignoring dup edges.
    edges = {}
    for seed_image_id, seed_values in results_dict.iteritems():
        for image_id, values in seed_values.iteritems():
            max_value = values[0]
            angle = values[1]
            node1 = seed_image_id
            node2 = image_id
            edge_value = [max_value, angle]

            # flip the node order so the first node image_id is always lower
            if node1 < node2:
                temp = node1
                node1 = node2
                node2 = temp
                edge_value = [max_value, -angle]
            edge_key = (node1, node2)
            if not edge_key in edges.keys():
                edges[edge_key] = edge_value
    return edges


def find_most_connected_seeds(data_dir, seed_image_id, min_connections, max_depth):
    most_connected_seeds = {}
    nodes = pickle.load(open(data_dir + 'nodes.pickle', "rb"))
    edges = pickle.load(open(data_dir + 'edges.pickle', "rb"))
    test_images = results_dict[seed_image_id]
    G = nx.Graph()
    G.add_nodes_from(nodes)
    G.add_edges_from(edges)
    return get_most_connected_seeds(G, edges, seed_image_id, most_connected_seeds, 0, 0, min_connections, max_depth)


# Warning: This function is recursive as it follows the graph:
def get_most_connected_seeds(G, edges, start_node, most_connected_seeds, total_path_angle, level, min_connections,
                             max_depth):
    if not start_node in results_dict.iterkeys():
        return most_connected_seeds

    paths = []
    for end_node in results_dict[start_node]:
        if end_node not in most_connected_seeds.iterkeys():
            paths.append(list(nx.all_simple_paths(G, start_node, end_node, 2)))

    bad_paths = []
    graph_results = []

    for edge_paths in paths:
        max_value_ave = 0
        test_image_id = 0
        test_max_value = 0
        test_image_angle = 0
        angles = {}
        max_value_path_total = 0

        for path in edge_paths:
            node1 = -1
            node2 = -1
            angle_total = 0
            max_value_edge_path_total = 0
            max_value = 0
            for node in path:
                if node1 == -1:
                    node1 = node
                    continue
                node2 = node
                key = (node1, node2)
                if key in edges:
                    max_value, angle = edges[(node1, node2)]
                else:
                    max_value, angle = edges[(node2, node1)]
                    angle = -angle
                # print node1, node2, max_value, angle
                angle_total += angle
                max_value_edge_path_total += max_value
                node1 = node

            max_value_path_total += max_value_edge_path_total / len(path)
            angle_total = ci.get_formated_angle(angle_total)
            if len(path) == 2:
                test_image_id = node2
                test_max_value = max_value
                test_image_angle = angle_total
                #print '                       ', path, angle_total, '\n'
            else:
                angles[tuple(path)] = angle_total
                #print '    ', path, angle_total, '\n'
        good_paths_count = 0
        for saved_path, angle in angles.iteritems():
            if abs(test_image_angle - angle) < 3:
                good_paths_count += 1
            else:
                print saved_path, angle, test_image_angle
                bad_paths.append(saved_path)

        max_value_ave = max_value_path_total / len(edge_paths)
        graph_results.append(
            [test_image_id, test_image_angle, total_path_angle + test_image_angle, test_max_value, max_value_ave,
             len(edge_paths) - 1, good_paths_count])

    graph_results = sorted(graph_results, key=lambda graph_results: graph_results[3], reverse=True)

    for result in graph_results:
        seed_image_id = result[0]
        if seed_image_id not in widened_seeds:
            if seed_image_id not in most_connected_seeds:
                if seed_image_id != start_node:
                    new_total_path_angle = total_path_angle + result[1]
                    # You want the path mostly growing away from the first seed.
                    # So this puts a natural limit on depth
                    if (abs(new_total_path_angle) - abs(total_path_angle)) >= 0:
                        if abs(total_path_angle + result[1]) <= 180:  # Only follow nodes to +180 or -180
                            if result[6] > min_connections:
                                most_connected_seeds[seed_image_id] = result
                                if level < max_depth:
                                    most_connected_seeds = get_most_connected_seeds(G, edges, seed_image_id,
                                                                                    most_connected_seeds,
                                                                                    total_path_angle + result[1],
                                                                                    level + 1, min_connections,
                                                                                    max_depth)
                                    print level

    return most_connected_seeds


def clean_dir(dir):
    if os.path.exists(dir):
        shutil.rmtree(dir)

    if not os.path.exists(dir):
        os.makedirs(dir)


def create_composite_images(crop_dir, html_dir, crop_size, rows, cols, seed_image_ids=None, remove_image_ids=None):
    clean_dir(html_dir)
    if seed_image_ids is None:
        results = results_dict
    else:
        results = {seed_image_id: results_dict[seed_image_id] for seed_image_id in seed_image_ids}

    if remove_image_ids is None:
        remove_image_ids = []

    for seed_image_id, seed_values in results.iteritems():
        images = []
        rotated_crop = ci.get_rotated_crop(crop_dir, seed_image_id, crop_size, 0)
        if rotated_crop is None:
            continue
        images.append(rotated_crop)

        results = []
        for image_id, values in seed_values.iteritems():
            max_value, angle = values
            if image_id not in remove_image_ids:
                results.append([image_id, max_value, angle])

        sorted_results = sorted(results, key=lambda result: result[1], reverse=True)
        for image_id, max_value, angle in sorted_results:
            crop = ci.get_rotated_crop(crop_dir, image_id, crop_size, angle)
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(crop, str(max_value)[0:5], (10, 20), font, .7, (0, 255, 0), 2)
            cv2.putText(crop, str(image_id)[0:5], (10, 90), font, .7, (0, 255, 0), 2)
            images.append(crop)
        calc_rows = int(len(images) / cols) + 1
        composite_image = ci.get_composite_image(images, calc_rows, cols)
        cv2.imwrite(html_dir + str(seed_image_id).zfill(5) + '.png', composite_image)


def create_date_composite_image(crop_dir, html_dir, seed_image_id, max_images, remove_image_ids):
    image_size = 448
    crop_radius = 56
    heads_date_angle = 158
    date_center_offset = 166
    # This is the angle seed 100 was imaged at:
    seed_id_100_angle = 148

    # closer up for translation:
    # crop_radius = 28
    # heads_date_angle = 158
    # date_center_offset = 166
    # This is the angle seed 100 was imaged at:
    # seed_id_100_angle = 148

    seed_values = results_dict[seed_image_id]

    images = []
    results = []

    for image_id, values in seed_values.iteritems():
        max_value, angle = values
        if image_id not in remove_image_ids:
            results.append([image_id, max_value, angle])
    sorted_results = sorted(results, key=lambda result: result[0], reverse=False)

    count = 0
    for image_id, max_value, angle in sorted_results:
        if count >= max_images:
            continue
        crop = cv2.imread(crop_dir + str(image_id) + '.png')
        if crop == None:
            pass

        center_x = image_size / 2 + math.cos(math.radians(heads_date_angle + angle)) * date_center_offset
        center_y = image_size / 2 - math.sin(math.radians(heads_date_angle + angle)) * date_center_offset
        print image_id, max_value, angle, center_x, center_y
        date_crop = crop[center_x - crop_radius:center_x + crop_radius, center_y - crop_radius:center_y + crop_radius]
        rotated_date_crop = ci.rotate(date_crop, angle - seed_id_100_angle, crop_radius, crop_radius, crop_radius * 2,
                                      crop_radius * 2)
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(rotated_date_crop, str(image_id)[0:5], (10, 90), font, .7, (0, 255, 0), 2)
        cv2.circle(rotated_date_crop, (crop_radius, crop_radius), 2, (0, 0, 255), 1)
        crop_rows, crop_cols, channels = rotated_date_crop.shape
        if crop_rows != crop_cols:
            print image_id, crop_rows, crop_cols
            rotated_date_crop = cv2.resize(rotated_date_crop, (crop_radius * 2, crop_radius * 2),
                                           interpolation=cv2.INTER_AREA)

        #cv2.imwrite('/home/pkrush/cent-dates/' + str(image_id) + '.png',rotated_date_crop)
        images.append(rotated_date_crop)
        count += 1

    calc_rows = int(len(images) / 10) + 1
    composite_image = ci.get_composite_image(images, calc_rows, 10)
    cv2.imwrite(html_dir + 'dates.png', composite_image)
    print count, 'Date images written'

def create_composite_image(crop_dir, html_dir, crop_size, rows, cols, seed_image_ids):
    images = []
    for seed_image_id in seed_image_ids:
        crop = ci.get_rotated_crop(crop_dir, seed_image_id, crop_size, 0)
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(crop, str(seed_image_id)[0:5], (10, 90), font, .7, (0, 255, 0), 2)
        cv2.circle(crop, (crop_size / 2, crop_size / 2), 2, (0, 0, 255), 1)
        images.append(crop)
    composite_image = ci.get_composite_image(images, rows, cols)
    cv2.imwrite(html_dir + 'composite_image.png', composite_image)


def create_composite_image_from_filedata(crop_dir, html_dir, crop_size, rows, cols, filedata):
    images = []
    for image_id, filename, angle_offset in filedata:
        crop = ci.get_rotated_crop(crop_dir, image_id, crop_size, angle_offset)
        images.append(crop)
    composite_image = ci.get_composite_image(images, rows, cols)
    cv2.imwrite(html_dir + 'composite_image.png', composite_image)


def save_gray(src_dir, dst_dir, size):
    for full_filename in glob.iglob(src_dir + '*.png'):
        image = cv2.imread(full_filename)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        small = cv2.resize(gray, (size, size), interpolation=cv2.INTER_AREA)
        new_filename = full_filename.replace(src_dir, dst_dir)
        cv2.imwrite(new_filename, small)

def save_widened_seeds(data_dir, seed_image_id,cut_off):
    widened_seeds = []
    values = results_dict[seed_image_id].iteritems()
    for test_image_id, test_values in values:
        max_value, angle = test_values
        if max_value > cut_off:
            widened_seeds.append(test_image_id)
    print 'test_images_saved: ' , len(widened_seeds)

    pickle.dump(widened_seeds, open(data_dir + str(seed_image_id) + '.pickle', "wb"))

