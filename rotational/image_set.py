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


def read_results(cut_off, data_dir, seed_image_ids=None, seeds_share_test_images=True, remove_widened_seeds=False,
                 bad_coin_ids=None, ground_truth=None, remove_coin_ids=None):
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
        if count == 0:
            continue
        average_max_value = total_max_value / count
        print seed_image_id, count, total_max_value, average_max_value
        average_max_values[seed_image_id] = average_max_value
        grand_total_max_value += total_max_value
        grand_total_count += count
    overall_average_max_value = grand_total_max_value / grand_total_count
    print 'overall_average_max_value', overall_average_max_value

    for results in all_results:
        for seed_image_id, image_id, angle, max_value in results:
            seed_coin_id = seed_image_id / 100
            image_coin_id = image_id / 100
            if ground_truth is not None:
                if image_coin_id in ground_truth.iterkeys():
                    if ground_truth[image_coin_id] != seed_coin_id:
                        continue
            if bad_coin_ids is not None:
                if [seed_coin_id, image_coin_id] in bad_coin_ids:
                    continue

            if remove_coin_ids is not None:
                if image_coin_id in remove_coin_ids:
                    continue

                            # Model Balancing:
            # Some models are more confident than others
            # So balancing models will help them score better.

            # This is a generic adjustment:
            # if seed_image_id in (0,100): #40% of class
            # adjustment = overall_average_max_value / average_max_values[seed_image_id]
            # max_value = max_value * adjustment * .7

            # This is cheating by know how the models will perform and what the class sizes are.
                    # if seed_image_id in ( 21600,22300,25100):
                    #     max_value = max_value * .85
                    #
                    # if seed_image_id in (100,999):
                    #     max_value = max_value * .68
                    #
                    # if seed_image_id in (26200, 29300):
                    #     max_value = max_value * 1.05
                    #
                    # if seed_image_id in (0, 999):
                    #     max_value = max_value * 1.35
                    #
                    # if seed_image_id in (22400, 999):
                    #     max_value = max_value * 1.0
                    #
                    # if seed_image_id in (20000, 26200, 15700,16300):
                    #     max_value = max_value * 1.4

                    # #
            # if seed_image_id in (20000,19500,15700,9999):
            #     max_value = max_value * 1.4
            #
            # if seed_image_id in (22400,9999):
            #     max_value = max_value * 2
            #
            #
            # if seed_image_id in (14300,999):
            #      max_value = max_value * .7


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

def remove_angles_for_dates_in_all_results(data_dir):
    all_results = pickle.load(open(data_dir + 'all_results.pickle', "rb"))
    # columns = ['seed_image_id', 'image_id', 'angle', 'max_value']

    new_all_results = []
    for results in all_results:
        new_results = []
        for seed_image_id, image_id, angle, max_value in results:
            if not (15 < angle < 345):
                new_results.append([seed_image_id, image_id, angle, max_value])
        new_all_results.append(new_results)
    pickle.dump(new_all_results, open(data_dir + 'all_results.pickle', "wb"))


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


def get_image_id(image_id,show_back):
    if image_id < 300:
        return  image_id
    coin_id = image_id / 100
    if show_back:
        if coin_id % 2 == 0:
            return image_id + 300
        else:
            return image_id - 300
    else:
        return image_id

def create_composite_image_ground_truth_designs(crop_dir, data_dir, crop_size, rows, cols, coin_angles,ground_truth_designs,sort,show_marked,show_back,for_dates):
    html_dir = data_dir + 'html/'
    clean_dir(html_dir)
    results = results_dict
    coin_id_seed_id = {}
    images_by_seed = {}

    for coin_id, values in ground_truth_designs.iteritems():
        seed_image_id = values[0]
        result = values[1]
        marked = values[2]
        if not show_marked and marked == 1:
            continue
        image_id = (coin_id * 100) + 54
        if not seed_image_id in images_by_seed.iterkeys():
            images_by_seed[seed_image_id] = []
        angle = 0
        if coin_id in coin_angles.iterkeys():
            angle = coin_angles[coin_id]
        if for_dates and 10 < angle < 350:
            continue
        images_by_seed[seed_image_id].append([image_id, result, angle])


    for seed_image_id, values in images_by_seed.iteritems():
        images = []
        rotated_crop = ci.get_rotated_crop(crop_dir,  get_image_id(seed_image_id,show_back) , crop_size, 0)
        if rotated_crop is None:
            continue
        images.append(rotated_crop)

        sorted_results = sorted(values, key=lambda result: result[1], reverse=sort)
        for image_id, max_value, angle in sorted_results:
            crop = ci.get_rotated_crop(crop_dir, get_image_id(image_id,show_back), crop_size, angle)
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(crop, str(max_value)[0:5], (4, 20), font, .7, (0, 255, 0), 2)
            cv2.putText(crop, str(get_image_id(image_id,show_back)/100)[0:4], (4, 90), font, .7, (0, 255, 0), 2)
            images.append(crop)
        calc_rows = int(len(images) / cols) + 1
        if rows >= calc_rows:
            rows_to_pass = calc_rows
        else:
            rows_to_pass = rows
        composite_image = ci.get_composite_image(images, rows, cols)
        cv2.imwrite(html_dir + str(seed_image_id).zfill(5) + '.png', composite_image)
    return



def create_composite_image_total_result(crop_dir, data_dir, crop_size, rows, cols, coin_angles,total_coin_results):
    html_dir = data_dir + 'html/'
    clean_dir(html_dir)
    results = results_dict
    coin_id_seed_id = {}

    for seed_image_id, seed_values in results.iteritems():
        images = []
        results = []
        coin_results = {}
        seed_coin_results = total_coin_results[seed_image_id]
        for coin_id in seed_coin_results.iterkeys():
            image_id = (coin_id * 100) + 54
            if coin_id in seed_coin_results.iterkeys():
                if coin_id in coin_angles.iterkeys():
                    results.append([image_id, seed_coin_results[coin_id], coin_angles[coin_id]])

        sorted_results = sorted(results, key=lambda result: result[1], reverse=True)
        for image_id, max_value, angle in sorted_results:
            crop = ci.get_rotated_crop(crop_dir, image_id, crop_size, angle)
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(crop, str(max_value)[0:5], (4, 20), font, .7, (0, 255, 0), 2)
            cv2.putText(crop, str(image_id)[0:6], (4, 90), font, .7, (0, 255, 0), 2)
            images.append(crop)
        calc_rows = int(len(images) / cols) + 1
        if rows >= calc_rows:
            rows_to_pass = calc_rows
        else:
            rows_to_pass = rows

        composite_image = ci.get_composite_image(images, rows, cols)
        cv2.imwrite(html_dir + str(seed_image_id).zfill(5) + '.png', composite_image)
    return


def create_composite_images(crop_dir, data_dir, crop_size, rows, cols, seed_image_ids=None, remove_image_ids=None,
                            use_only_best_coin_image=False):
    html_dir = data_dir + 'html/'
    clean_dir(html_dir)
    if seed_image_ids is None:
        results = results_dict
    else:
        results = {seed_image_id: results_dict[seed_image_id] for seed_image_id in seed_image_ids}

    if remove_image_ids is None:
        remove_image_ids = []
    coin_id_seed_id = {}


    for seed_image_id, seed_values in results.iteritems():
        images = []

        #Get the seed image first:
        rotated_crop = ci.get_rotated_crop(crop_dir, seed_image_id, crop_size, 0)
        if rotated_crop is None:
            continue
        images.append(rotated_crop)

        results = []
        coin_results = {}

        if use_only_best_coin_image:
            for image_id, values in seed_values.iteritems():
                max_value, angle = values
                if image_id not in remove_image_ids:
                    coin_id = image_id / 100
                    if coin_id not in coin_results.iterkeys():
                        coin_results[coin_id] = []
                    coin_results[coin_id].append([image_id, max_value, angle])
            for coin_id, values in coin_results.iteritems():
                image_id, max_value, angle = max(values, key=lambda item: item[1])
                # for Dates:
                if 10 > angle or 350 < angle:
                    results.append([image_id, max_value, angle])
                    coin_id_seed_id[coin_id] = [seed_image_id / 100, image_id, max_value, angle]

        else:
            for image_id, values in seed_values.iteritems():
                max_value, angle = values
                if image_id not in remove_image_ids:
                    results.append([image_id, max_value, angle])

        sorted_results = sorted(results, key=lambda result: result[1], reverse=True)
        for image_id, max_value, angle in sorted_results:
            crop = ci.get_rotated_crop(crop_dir, image_id, crop_size, angle)
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(crop, str(max_value)[0:5], (4, 20), font, .7, (0, 255, 0), 2)
            cv2.putText(crop, str(image_id)[0:6], (4, 90), font, .7, (0, 255, 0), 2)
            images.append(crop)
        calc_rows = int(len(images) / cols) + 1
        if rows >= calc_rows:
            rows_to_pass = calc_rows
        else:
            rows_to_pass = rows

        composite_image = ci.get_composite_image(images, rows, cols)
        cv2.imwrite(html_dir + str(seed_image_id).zfill(5) + '.png', composite_image)


    return
    #todo ground_truth should be somewhere else:
    ground_truth_coin_ids = pickle.load(open(data_dir + 'ground_truth_coin_ids.pickle', "rb"))
    misclassify_count = 0
    seeds = []
    seed_totals = {}
    predicted_totals = {}
    for coin_id in range(0, 306):
        if ground_truth_coin_ids[coin_id][0] not in seeds:
            seeds.append(ground_truth_coin_ids[coin_id][0])
            seed_totals[ground_truth_coin_ids[coin_id][0]] = 0
            predicted_totals[ground_truth_coin_ids[coin_id][0]] = 0
        if coin_id not in coin_id_seed_id.iterkeys():
            # 204 is just the bad dump coin_id for now.
            # Wow this is sloppy code! It's hard when you can't plan it out.
            coin_id_seed_id[coin_id] = [204, coin_id * 100, 0, 0]
            print 'Coin Missing', coin_id
            misclassify_count += 1
        else:
            if coin_id_seed_id[coin_id][0] <> ground_truth_coin_ids[coin_id][0]:
                print 'Wrong Seed for ', coin_id
                misclassify_count += 1
    seeds.sort()
    print 'Misclassify Count:', misclassify_count

    confusion_matrix = '\t\t\t'
    for ground_truth_seed_id in seeds:
        confusion_matrix += str(ground_truth_seed_id) + '\t'
    confusion_matrix += 'Actual\n'
    for ground_truth_seed_id in seeds:
        total = 0
        line = 'Actual:\t' + str(ground_truth_seed_id)
        for coin_id in range(0, 306):
            if ground_truth_coin_ids[coin_id][0] == ground_truth_seed_id:
                seed_totals[coin_id_seed_id[coin_id][0]] += 1
                predicted_totals[coin_id_seed_id[coin_id][0]] += 1
                total += 1
        for coin_id in seeds:
            line += '\t' + str(seed_totals[coin_id])
            seed_totals[coin_id] = 0
        confusion_matrix += line + '\t\t' + str(total) + '\n'
    total = 0
    line = 'Pred Total:'
    for coin_id in seeds:
        line += '\t' + str(predicted_totals[coin_id])
        total += predicted_totals[coin_id]
    line += '\t\t' + str(total)
    print confusion_matrix + line


# pickle.dump(coin_id_seed_id, open(data_dir + 'ground_truth_coin_ids.pickle', "wb"))


def create_date_composite_image(crop_dir, data_dir, seed_image_id, max_images,coin_angles,only_show_one_image,save_files):
    #todo:OK how are heads_date_angle and date_center_offset releated?
    #I think this is only one angle.

    html_dir = data_dir + 'html/'
    image_size = 448
    crop_radius = 28
    heads_date_angle = 220
    #date_center_offset = 166
    date_center_offset = 172

    # This is the angle seed 100 was imaged at:
    seed_id_100_angle = 75

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
        coin_id = image_id / 100
        if coin_id in coin_angles:
            results.append([image_id, max_value, angle])
    sorted_results = sorted(results, key=lambda result: result[0], reverse=False)

    count = 0
    for image_id, max_value, angle in sorted_results:
        if only_show_one_image:
            if image_id  % 54 != 0:
                continue
        if count >= max_images:
            continue
        coin_id = image_id / 100
        coin_angle = coin_angles[coin_id]
        #print angle, coin_angle

        filename = ci.get_filename_from(image_id,crop_dir)
        crop = cv2.imread(filename)

        if crop == None:
            print 'None?:', filename
            continue

        center_x = image_size / 2 + math.cos(math.radians(heads_date_angle + coin_angle)) * date_center_offset
        center_y = image_size / 2 - math.sin(math.radians(heads_date_angle + coin_angle)) * date_center_offset
        #print image_id, max_value, coin_angle, center_x, center_y
        date_crop = crop[center_x - crop_radius:center_x + crop_radius, center_y - crop_radius:center_y + crop_radius]
        rotated_date_crop = ci.rotate(date_crop, coin_angle - seed_id_100_angle, crop_radius, crop_radius, crop_radius * 2,
                                      crop_radius * 2)
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(rotated_date_crop, str(image_id)[0:5], (10, 90), font, .7, (0, 255, 0), 2)
        cv2.circle(rotated_date_crop, (crop_radius, crop_radius), 2, (0, 0, 255), 1)
        crop_rows, crop_cols, channels = rotated_date_crop.shape
        if crop_rows != crop_cols:
            print image_id, crop_rows, crop_cols
            rotated_date_crop = cv2.resize(rotated_date_crop, (crop_radius * 2, crop_radius * 2),
                                           interpolation=cv2.INTER_AREA)

        if save_files:
            dir = '/home/pkrush/cent-dates/' + str(coin_id/100)+'/'
            if not os.path.exists(dir):
                os.makedirs(dir)
            cv2.imwrite(dir + str(image_id).zfill(7)  + '.png',rotated_date_crop)
        else:
            images.append(rotated_date_crop)
        count += 1

    if not save_files:
        calc_rows = int(len(images) / 10) + 1
        composite_image = ci.get_composite_image(images, calc_rows, 20)
        cv2.imwrite(html_dir + 'dates.png', composite_image)

    print count, 'Date images written'


def create_composite_image(crop_dir, data_dir, crop_size, rows, cols, seed_image_ids):
    html_dir = data_dir + 'html/'
    images = []
    for seed_image_id in seed_image_ids:
        crop = ci.get_rotated_crop(crop_dir, seed_image_id, crop_size, 0)
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(crop, str(seed_image_id)[0:5], (10, 90), font, .7, (0, 255, 0), 2)
        cv2.circle(crop, (crop_size / 2, crop_size / 2), 2, (0, 0, 255), 1)
        images.append(crop)
    composite_image = ci.get_composite_image(images, rows, cols)
    cv2.imwrite(html_dir + 'composite_image.png', composite_image)


def create_composite_image_from_filedata(crop_dir, data_dir, crop_size, rows, cols, filedata):
    html_dir = data_dir + 'html/'
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

def save_good_coin_ids(data_dir, seed_image_id,cut_off,remove_image_ids):
    #todo save_good_test_ids is not correct this needs a database:
    good_coin_ids = {}
    filename = data_dir + 'good_coin_ids.pickle'
    if os.path.exists(filename):
        #good_coin_ids = set(pickle.load(open(filename, "rb")))
        pass

    values = results_dict[seed_image_id].iteritems()
    for test_image_id, test_values in values:
        max_value, angle = test_values
        coin_id = test_image_id/100
        if max_value > cut_off:
            good_coin_ids.add(test_image_id)
        good_coin_ids.difference_update(remove_image_ids)
    print 'good_test_ids len: ' , len(good_coin_ids)
    pickle.dump(good_coin_ids, open(filename, "wb"))

