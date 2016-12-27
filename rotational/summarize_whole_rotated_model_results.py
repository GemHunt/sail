import numpy as np
import time

import pandas as pd
from pandas import Series


# import self_supervised_whole_rotated_crops as self_supervised_whole_rotated_crops

def summarize_whole_rotated_model_results(filename, seed_image_id, low_angle, high_angle):
    pd.set_option('display.max_rows', 10000)
    start_time = time.time()
    df = pd.read_csv(filename)
    # this needs to be taken out of the process earlier, key is useless data:
    del df['temp_key']

    print 'Done 1 %s seconds' % (time.time() - start_time,)
    results = np.zeros((360, 1), dtype=np.float)
    # OK to Comment out?:
    # result_totals = np.zeros((360, 1), dtype=np.float)
    correction = np.zeros((360, 1), dtype=np.float)
    result_mean = np.zeros((360, 1), dtype=np.float)
    print 'Done 2 %s seconds' % (time.time() - start_time,)

    df.prediction = df.ground_truth - (df.prediction - 1000)
    df_plus = df[df.prediction >= 0]
    df_neg = df[df.prediction < 0]
    df_neg.prediction += 360
    df = pd.concat([df_plus, df_neg])

    # Reflect on 180
    # df.prediction = df.prediction - 180
    # df_plus = df[df.prediction >= 0]
    # df_neg = df[df.prediction < 0]
    # df_neg.prediction = df_neg.prediction + 180
    # df = pd.concat([df_plus,df_neg])

    result_totals = df.groupby('prediction')['result'].sum().values
    # correction  = result_totals / (result_totals.sum() / 360)
    keys = df.key.unique()
    results = []

    # for count in range(0, 360):
    #    if correction[count] < .5:
    #        correction[count] = .5
    #    correction[count] = 1 / correction[count]

    print 'Done 2.5 %s seconds' % (time.time() - start_time,)

    index = range(0, 360)
    full_index = Series([0] * 360, index=index)

    for key in keys:
        # This is being done wrong and can be speed up somehome:
        df_filtered = df[df.key == key]

        result_totals = df_filtered.groupby('prediction')['result'].sum()
        result_totals = result_totals + full_index
        angle = np.argmax(result_totals)
        max_value = np.amax(result_totals)
        total_value = np.sum(result_totals)
        results.append([key, max_value, angle, total_value])
    sorted_results = sorted(results, key=lambda result: result[1], reverse=True)
    # filenames = []
    filtered_results = []
    for key, max_value, angle, total_value in sorted_results:
        # filename = '/home/pkrush/cents/' + str(key) + '.jpg'
        # filenames.append([filename, angle])
        if (angle < low_angle) or (angle > high_angle):
            filtered_results.append([seed_image_id, key, angle, max_value])

    print str(seed_image_id) + 'Done %s seconds' % (time.time() - start_time,)
    return filtered_results

    # self_supervised_whole_rotated_crops.create_lmdbs(filenames,True)
