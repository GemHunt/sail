import time

import pandas as pd


def get_results(filename, seed_image_id, low_angle, high_angle):
    pd.set_option('display.max_rows', 10000)
    start_time = time.time()

    df = pd.read_csv(filename)
    # temp_key, key, ground_truth, prediction, result
    # 00000, 00000, 43, 1095, 0.3076

    if len(df.index) == 0:
        # Wow, it could not find anything. The image must be mud...
        return []

    del df['temp_key']
    df.prediction = df.ground_truth - (df.prediction - 1000)
    df_plus = df[df.prediction >= 0]
    df_neg = df[df.prediction < 0]
    df_neg.prediction += 360
    df = pd.concat([df_plus, df_neg])
    del df['ground_truth']
    df = df.groupby(['key', 'prediction']).result.sum().reset_index()
    filtered_results = []
    for image_id, image_results in df.groupby(['key']):
        top_result_index = image_results['result'].idxmax()
        angle = image_results.ix[top_result_index]['prediction']
        max_value = image_results.ix[top_result_index]['result']
        if (angle < low_angle) or (angle > high_angle):
            filtered_results.append([seed_image_id, image_id, int(angle), max_value])

    print 'Done reading results, with slow python, for seed image ID:' + str(seed_image_id) + ' in %s seconds' % (
        time.time() - start_time)
    return filtered_results
