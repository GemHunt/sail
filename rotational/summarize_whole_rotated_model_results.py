import numpy as np
import time

import pandas as pd
from pandas import Series


# import self_supervised_whole_rotated_crops as self_supervised_whole_rotated_crops

def summarize_whole_rotated_model_results(filename, seed_image_id, low_angle, high_angle):
    # angle_offset = 170
    angle_offset = 35
    # img = cv2.imread('/home/pkrush/lmdb-files/crops/1220.jpg')
    # cv2.imshow('test', img)
    # gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # gray = ci.center_rotate(gray, angle_offset)
    # cv2.imshow('test_rotated', gray)

    pd.set_option('display.max_rows', 10000)

    start_time = time.time()
    df = pd.read_csv(filename)
    # this needs to be taken out of the process earlier, key is useless data:
    del df['temp_key']

    print 'Done 1 %s seconds' % (time.time() - start_time,)
    results = np.zeros((360, 1), dtype=np.float)
    result_totals = np.zeros((360, 1), dtype=np.float)
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
        # plt.gcf().clear()
        # print 'Done 1 %s seconds' % (time.time() - start_time,)
        # figManager = plt.get_current_fig_manager()
        # figManager.window.showMaximized()
        # print 'Done 2 %s seconds' % (time.time() - start_time,)

        # This is being done wrong and can be speed up somehome:
        df_filtered = df[df.key == key]
        # print 'Done 3 %s seconds' % (time.time() - start_time,)
        result_totals = df_filtered.groupby('prediction')['result'].sum()
        # print 'Done 4 %s seconds' % (time.time() - start_time,)
        result_totals = result_totals + full_index
        # print 'Done 5 %s seconds' % (time.time() - start_time,)

        # result_totals = result_totals * correction
        # print 'Done 6 %s seconds' % (time.time() - start_time,)

        # plt.title(key)
        # smoth = result_totals
        # smoth = np.convolve(result_totals, [.0214, .1359, .3413, .3413, .1359, .0214 ], 'same')
        # smoth = np.convolve(result_totals, [.1,.1,.1,.1,.2,.2,.2,.2,.3,.3,.3,.3, .2,.2,.2,.2,.1,.1,.1,.1 ],'same')
        # print 'Done 7 %s seconds' % (time.time() - start_time,)

        angle = np.argmax(result_totals)
        # print 'Done 8 %s seconds' % (time.time() - start_time,)

        max_value = np.amax(result_totals)
        total_value = np.sum(result_totals)
        # print 'Done 9 %s seconds' % (time.time() - start_time,)

        results.append([key, max_value, angle, total_value])

    sorted_results = sorted(results, key=lambda result: result[1], reverse=True)
    index = 0
    filenames = []
    filtered_results = []
    for key, max_value, angle, total_value in sorted_results:
        filename = '/home/pkrush/lmdb-files/crops/' + str(key) + '.jpg'
        # img = cv2.imread(filename)
        # cv2.imshow('image', img)
        # gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)x
        # adjusted_angle = angle + angle_offset
        # gray = ci.center_rotate(gray, adjusted_angle)
        # cv2.imshow('image_rotated', gray)
        index += 1
        filenames.append([filename, angle])
        # cv2.waitKey(0)
        if (angle < low_angle) or (angle > high_angle):
            filtered_results.append([seed_image_id, key, angle, max_value])

    print str(seed_image_id) + 'Done %s seconds' % (time.time() - start_time,)
    return filtered_results

    # self_supervised_whole_rotated_crops.create_lmdbs(filenames,True)


ss = '''

    #for index, row in df.iterrows():
    #    angle = int(row['ground_truth']) - (1000-int(row['prediction']))
    #    if angle < 0:
    #        angle += 359

    #    if angle > 359:
    #        angle -= 359
    #    results[angle] += float(row['result'])
    #    result_totals[(1000-int(row['prediction']))] += float(row['result'])


    for count in result_totals(0, 360):
        if result_totals[count] < .1:
            result_totals[count] = .1
        result_totals[count] = 1/result_totals[count]
    print 'Done 4 %s seconds' % (time.time() - start_time,)
    for count in range(1, 359):
        correction[count] = results[count] * result_totals[count]
        mean = results[count-1] * result_totals[count-1] + results[count] * result_totals[count] + results[count+1] * result_totals[count+1]
        mean = mean/3
        result_mean[count] = mean
        print count,result_mean[count],correction[count],results[count] , result_totals[count]
    print 'Done 4 %s seconds' % (time.time() - start_time,)

    print correction
    print result_mean
    #plt.plot(corrected_results)
    #plt.plot(result_mean)
    #plt.show()



    rt =  np.sort(correction, axis=None)
    rt = rt[::-1]
    rt = 1/rt
    print ["{0:0.2f}".format(i) for i in rt]


    print results


    scale = df['result'].sum()
    print scale * 360




    all_classes = df.groupby('prediction')['result'].sum()



    print all_classes
    #scaled = all_classes.multiply(1/scale, fill_value=0)
    #print scaled
    #print scaled.sum()
    print 'Done after %s seconds' % (time.time() - start_time,)


'''
