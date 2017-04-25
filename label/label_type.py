import cPickle as pickle
import cv2
import os
import numpy as np
import sys

def save_and_exit(labeled_dates,ground_truth_dates):
    pickle.dump(labeled_dates, open(data_dir + 'labeled_dates.pickle', "wb"))
    new_ground_truth_dates = []
    for seed_id, coin_id, result, labeled_date, bad_angle, bad_image in ground_truth_dates:
       if coin_id in labeled_dates.iterkeys():
           labeled_date = labeled_dates[coin_id]
       new_ground_truth_dates.append([seed_id, coin_id, result, labeled_date, bad_angle, bad_image])

    pickle.dump(new_ground_truth_dates, open(data_dir + 'ground_truth_dates.pickle', "wb"))
    sys.exit()


data_dir = '/home/pkrush/cent-date-models/metadata/'
crop_dir = '/home/pkrush/cent-dates/'

cv2.namedWindow("next")

ground_truth_dates = pickle.load(open(data_dir + 'ground_truth_dates.pickle', "rb"))
labeled_dates = pickle.load(open(data_dir + 'labeled_dates.pickle', "rb"))
ground_truth_date_dict = {}

images = {}

ground_truth_dates = sorted(ground_truth_dates, key=lambda x: x[3], reverse=False)

for count in range(0,len(ground_truth_dates)):
    seed_id,coin_id, result,labeled_date,bad_angle,bad_image = ground_truth_dates[count]
    if coin_id in labeled_dates.iterkeys():
        pass
        #continue
    crop_dir = '/home/pkrush/cent-dates/'
    dir = crop_dir + str(coin_id / 100) + '/'
    filename = dir + str(coin_id).zfill(5) + str(54).zfill(2) + '.png'
    print filename
    if not os.path.exists(filename):
        labeled_dates[coin_id] = -2
        continue
    image = cv2.imread(filename)
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(image, str(labeled_date % 100)[0:5], (4, 20), font, .7, (0, 255, 0), 2)

    images[coin_id] = image

    last_labeled_date = 0

for count in range(0, len(ground_truth_dates)):
    seed_id, coin_id, result, old_labeled_date, bad_angle, bad_image = ground_truth_dates[count]
    if not coin_id in images.iterkeys():
        continue
    next = np.zeros((56, 560, 3), np.uint8)
    for count2 in range(0,10):
        if count + count2 >= len(ground_truth_dates):
            continue
        next_coin_id = ground_truth_dates[count + count2 ][1]
        if next_coin_id in images.iterkeys():
            next_image = images[next_coin_id]
        else:
            next_image = np.zeros((56, 56, 3), np.uint8)
        next[0:56,(9-count2) * 56:((10-count2) * 56)] = next_image

    decade  = -1
    year = -1
    labeled_date = -999

    while True:
        # display the image and wait for a keypress
        cv2.imshow("image", images[coin_id])
        cv2.imshow("next", next)

        key = cv2.waitKey(1) & 0xFF
        if key !=255:
            print key

        if key == ord("a"):
            labeled_date = 0
        if key == ord("b"):
            labeled_date = -1


        if 47 < key < 58:
            if decade == -1:
                decade = key - 48
            else:
                year = key - 48
                if decade < 2:
                    labeled_date = 2000 + decade * 10 + year
                else:
                    labeled_date  = 1900 + decade * 10 + year

        if key == 10: #enter
            labeled_date = last_labeled_date

        if labeled_date != -999:
             labeled_dates[coin_id] = labeled_date
             print labeled_date
             last_labeled_date = labeled_date
             break

        if key == 27: #esc
            break

        if key == ord("q"):
            save_and_exit(labeled_dates,ground_truth_dates)

save_and_exit(labeled_dates,ground_truth_dates)
cv2.destroyAllWindows()