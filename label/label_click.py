#http://www.pyimagesearch.com/2015/03/09/capturing-mouse-click-events-with-python-and-opencv/
import argparse
import cv2

# initialize the list of reference points and boolean indicating
# whether cropping is being performed or not
refPt = []
cropping = False


def click_and_crop(event, x, y, flags, param):
    # grab references to the global variables
    global refPt

    # if the left mouse button was clicked, record the starting
    # (x, y) coordinates and indicate that cropping is being
    # performed
    if event == cv2.EVENT_LBUTTONDOWN:
        refPt = [(x, y)]
        print refPt

home_dir = '/home/pkrush/cent-date-models/metadata/html/'

filename= home_dir + '00008200.png'

# load the image, clone it, and setup the mouse callback function
image = cv2.imread(filename)
cv2.namedWindow("image")
cv2.setMouseCallback("image", click_and_crop)

# keep looping until the 'q' key is pressed
while True:
    # display the image and wait for a keypress
    cv2.imshow("image", image)
    key = cv2.waitKey(1) & 0xFF

      # if the 'c' key is pressed, break from the loop
    if key == ord("c"):
        break


cv2.destroyAllWindows()