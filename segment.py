# organize imports
import cv2
import imutils
import numpy as np
import time
from keras.models import Sequential
from keras.layers import *

def create_model():
    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(256, 256, 3)))
    model.add(MaxPool2D((2, 2)))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPool2D((2, 2)))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPool2D((2, 2)))
    model.add(Flatten())
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.4))
    model.add(Dense(3, activation='softmax'))
    model.load_weights("best_weights.hdf5")
    return model

def get_prediction(img, model):
    img = cv2.resize(img, (256,256), interpolation=cv2.INTER_AREA)
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    y_pred = model.predict(img.reshape(1, 256, 256, 3))
    y_pred = np.argmax(y_pred[0])
    return y_pred

def capture(cam):
    pred = []
    labels = {0: 'paper', 1: 'rock', 2: 'scissor'}
    i = 0
    top, right, bottom, left = 30, 350, 325, 630
    while i < 5:
        ret, frame = cam.read()
        frame = cv2.flip(frame, 1)

        # get the height and width of the frame
        (height, width) = frame.shape[:2]

        # get the ROI
        roi = frame[top:bottom, right:left]

        # convert the roi to grayscale and blur it
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (7, 7), 0)
        hand = segment(gray)
        if hand is not None:
            i += 1
            # if yes, unpack the thresholded image and
            # segmented region
            (thresholded, segmented) = hand
            pred.append(get_prediction(thresholded))

    pred = np.flip(pred)
    _, indices, c = np.unique(pred, return_counts=True, return_index=True)
    print("This worked till here")
    return labels[pred[indices[np.argmax(c)]]]


# global variables
bg = None

#--------------------------------------------------
# To find the running average over the background
#--------------------------------------------------


def run_avg(image, aWeight):
    global bg
    # initialize the background
    if bg is None:
        bg = image.copy().astype("float")
        return

    # compute weighted average, accumulate it and update the background
    cv2.accumulateWeighted(image, bg, aWeight)

#---------------------------------------------
# To segment the region of hand in the image
#---------------------------------------------


def segment(image, threshold=25):
    global bg
    # find the absolute difference between background and current frame
    diff = cv2.absdiff(bg.astype("uint8"), image)

    # threshold the diff image so that we get the foreground
    thresholded = cv2.threshold(diff, threshold, 255, cv2.THRESH_BINARY)[1]

    # get the contours in the thresholded image
    (_, cnts, _) = cv2.findContours(thresholded.copy(),
                                    cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # return None, if no contours detected
    if len(cnts) == 0:
        return
    else:
        # based on contour area, get the maximum contour which is the hand
        segmented = max(cnts, key=cv2.contourArea)
        return (thresholded, segmented)


#-----------------
# MAIN FUNCTION
#-----------------
if __name__ == "__main__":
    # initialize weight for running average
    aWeight = 0.5

    # get the reference to the webcam
    camera = cv2.VideoCapture(0)

    # region of interest (ROI) coordinates
    top, right, bottom, left = 30, 350, 325, 630

    # initialize num of frames
    num_frames = 0
    capture = 0
    path = "Images/"

    model = create_model()
    labels = {0: 'paper', 1: 'scissor'}

    font = cv2.FONT_HERSHEY_SIMPLEX

    # org
    org = (50, 50)

    # fontScale
    fontScale = 1

    # Blue color in BGR
    color = (255, 0, 0)

    # Line thickness of 2 px
    thickness = 2
    run = 0

    # keep looping, until interrupted
    while(True):
        # get the current frame
        (grabbed, frame) = camera.read()

        # resize the frame
        #frame = imutils.resize(frame, width=700)

        # flip the frame so that it is not the mirror view
        frame = cv2.flip(frame, 1)

        # clone the frame
        clone = frame.copy()

        # get the height and width of the frame
        (height, width) = frame.shape[:2]

        # get the ROI
        roi = frame[top:bottom, right:left]

        # convert the roi to grayscale and blur it
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (7, 7), 0)

        # to get the background, keep looking till a threshold is reached
        # so that our running average model gets calibrated
        if num_frames < 30:
            run_avg(gray, aWeight)
        elif num_frames >= 30 and run < 60:

            print("----------------------------------------------------")
            print("Rock")
            #time.sleep(1)
            print("Paper")
            #time.sleep(1)
            print("Scissors")
            #cv2.destroyWindow("Video Feed")

            pred = []
            i = 0
            top, right, bottom, left = 30, 350, 325, 630
            while i < 5:
                ret, frame = camera.read()
                frame = cv2.flip(frame, 1)

                # get the height and width of the frame
                (height, width) = frame.shape[:2]

                # get the ROI
                roi = frame[top:bottom, right:left]

                # convert the roi to grayscale and blur it
                gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
                #disp_gray = cv2.putText(gray, 'OpenCV', org, font, fontScale, color, thickness, cv2.LINE_AA)
                #cv2.imshow("Video Feed", disp_gray)
                disp_gray = gray.copy()
                gray = cv2.GaussianBlur(gray, (7, 7), 0)
                hand = segment(gray)
                if hand is not None:
                    i += 1
                    # if yes, unpack the thresholded image and
                    # segmented region
                    #(thresholded, segmented) = hand
                    pred.append(get_prediction(gray, model))
                    cv2.imshow("Your Move", disp_gray)
                    #cv2.imshow("Thesholded", thresholded)

            #pred = np.flip(pred)
            _, indices, c = np.unique(pred, return_counts=True, return_index=True)
            print("Prediction: {}".format(labels[pred[indices[np.argmax(c)]]]))

            print("----------------------------------------------------")
            run = 0
            """
            # segment the hand region
            hand = segment(gray)

            # check whether hand region is segmented
            if hand is not None:
                capture += 1
                # if yes, unpack the thresholded image and
                # segmented region
                (thresholded, segmented) = hand

                # draw the segmented region and display the frame
                #cv2.drawContours(clone, [segmented + (right, top)], -1, (0, 0, 255))

                if capture%20 == 0 and capture<121:
                    cv2.imwrite(path+"img_"+str(int(capture/20))+".jpg", thresholded)

                print("Prediction is {}".format(get_prediction(thresholded)))
                print("----------------------------------------------------")
                cv2.imshow("Thesholded", thresholded)

            """
                

        # draw the segmented hand
        cv2.rectangle(clone, (left, top), (right, bottom), (0, 255, 0), 2)

        # increment the number of frames
        num_frames += 1

        # display the frame with segmented hand
        cv2.imshow("Video Feed", clone)
        run += 1

        # observe the keypress by the user
        keypress = cv2.waitKey(1) & 0xFF

        # if the user pressed "q", then stop looping
        if keypress == ord("q"):
            break

# free up memory
camera.release()
cv2.destroyAllWindows()
