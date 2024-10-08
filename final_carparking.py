import cv2
import pickle

import numpy as np

width = 107
height = 48

with open('carparkpos', 'rb') as f:
    posList = pickle.load(f)

cap = cv2.VideoCapture("Image_Video/video.mp4")

# count non-zero pixels in each of the parking slot
def checkparkingspace(preprocessed_frame):
    counter = 0
    if len(posList)!=0:
        for pos in posList:
            x, y = pos
            cropped_frame = preprocessed_frame[y:y+height, x:x+width]
            # it give all 71 frame individual
            # cv2.imshow(str(x*y), cropped_frame)

            count = cv2.countNonZero(cropped_frame)

            if count<900:
                counter+=1
                # change the color of vacant parking space
                color = (100,255,100)
            else:
                color = (100,100,255)

            cv2.rectangle(frame, (pos[0],pos[1]), (pos[0] + width, pos[1] + height), color, 2)
            cv2.putText(frame, str(count), (pos[0], pos[1]+5), 0 , 0.5, [255,255,255], thickness=1, lineType=cv2.LINE_AA)

        cv2.rectangle(frame, (51,15), (51+width+100, 15+height+10), (255, 0, 255), cv2.FILLED)
        cv2.putText(frame, f'Free: {counter}/{len(posList)}', (52, 15+height), 0, 1, [255,255,255], thickness=2, lineType=cv2.LINE_AA)

while True:
    # video is short in length then we want to continue play this video, don't want to stop it
    # position of current frame is equal to total frame count then
    if cap.get(cv2.CAP_PROP_POS_FRAMES) == cap.get(cv2.CAP_PROP_FRAME_COUNT):
        # it start with again zero frame
        cap.set(cv2.CAP_PROP_POS_FRAMES,0)

    ret, frame = cap.read()
    if ret:
        # convert our frame to grayscale
        gray_scale = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Add gaussian Blur
        blur = cv2.GaussianBlur(gray_scale, (3,3), 1)

        # Applying threshold on each of the frame of the video             and 25, 16 is depend on your video
        frame_threshold = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 25, 16)

        # to remove the noise dots as will apply median blur
        median_blur = cv2.medianBlur(frame_threshold, 5)

        # Applying dialation to increase the thickness of our edges
        kernel = np.ones((3,3), np.uint8)
        frame_dilate = cv2.dilate(median_blur, kernel, iterations=1)

        checkparkingspace(frame_dilate)

        cv2.imshow("video", frame)
        if cv2.waitKey(1) & 0xFF == ord('1'):
            break
    else:
        break