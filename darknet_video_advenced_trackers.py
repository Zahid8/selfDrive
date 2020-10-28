from ctypes import *
import math
import random
import os
import cv2
import numpy as np
import time
from imutils.video import FPS
from object_tracking import CentroidTracker
import darknet
# import the necessary packages
import imutils
import dlib


def convertBack(x, y, w, h):
    xmin = int(round(x - (w / 2)))
    xmax = int(round(x + (w / 2)))
    ymin = int(round(y - (h / 2)))
    ymax = int(round(y + (h / 2)))
    return xmin, ymin, xmax, ymax

def cvDrawBoxes(detections, img):
    for detection in detections:
        if detection[0].decode() == 'stop sign':
            x, y, w, h = detection[2][0],\
                detection[2][1],\
                detection[2][2],\
                detection[2][3]
            xmin, ymin, xmax, ymax = convertBack(
                float(x), float(y), float(w), float(h))
            pt1 = (xmin, ymin)
            pt2 = (xmax, ymax)
            cv2.rectangle(img, pt1, pt2, (0, 255, 0), 1)
            cv2.putText(img,
                        detection[0].decode() +
                        " [" + str(round(detection[1] * 100, 2)) + "]",
                        (pt1[0], pt1[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        [0, 255, 0], 2)
    return img


netMain = None
metaMain = None
altNames = None


def YOLO():

    global metaMain, netMain, altNames
    configPath = "./yolov3.cfg"
    weightPath = "./yolov3.weights"
    metaPath = "./coco.data"
    if not os.path.exists(configPath):
        raise ValueError("Invalid config path `" +
                         os.path.abspath(configPath)+"`")
    if not os.path.exists(weightPath):
        raise ValueError("Invalid weight path `" +
                         os.path.abspath(weightPath)+"`")
    if not os.path.exists(metaPath):
        raise ValueError("Invalid data file path `" +
                         os.path.abspath(metaPath)+"`")
    if netMain is None:
        netMain = darknet.load_net_custom(configPath.encode(
            "ascii"), weightPath.encode("ascii"), 0, 1)  # batch size = 1
    if metaMain is None:
        metaMain = darknet.load_meta(metaPath.encode("ascii"))
    if altNames is None:
        try:
            with open(metaPath) as metaFH:
                metaContents = metaFH.read()
                import re
                match = re.search("names *= *(.*)$", metaContents,
                                  re.IGNORECASE | re.MULTILINE)
                if match:
                    result = match.group(1)
                else:
                    result = None
                try:
                    if os.path.exists(result):
                        with open(result) as namesFH:
                            namesList = namesFH.read().strip().split("\n")
                            altNames = [x.strip() for x in namesList]
                except TypeError:
                    pass
        except Exception:
            pass
    cap = cv2.VideoCapture('http://192.168.0.11:4747/video')
    # cap = cv2.VideoCapture("project_video.mp4")

    # out = cv2.VideoWriter(
    #     "output.avi", cv2.VideoWriter_fourcc(*"MJPG"), 10.0,
    #     (darknet.network_width(netMain), darknet.network_height(netMain)))
    print("Starting the YOLO loop...")

    # Create an image we reuse for each detect
    darknet_image = darknet.make_image(darknet.network_width(netMain),
                                    darknet.network_height(netMain),3)

    fps = FPS().start()

    trackers = []
    labels = []

    while True:

        prev_time = time.time()
        ret, frame_read = cap.read()
        frame_rgb = cv2.cvtColor(frame_read, cv2.COLOR_BGR2RGB)
        frame_resized = cv2.resize(frame_rgb,
                                   (darknet.network_width(netMain),
                                    darknet.network_height(netMain)),
                                   interpolation=cv2.INTER_LINEAR)

        darknet.copy_image_from_bytes(darknet_image,frame_resized.tobytes())

        detections = darknet.detect_image(netMain, metaMain, darknet_image, thresh=0.8)
        if len(trackers) == 0:
            for detection in detections:
                if detection[0].decode() == 'stop sign':
                    x, y, w, h = detection[2][0], \
                                 detection[2][1], \
                                 detection[2][2], \
                                 detection[2][3]
                    xmin, ymin, xmax, ymax = convertBack(
                        float(x), float(y), float(w), float(h))
                    pt1 = (xmin, ymin)
                    pt2 = (xmax, ymax)

                    label = detection[0].decode()

                    t = dlib.correlation_tracker()
                    rect = dlib.rectangle(xmin, ymin, xmax, ymax)
                    t.start_track(frame_resized, rect)

                    labels.append(label)
                    trackers.append(t)

                    cv2.rectangle(frame_resized, pt1, pt2, (0, 255, 0), 1)
                    cv2.putText(frame_resized,
                                detection[0].decode() +
                                " [" + str(round(detection[1] * 100)) + "]",
                                (pt1[0], pt1[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                [0, 255, 0], 2)
        else:
            # loop over each of the trackers
            for (t, l) in zip(trackers, labels):
                # update the tracker and grab the position of the tracked
                # object
                t.update(frame_resized)
                pos = t.get_position()
                # unpack the position object
                startX = int(pos.left())
                startY = int(pos.top())
                endX = int(pos.right())
                endY = int(pos.bottom())
                # draw the bounding box from the correlation object tracker
                cv2.rectangle(frame_resized, (startX, startY), (endX, endY),
                              (0, 255, 0), 1)
                cv2.putText(frame_resized, l, (startX, startY - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)



        # image = cvDrawBoxes(detections, frame_resized)
        image = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
        cv2.imshow('Demo', image)
        keys = cv2.waitKey(1) & 0xFF
        fps.update()

        print(labels)
        if keys == ord("q"):
            break
    cap.release()
    fps.stop()
    print("[INFO] elasped time: {:.2f}".format(fps.elapsed()))
    print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))
    # out.release()

if __name__ == "__main__":
    YOLO()
