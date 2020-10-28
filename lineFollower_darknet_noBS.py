import numpy as np
import cv2
import socket
import msgmaker
from pynput import keyboard
from pynput.keyboard import Key
import datetime
import os
import time
from object_tracking2 import CentroidTracker
import darknet
from imutils.video import WebcamVideoStream
from matplotlib import pyplot as plt
import pathlib
from fastai.vision import *
import re
from predictor import predictor

# print(cv2.getBuildInformation())
# ustanawianie połączenia:
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM,0)
s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
s.bind(('192.168.0.2', 1234))
s.listen(0)
clientsocket, address = s.accept()
print(f"Connection from {address} has been established.")

filename = 'LineFollower' + str(datetime.datetime.now().strftime("%Y-%m-%d (%H!%M!%S)")) + '.avi'

frame_save = True
on_board_camera = False
motion_capture = False
frameWidth = 640
frameHeight = 480

is_wheel_connected = False
lights = False
gear = 0

auto = False
autoEnv = False

speed = 60
manualspeed = speed
autospeed = 45
turn = 40
spin = 45
arrows = [Key.up, Key.down, Key.left, Key.right]

last_throttle = 0

def leftnup(speed, turn):
    leftsign = 1
    if 0 <= speed < 10:
        leftspeedstring = '00' + str(speed)
    elif 10 <= speed < 100:
        leftspeedstring = '0' + str(speed)
    elif speed >= 100:
        leftspeedstring = str(speed)
    rightsign = 1
    rightturn = speed + turn
    if 0 <= rightturn < 10:
        rightspeedstring = '00' + str(rightturn)
    elif 10 <= rightturn < 100:
        rightspeedstring = '0' + str(rightturn)
    elif rightturn >= 100:
        rightspeedstring = str(rightturn)
    msg = '<1' + str(leftsign) + leftspeedstring + str(rightsign) + rightspeedstring + '>'
    clientsocket.send(bytes(msg + "\n", "utf-8"))


def rightnup(speed, turn):
    leftsign = 1
    leftturn = speed + turn
    if 0 <= leftturn < 10:
        leftspeedstring = '00' + str(leftturn)
    elif 10 <= leftturn < 100:
        leftspeedstring = '0' + str(leftturn)
    elif leftturn >= 100:
        leftspeedstring = str(leftturn)
    rightsign = 1
    if 0 <= speed < 10:
        rightspeedstring = '00' + str(speed)
    elif 10 <= speed < 100:
        rightspeedstring = '0' + str(speed)
    elif speed >= 100:
        rightspeedstring = str(speed)
    msg = '<1' + str(leftsign) + leftspeedstring + str(rightsign) + rightspeedstring + '>'
    clientsocket.send(bytes(msg + "\n", "utf-8"))


def rightndown(speed, turn):
    leftsign = 0
    leftturn = speed + turn
    if 0 <= leftturn < 10:
        leftspeedstring = '00' + str(leftturn)
    elif 10 <= leftturn < 100:
        leftspeedstring = '0' + str(leftturn)
    elif leftturn >= 100:
        leftspeedstring = str(leftturn)
    rightsign = 0
    if 0 <= speed < 10:
        rightspeedstring = '00' + str(speed)
    elif 10 <= speed < 100:
        rightspeedstring = '0' + str(speed)
    elif speed >= 100:
        rightspeedstring = str(speed)
    msg = '<1' + str(leftsign) + leftspeedstring + str(rightsign) + rightspeedstring + '>'
    clientsocket.send(bytes(msg + "\n", "utf-8"))


def leftndown(speed, turn):
    leftsign = 0
    if 0 <= speed < 10:
        leftspeedstring = '00' + str(speed)
    elif 10 <= speed < 100:
        leftspeedstring = '0' + str(speed)
    elif speed >= 100:
        leftspeedstring = str(speed)
    rightsign = 0
    rightturn = speed + turn
    if 0 <= rightturn < 10:
        rightspeedstring = '00' + str(rightturn)
    elif 10 <= rightturn < 100:
        rightspeedstring = '0' + str(rightturn)
    elif rightturn >= 100:
        rightspeedstring = str(rightturn)
    msg = '<1' + str(leftsign) + leftspeedstring + str(rightsign) + rightspeedstring + '>'
    clientsocket.send(bytes(msg + "\n", "utf-8"))


# Create a mapping of keys to function (use frozenset as sets are not hashable - so they can't be used as keys)
combination_to_function = {
    frozenset([Key.up, Key.left]): leftnup,
    frozenset([Key.up, Key.right]): rightnup,
    frozenset([Key.down, Key.left]): leftndown,
    frozenset([Key.down, Key.right]): rightndown,
}

current_keys = set()

lista = []


def up(speed):
    leftsign = 1
    rightsign = 1
    if 0 <= speed < 10:
        speedstring = '00' + str(speed)
    elif 10 <= speed < 100:
        speedstring = '0' + str(speed)
    elif speed >= 100:
        speedstring = str(speed)
    msg = '<1' + str(leftsign) + speedstring + str(rightsign) + speedstring + '>'
    return msg


def down(speed):
    leftsign = 0
    rightsign = 0
    if 0 <= speed < 10:
        speedstring = '00' + str(speed)
    elif 10 <= speed < 100:
        speedstring = '0' + str(speed)
    elif speed >= 100:
        speedstring = str(speed)
    msg = '<1' + str(leftsign) + speedstring + str(rightsign) + speedstring + '>'
    return msg


def left(speed):
    leftsign = 0
    rightsign = 1
    if 0 <= speed < 10:
        speedstring = '00' + str(speed)
    elif 10 <= speed < 100:
        speedstring = '0' + str(speed)
    elif speed >= 100:
        speedstring = str(speed)
    msg = '<1' + str(leftsign) + speedstring + str(rightsign) + speedstring + '>'
    return msg


def right(speed):
    leftsign = 1
    rightsign = 0
    if 0 <= speed < 10:
        speedstring = '00' + str(speed)
    elif 10 <= speed < 100:
        speedstring = '0' + str(speed)
    elif speed >= 100:
        speedstring = str(speed)
    msg = '<1' + str(leftsign) + speedstring + str(rightsign) + speedstring + '>'
    return msg


def setturn(turning, setti):
    global turn
    if setti == 1:
        if 0 <= turn < 40:
            turn += 1
        print("Promień skrętu: " + str(turn) + "%.")
    elif setti == 0:
        if 0 < turn <= 40:
            turn -= 1
        print("Promień skrętu: " + str(turn) + "%.")


def setspeed(setti):
    global speed
    global autospeed
    if auto and setti == 1:
        if 0 <= autospeed < 100:
            autospeed += 1
        print("Moc silników: " + str(speed) + "%.")
    if auto and setti == 0:
        if 0 <= autospeed <= 100:
            autospeed -= 1
        print("Moc silników: " + str(speed) + "%.")
    if setti == 1:
        if 0 <= speed < 100:
            speed += 1
        print("Moc silników: " + str(speed) + "%.")
    elif setti == 0:
        if 0 < speed <= 100:
            speed -= 1
        print("Moc silników: " + str(speed) + "%.")


def on_press(key):
    global auto
    global speed
    global manualspeed
    global autospeed
    global lights
    global is_wheel_connected
    if auto and key in arrows:
        # song.terminate()
        auto = False
        speed = manualspeed

        print("Tryb autonomiczny wyłączony.")
        print("Moc silników: " + str(speed) + "%.")
    if key not in lista:
        lista.append(key)
    current_keys.add(key)
    if frozenset(current_keys) in combination_to_function:
        # If the current set of keys are in the mapping, execute the function
        combination_to_function[frozenset(current_keys)](speed, turn)
    else:
        if key == Key.up:
            clientsocket.send(bytes(up(speed) + "\n", "utf-8"))
        elif key == Key.left:
            clientsocket.send(bytes(left(spin) + "\n", "utf-8"))
        elif key == Key.right:
            clientsocket.send(bytes(right(spin) + "\n", "utf-8"))
        elif key == Key.down:
            clientsocket.send(bytes(down(speed) + "\n", "utf-8"))
        # Ustawianie prędkości bazowej:
        elif key == keyboard.KeyCode(char='t'):
            setspeed(1)
        elif key == keyboard.KeyCode(char='r'):
            setspeed(0)
        # Ustawianie promienia skrętu:
        elif key == keyboard.KeyCode(char='g'):
            setturn(turn, 1)
        elif key == keyboard.KeyCode(char='f'):
            setturn(turn, 0)
        elif key == keyboard.KeyCode(char='a'):

            auto = True
            autospeed = 40
            print("Włączono tryb autonomiczny.")
            print("Moc silników: " + str(autospeed) + "%.")

        elif key == keyboard.KeyCode(char='l'):
            if lights is True:
                lights = False
                clientsocket.send(bytes("<200>" + "\n", "utf-8"))
            elif lights is False:
                lights = True
                clientsocket.send(bytes("<201>" + "\n", "utf-8"))
        elif key == keyboard.KeyCode(char='q'):
            is_wheel_connected = False


def on_release(key):
    if key == keyboard.Key.esc:
        print('we are done')
        return False
    current_keys.remove(key)
    lista.remove(key)
    if len(lista) > 0 and key in arrows:
        on_press(lista[0])
    elif key in arrows:
        msg = '<110001000>'
        clientsocket.send(bytes(msg + "\n", "utf-8"))


listener = keyboard.Listener(
    on_press=on_press,
    on_release=on_release)
listener.start()


def empty(x):
    pass

trackbars = np.zeros((300, 600, 3), np.uint8)
cv2.namedWindow('Trackbars')
cv2.createTrackbar('Kp', 'Trackbars', 50, 100, empty)
cv2.createTrackbar('Kd', 'Trackbars', 40, 100, empty)
cv2.createTrackbar('Ka', 'Trackbars', 0, 100, empty)
cv2.createTrackbar('ROI_X', 'Trackbars', 200, int(frameWidth/2), empty)
cv2.createTrackbar('ROI_Y', 'Trackbars', 100, int(frameHeight/2), empty)
cv2.createTrackbar('ROI_X_OFFSET', 'Trackbars', 320, frameWidth, empty)
cv2.createTrackbar('ROI_Y_OFFSET', 'Trackbars', 350, frameHeight, empty)
cv2.createTrackbar('Erode', 'Trackbars', 5, 20, empty)
cv2.createTrackbar('Dilate', 'Trackbars', 9, 20, empty)
cv2.createTrackbar('All', 'Trackbars', 1, 1, empty)
cv2.createTrackbar('Black_thresh', 'Trackbars', 30, 200, empty)

cv2.namedWindow("preview")
cv2.moveWindow("pipeline", 600,600)
cv2.moveWindow("preview", 667,0)
cv2.moveWindow("Trackbars", 0, 0)


last_error = 0
last_total_error = 0
current_error = 0



### DARKNET:
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


# global metaMain, netMain, altNames
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

# Create an image we reuse for each detect
darknet_image = darknet.make_image(darknet.network_width(netMain),
                                darknet.network_height(netMain),3)

ct = CentroidTracker()

# capture = cv2.VideoCapture('http://192.168.0.11:4747/video')
capture = WebcamVideoStream(src='http://192.168.0.11:4747/video').start()

on_board_camera = True

executing_stop = False
stop_timer = None
signs_executed = []

cv2.namedWindow("preview", cv2.WINDOW_NORMAL)


# Graph data:
error_thresh = 100
x_val = list(range(-error_thresh, error_thresh + 1))
y_val = [0 for element in x_val]

# Making directories for frames saving (ImageNet style):
if frame_save:
    for c in x_val:
        pathlib.Path(f'/home/rafal/PycharmProjects/naukaOpenCV/saves_test/{c}').mkdir(parents=True, exist_ok=True)


def file_counter(e):
    return len(os.listdir(f'/home/rafal/PycharmProjects/naukaOpenCV/saves_test/{e}'))


learn = load_learner(path='/home/rafal/PycharmProjects/naukaOpenCV/saves', file="stg1.4-bs32-s-60-80-best.pkl")

done_recording = False

defaults.device = torch.device('cuda')


counter = 0
# time.sleep(1)
start_time = time.time()
while True:
    rects = []
    counter += 1
    image = capture.read()
    # category, tensor, all_tensors = learn.predict(Image(pil2tensor(image, np.float32).div_(255)))
    # print(tensor)
    # new_string = re.search(r'\d+', str(tensor))
    # print(new_string.group(0))
    # label = int(classes[int(new_string.group(0))])
    # print(label)
    image_to_record = image.copy()
    # succesfull, image = capture.read()
    frame_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    frame_resized = cv2.resize(frame_rgb,
                               (darknet.network_width(netMain),
                                darknet.network_height(netMain)),
                               interpolation=cv2.INTER_LINEAR)

    darknet.copy_image_from_bytes(darknet_image, frame_resized.tobytes())

    detections = darknet.detect_image(netMain, metaMain, darknet_image, thresh=0.4)
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
            cv2.rectangle(frame_resized, pt1, pt2, (0, 255, 0), 1)
            cv2.putText(frame_resized,
                        detection[0].decode() +
                        " [" + str(round(detection[1] * 100)) + "]",
                        (pt1[0], pt1[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        [0, 255, 0], 2)
            rects.append([xmin, ymin, xmax, ymax])

    objects = ct.update(rects)

    # loop over the tracked objects
    for (objectID, centroid) in objects.items():
        # print(f'Wysokość znaku {objectID}: {centroid[2]}[BpFE]')
        text = f'Height:{centroid[2]}[BpFE]'
        # draw both the ID of the object and the centroid of the
        # object on the output frame
        # text = "ID {}".format(objectID)
        cv2.putText(frame_resized, text, (centroid[0] - 10, centroid[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        # cv2.circle(frame_resized, (centroid[0], centroid[1]), 4, (0, 255, 0), -1)

    image = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (frameWidth, frameHeight), interpolation=cv2.INTER_LINEAR)

    kp = cv2.getTrackbarPos('Kp', 'Trackbars') / 25 #100
    kd = cv2.getTrackbarPos('Kd', 'Trackbars') / 5 #25
    ka = cv2.getTrackbarPos('Ka', 'Trackbars') / 100
    erode = cv2.getTrackbarPos('Erode', 'Trackbars')
    dilate = cv2.getTrackbarPos('Dilate', 'Trackbars')
    show_all = cv2.getTrackbarPos('All', 'Trackbars')
    roi_x = cv2.getTrackbarPos('ROI_X', 'Trackbars')
    roi_y = cv2.getTrackbarPos('ROI_Y', 'Trackbars')
    roi_x_offset = cv2.getTrackbarPos('ROI_X_OFFSET', 'Trackbars')
    roi_y_offset = cv2.getTrackbarPos('ROI_Y_OFFSET', 'Trackbars')
    black_thresh = cv2.getTrackbarPos('Black_thresh', 'Trackbars')

    # Setting up Region Of Interest size (cannot be less than zero):
    cameraOffset = 0
    if roi_x_offset < int(frameWidth/2):
        if int(frameWidth/2) - roi_x - (int(frameWidth/2) - roi_x_offset) > 0:
            x1 = int(frameWidth/2) - roi_x - (int(frameWidth/2) - roi_x_offset)
        else:
            x1 = 0
    else:
        x1 = int(frameWidth/2) - roi_x + roi_x_offset - int(frameWidth/2)

    if roi_x_offset < int(frameWidth/2):
        x2 = int(frameWidth/2) + roi_x - (int(frameWidth/2) - roi_x_offset)
    else:
        if int(frameWidth/2) + roi_x + roi_x_offset - int(frameWidth/2) < frameWidth:
            x2 = int(frameWidth/2) + roi_x + roi_x_offset - int(frameWidth/2)
        else:
            x2 = frameWidth

    if roi_y_offset < int(frameHeight/2):
        if int(frameHeight/2) - roi_y - (int(frameHeight/2) - roi_y_offset) > 0:
            y1 = int(frameHeight/2) - roi_y - (int(frameHeight/2) - roi_y_offset)
        else:
            y1 = 0
    else:
        y1 = int(frameHeight/2) - roi_y + roi_y_offset - int(frameHeight/2)

    if roi_y_offset < int(frameHeight/2):
        y2 = int(frameHeight/2) + roi_y - (int(frameHeight/2) - roi_y_offset)
    else:
        if int(frameHeight/2) + roi_y + roi_y_offset - int(frameHeight/2) < frameHeight:
            y2 = int(frameHeight/2) + roi_y + roi_y_offset - int(frameHeight/2)
        else:
            y2 = frameHeight

    if x2 - x1 > 0 and y2 - y1 > 0:
        roi = image[y1:y2, x1:x2] # OD DO: WYSOKOSĆ, OD DO: SZEROKOŚĆ
    else:
        roi = image[y1:frameHeight, 0:frameWidth]  # OD DO: WYSOKOSĆ, OD DO: SZEROKOŚĆ
    if show_all == 1:
        cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 0), 5)   # X1,Y1
    # print(x2-x1)
    hsv = int(cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)[..., 2].mean())
    # print(hsv)
    #
    # blk_thresh = black_thresh  # MI MNIEJSZA TYM MNIEJ LINI WIDZI ALE WIĘCEJ FILTRUJE. IM CIEMNIEJ TYM MUSI BYĆ NIŻSZA
    blk_thresh = hsv + black_thresh  # MI MNIEJSZA TYM MNIEJ LINI WIDZI ALE WIĘCEJ FILTRUJE. IM CIEMNIEJ TYM MUSI BYĆ NIŻSZA
    # cv2.putText(image, str(blk_thresh), (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
    #
    Blackline = cv2.inRange(roi, (blk_thresh, blk_thresh, blk_thresh), (255, 255, 255))
    kernel = np.ones((3, 3), np.uint8)
    Blackline = cv2.erode(Blackline, kernel, iterations=erode)
    Blackline = cv2.dilate(Blackline, kernel, iterations=dilate)
    contours_blk, hierarchy_blk = cv2.findContours(Blackline, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours_blk_len = len(contours_blk)
    if show_all == 1:
        cv2.drawContours(roi, contours_blk, -1, (0, 255, 0), 3)
    if len(contours_blk) == 0 and auto is False and show_all == 1:
        cv2.putText(image, "Auto: ", (240, 470), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)  # NA CZERWONO
        autoEnv = False
    if len(contours_blk) == 0 and auto is True and show_all == 1:
        cv2.putText(image, "Auto: ON", (240, 470), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)  # NA ZIELONO

    if contours_blk_len > 0:
        if contours_blk_len == 0:
            blackbox = cv2.minAreaRect(contours_blk[0])
        else:
            candidates = []
            for con_num in range(contours_blk_len):
                blackbox = cv2.minAreaRect(contours_blk[con_num])
                box = cv2.boxPoints(blackbox)
                (x_box, y_box) = box[0]
                candidates.append((y_box, con_num))
            candidates = sorted(candidates)
            (y_highest, con_highest) = candidates[contours_blk_len-1]
            blackbox = cv2.minAreaRect(contours_blk[con_highest])
        x, y, w, h = cv2.boundingRect(contours_blk[0])
        heightOfContour = h
        # if show_all == 1:
        #     cv2.rectangle(roi, (x, y), (x + w, y + h), (0, 255, 255), 3)
        #     cv2.line(roi, (x + (w // 2), 0), (x + (w // 2), 640), (255, 0, 0), 3)

        box = cv2.boxPoints(blackbox)
        if show_all == 1:
            cv2.drawContours(roi, contours_blk, -1, (0, 255, 0), 3)
        (x_min, y_min), (w_min, h_min), ang = blackbox
        if ang < -45:
            ang = 90 + ang
        if w_min < h_min and ang > 0:
            ang = (90 - ang) * -1
        if w_min > h_min and ang < 0:
            ang = 90 + ang

        ang = int(ang)
        box = cv2.boxPoints(blackbox)
        box = np.int0(box)
        if show_all == 1:
            cv2.drawContours(roi, [box], 0, (0, 0, 255), 3)
        # cv2.putText(image, str(ang), (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

        if h < 50 and auto is False and show_all == 1:
            cv2.putText(image, "Auto: ", (240, 470), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)  # NA CZERWONO
            autoEnv = False
        elif h >= 50 and auto is False and show_all == 1:
            cv2.putText(image, "Auto: ", (240, 470), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)  # NA ZIELONO
            autoEnv = True

        setpoint = (x2 - x1)/2 + cameraOffset
        error1 = int(x_min - setpoint)

        if show_all == 1:
            cv2.line(roi, (int(x_min), 0), (int(x_min), 640), (255, 0, 0), 3)
        # print("last error: " + str(last_error) + "Error: " + str(error))
        # error_thresh = 60

        # print(kp)

        error = predictor(image_to_record)
        print(error)
        # if abs(error - last_error) >

        if error > error_thresh:
            error = error_thresh
        if error < -error_thresh:
            error = -error_thresh
        centertext = "Error = " + str(error)
        cv2.putText(image_to_record, centertext, (0, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        total_error = round((error * kp) + (kd * (error - last_error)) + (ka * ang))
        # print(total_error)
        error_data = "Total Error: " + str(total_error) + "Error: " + str(int(error * kp)) + " P:" + str(int(kp*error)) + " D:" + str(int(kd*error)) + " A:" + str(int(ka*error))
        if abs(total_error) > 30:
            current_error = total_error
            autowheelturn = True
        # cv2.putText(image, error_data, (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

        # Wysyłanie do clienta:
        if auto and autoEnv and motion_capture is False:

            recording = False

            # Saving frame in folder with error name (imagenet style):
            if frame_save and file_counter(error) <= 100 and recording:
                cv2.imwrite(f'/home/rafal/PycharmProjects/naukaOpenCV/saves_test/{error}/{datetime.datetime.now().time()}.png', image_to_record)

            if all(file_counter(e) for e in x_val) == 101 and recording:
                for e in x_val:
                    number_files = len(os.listdir(f'/home/rafal/PycharmProjects/naukaOpenCV/saves_test/{e}'))
                    print(f'{e}: {number_files}')
                done_recording = True
                print('done')
                break


            # Graph:
            y_val[x_val.index(error)] += 1

            speed = autospeed
            if show_all == 1:
                cv2.putText(image, "Auto: ON ", (240, 470), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            for (objectID, centroid) in objects.items():
                if centroid[2] >= 60 and objectID not in signs_executed:
                    executing_stop = True
                    print(executing_stop)
                    signs_executed.append(objectID)
            if executing_stop:
                if stop_timer is None:
                    stop_timer = time.time()
                    print(stop_timer)
                    clientsocket.send(bytes('<110001000>' + "\n", "utf-8"))
                if time.time() - stop_timer < 3:
                    print('stopped')
                    clientsocket.send(bytes('<110001000>' + "\n", "utf-8"))
                else:
                    stop_timer = None
                    executing_stop = False
                    print("moving on")
            else:
                clientsocket.send(bytes(msgmaker.msgmaker3(total_error, speed) + "\n", "utf-8"))
            last_error = error
            last_total_error = total_error
    cv2.namedWindow("preview", cv2.WINDOW_NORMAL)

    cv2.imshow("preview", image_to_record)
    # cv2.imshow("image_to_record", image_to_record)


    # KONIEC:
    keys = cv2.waitKey(1) & 0xFF
    if keys == ord("q"):
        msg = '<110001000>'
        clientsocket.send(bytes(msg + "\n", "utf-8"))
        clientsocket.send(bytes('<200>' + "\n", "utf-8"))
        msg = 'koniec'
        clientsocket.send(bytes(msg + "\n", "utf-8"))
        clientsocket.close()
        s.close()
        break
# If break from recorder:
if done_recording:
    msg = '<110001000>'
    clientsocket.send(bytes(msg + "\n", "utf-8"))
    clientsocket.send(bytes('<200>' + "\n", "utf-8"))
    msg = 'koniec'
    clientsocket.send(bytes(msg + "\n", "utf-8"))
    clientsocket.close()
    s.close()
###

finish_time = time.time()
fps = counter / (finish_time-start_time)
print('Frames per second: ' + str(fps))
print(f"Liczba klatek: {counter}")
cv2.destroyAllWindows()
plt.bar(x_val, y_val)
plt.show()

