print("Loading libraries and models")
import numpy as np
import cv2
import socket
import msgmaker
from pynput import keyboard
from pynput.keyboard import Key
import datetime
import os
import time
import utlis
import darknet
from imutils.video import WebcamVideoStream
from matplotlib import pyplot as plt
import pathlib
from fastai.vision import *
import predictor
from object_tracking2 import CentroidTracker


print("Connecting to host")
# print(cv2.getBuildInformation())
# ustanawianie połączenia:
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM,0)
s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
s.bind(('192.168.0.2', 1234))
s.listen(0)
clientsocket, address = s.accept()
print(f"Connection from {address} has been established.")


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
autospeed = 40
turn = 40
spin = 45
arrows = [Key.up, Key.down, Key.left, Key.right]

behaviour_pipeline = ['left', 'right', 'straight', 'straight', 'left', 'right']
behaviour_counter = 0

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
    global behaviour_counter
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
            speed = autospeed
            behaviour_counter = 0
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

### DARKNET:
def convertBack(x, y, w, h):
    xmin = int(round(x - (w / 2)))
    xmax = int(round(x + (w / 2)))
    ymin = int(round(y - (h / 2)))
    ymax = int(round(y + (h / 2)))
    return xmin, ymin, xmax, ymax

def cvDrawBoxes(detections, img):
    for detection in detections:

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
                    (pt1[0], pt1[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.3,
                    [0, 255, 0], 2)
    return img


netMain = None
metaMain = None
altNames = None


# global metaMain, netMain, altNames
configPath = "./yolov3-custom_traffic1.2.cfg"
weightPath = "./yolov3-custom_traffic1.2.weights"
metaPath = "./traffic1.2.data"
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

ct_stop = CentroidTracker()
ct_traffic = CentroidTracker()
ct_person = CentroidTracker()

###

# capture = cv2.VideoCapture('http://192.168.0.11:8080/video')
# capture = cv2.VideoCapture('http://192.168.0.7:4747/video')
capture = WebcamVideoStream(src='http://192.168.0.7:4747/video').start()
# capture = WebcamVideoStream(src='http://192.168.0.7:8080/video').start()


def empty(x):
    pass

trackbars = np.zeros((300, 600, 3), np.uint8)
cv2.namedWindow('Trackbars')
cv2.createTrackbar('Kp', 'Trackbars', 50, 100, empty)
cv2.createTrackbar('Kd', 'Trackbars', 40, 100, empty)


cv2.namedWindow("preview")
cv2.moveWindow("pipeline", 600,600)
cv2.moveWindow("preview", 667,0)
cv2.moveWindow("Trackbars", 0, 0)


last_error = 0
last_total_error = 0
current_error = 0


executing_stop = False
executing_traffic_light = False
stop_timer = None
red_light = False
red_close = False
signs_executed = []
traffic_lights_executed = []
person_executed = []

cv2.namedWindow("preview", cv2.WINDOW_NORMAL)


# Graph data:
error_thresh = 15
x_val = list(range(-error_thresh, error_thresh + 1))
y_val = [0 for element in x_val]

# defaults.device = torch.device('cuda')

auto_start = False
road_environment = ''
last_pred_env = 'main_road'
last_road_environment = 'main_road'

recording = False

counter = 0
env_counter = 0
start_time = time.time()
while True:
    stop_rects = []
    traffic_rects = []
    person_rects = []
    red_light = False
    red_close = False
    counter += 1
    # succesfull, image = capture.read()
    image = capture.read()

    frame_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    frame_resized = cv2.resize(frame_rgb,
                               (darknet.network_width(netMain),
                                darknet.network_height(netMain)),
                               interpolation=cv2.INTER_LINEAR)

    darknet.copy_image_from_bytes(darknet_image, frame_resized.tobytes())

    detections = darknet.detect_image(netMain, metaMain, darknet_image, thresh=0.8)
    for detection in detections:
        # if detection[0].decode() == 'Green light' or detection[0].decode() == 'Red light':
        x, y, w, h = detection[2][0], \
                     detection[2][1], \
                     detection[2][2], \
                     detection[2][3]
        xmin, ymin, xmax, ymax = convertBack(
            float(x), float(y), float(w), float(h))
        pt1 = (xmin, ymin)
        pt2 = (xmax, ymax)
        if detection[0].decode() == 'Green light':
            cv2.rectangle(frame_resized, pt1, pt2, (0, 255, 0), 1)
        elif detection[0].decode() == 'Red light' or detection[0].decode() == "Stop sign":
            cv2.rectangle(frame_resized, pt1, pt2, (255, 0, 0), 1)
        else:
            cv2.rectangle(frame_resized, pt1, pt2, (0, 0, 255), 1)
        if detection[0].decode() == 'Red light':
            red_light = True
        else:
            red_light = False
        cv2.putText(frame_resized,
                    detection[0].decode() +
                    " [" + str(round(detection[1] * 100)) + "]",
                    (pt1[0], pt1[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.3,
                    [0, 255, 0] if detection[0].decode() == 'Green light' else [255, 0, 0], 1)
        if detection[0].decode() == 'Stop sign':
            stop_rects.append([xmin, ymin, xmax, ymax])
        elif detection[0].decode() == 'Red light':
            traffic_rects.append([xmin, ymin, xmax, ymax])
        elif detection[0].decode() == 'Person':
            person_rects.append([xmin, ymin, xmax, ymax])

    stop_signs = ct_stop.update(stop_rects)
    traffic_lights = ct_traffic.update(traffic_rects)
    person = ct_person.update(person_rects)


    # loop over the tracked objects to mark them: (not really necessary..)
    for (objectID, centroid) in stop_signs.items():
        # print(f'Wysokość znaku {objectID}: {centroid[2]}[BpFE]')
        text = f'Height: {centroid[2]}[BpFE]'
        # draw both the ID of the object and the centroid of the
        # object on the output frame
        # text = "ID {}".format(objectID)
        # cv2.putText(frame_resized, text, (centroid[0] - 10, centroid[1] - 10),
        #             cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        # cv2.circle(frame_resized, (centroid[0], centroid[1]), 4, (0, 255, 0), -1)

    image = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (frameWidth, frameHeight))

    kp = cv2.getTrackbarPos('Kp', 'Trackbars') / 25
    kd = cv2.getTrackbarPos('Kd', 'Trackbars') / 5

    pred_env = predictor.is_on_intersection(image)

    if pred_env == last_pred_env:
        env_counter += 1
    else:
        env_counter = 0
    if env_counter >= 5:
        road_environment = pred_env

    #
    if road_environment == 'intersection':
        if behaviour_pipeline[behaviour_counter] == 'left':
            error = predictor.left_predict(image)
        elif behaviour_pipeline[behaviour_counter] == 'straight':
            error = predictor.straight_predict(image)
        elif behaviour_pipeline[behaviour_counter] == 'right':
            error = predictor.right_predict(image)
    else:
        error = predictor.main_predict(image)

    if last_road_environment == 'intersection' and road_environment == 'main_road' and behaviour_counter < len(behaviour_pipeline) - 1:
        behaviour_counter += 1
    elif last_road_environment == 'intersection' and road_environment == 'main_road' and behaviour_counter >= len(behaviour_pipeline) - 1:
        behaviour_counter = 0

    last_pred_env = pred_env
    last_road_environment = road_environment
    # print(f'{env_counter}: {pred_env}, actual: {road_environment}')
    # print(f'{env_counter}: {pred_env}, actual: {road_environment}, next up: {behaviour_pipeline}')

    centertext = "Error = " + str(error)
    if len(behaviour_pipeline)>0:
        cv2.putText(image, f'{road_environment}  Next up:{behaviour_pipeline[behaviour_counter]}', (0, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    total_error = round((error * kp) + (kd * (error - last_error)))
    if abs(total_error) > 30:
        current_error = total_error
        autowheelturn = True
    # Wysyłanie do clienta:
    msg, diff = msgmaker.msgmaker1(total_error, speed)

    if auto:
        # red_light = False
        speed = autospeed
        cv2.putText(image, "Auto: ON ", (240, 470), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Traffic light detection:
        for (objectID, centroid) in traffic_lights.items():
            if centroid[2] >= 50:
                red_close = True
                print(f'stopped on a red light')
        # Stop sign detection:
        for (objectID, centroid) in stop_signs.items():
            if centroid[2] >= 70 and objectID not in signs_executed:
                executing_stop = True
                print(executing_stop)
                signs_executed.append(objectID)
        # Person detection:
        for (objectID, centroid) in person.items():
            if centroid[2] >= 50:
                red_close = True
                print(f'person detected')

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

        elif red_light and red_close:
            clientsocket.send(bytes('<110001000>' + "\n", "utf-8"))
        elif red_close:
            clientsocket.send(bytes('<110001000>' + "\n", "utf-8"))
            # print("person detected")
        else:
            clientsocket.send(bytes(msg + "\n", "utf-8"))
            # print('self-driving!')


        filename = datetime.datetime.now().time()
        if recording:
            print(f"recording")
            cv2.imwrite(f'/home/rafal/PycharmProjects/naukaOpenCV/stop_sign/{filename}.png',
                            image)

    last_error = error
    last_total_error = total_error
    # print(f'{behaviour_pipeline}, behaviour counter: {behaviour_counter}')


    cv2.namedWindow("preview", cv2.WINDOW_NORMAL)

    # imgStacked = utlis.stackImages(1, ([image, mo_cap_image]))

    cv2.imshow("preview", image)
    # cv2.imshow("preview", imgStacked)
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

finish_time = time.time()
fps = counter / (finish_time-start_time)
print('Frames per second: ' + str(fps))
print(f"Liczba klatek: {counter}")
cv2.destroyAllWindows()
# plt.bar(x_val, y_val)
# plt.show()

