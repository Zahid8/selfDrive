
import cv2



def callback(value):
    pass


def setup_trackbars(range_filter):
    cv2.namedWindow("Trackbars", 0)

    for i in ["MIN", "MAX"]:
        v = 0 if i == "MIN" else 255

        for j in range_filter:
            cv2.createTrackbar("%s_%s" % (j, i), "Trackbars", v, 255, callback)


def get_trackbar_values(range_filter):
    values = []

    for i in ["MIN", "MAX"]:
        for j in range_filter:
            v = cv2.getTrackbarPos("%s_%s" % (j, i), "Trackbars")
            values.append(v)

    return values


range_filter = 'HSV'.upper()

# camera = cv2.VideoCapture('http://192.168.0.3:4747/video')
camera = cv2.VideoCapture(0)
# camera = cv2.imread(0)

setup_trackbars(range_filter)

while True:

    ret, image = camera.read()
    # image = cv2.imread('mapa1.png')

    blurred = cv2.GaussianBlur(image, (11, 11), 0)
    frame_to_thresh = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

    # mask = cv2.erode(frame_to_thresh, None, iterations=3)
    # mask = cv2.dilate(frame_to_thresh, None, iterations=0)

    v1_min, v2_min, v3_min, v1_max, v2_max, v3_max = get_trackbar_values(range_filter)


    thresh = cv2.inRange(frame_to_thresh, (v1_min, v2_min, v3_min), (v1_max, v2_max, v3_max))


    cv2.imshow("Original", image)
    cv2.imshow("Thresh", thresh)

    if cv2.waitKey(1) & 0xFF is ord('q'):
        break


