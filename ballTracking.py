# import the necessary packages
from collections import deque
from imutils.video import VideoStream
import numpy as np
import cv2
import imutils
import time
import msgmaker
import socket
from math import sin, cos, pi


# ustanawianie połączenia:
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM,0)
s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
s.bind(('192.168.0.2', 1234))
s.listen(0)
clientsocket, address = s.accept()
print(f"Connection from {address} has been established.")

# define the lower and upper boundaries of the "green"
# ball in the HSV color space, then initialize the
# list of tracked points
# greenLower = (46, 100, 0)
# greenUpper = (60, 255, 255)
greenLower = (35, 100, 100)
greenUpper = (150, 255, 255)

blueLower = (100, 86, 0)
blueUpper = (255, 255, 255)


pinkLower = (145, 45, 100)
pinkUpper = (175, 255, 255)

purpleLower = (125, 50, 95)
purpleUpper = (150, 255, 255)

# darkgreenLower = (79, 124, 70) #Kiedy jasno
# darkgreenUpper = (109, 255, 255)

darkgreenLower = (42, 60, 70) # Kiedy ciemno
darkgreenUpper = (84, 255, 255)

orangeLower = (100, 86, 0)
orangeUpper = (255, 255, 255)


buffer = 64
pts = deque(maxlen=buffer)

# vs = VideoStream(src='http://192.168.0.11:4747/video').start()
# mo_cap = cv2.VideoCapture('http://192.168.0.3:4747/video')
# mo_cap = VideoStream(src=0).start()
mo_cap = cv2.VideoCapture(0)
filename = 'zdjęcie1.jpg'

frameWidth = 640
frameHeight = 480

# cv2.namedWindow("Frame", cv2.WINDOW_NORMAL)
def empty(x):
    pass
trackbars = np.zeros((300, 600, 3), np.uint8)
cv2.namedWindow('Trackbars')
cv2.createTrackbar('Kp', 'Trackbars', 40, 100, empty)
cv2.createTrackbar('Kd', 'Trackbars', 80, 100, empty)
cv2.createTrackbar('Ka', 'Trackbars', 0, 100, empty)
cv2.createTrackbar('ROI_X', 'Trackbars', 200, int(frameWidth/2), empty)
cv2.createTrackbar('ROI_Y', 'Trackbars', 100, int(frameHeight/2), empty)
cv2.createTrackbar('ROI_X_OFFSET', 'Trackbars', 320, frameWidth, empty)
cv2.createTrackbar('ROI_Y_OFFSET', 'Trackbars', 350, frameHeight, empty)
cv2.createTrackbar('Erode', 'Trackbars', 3, 20, empty)
cv2.createTrackbar('Dilate', 'Trackbars', 1, 20, empty)
cv2.createTrackbar('All', 'Trackbars', 1, 1, empty)
cv2.createTrackbar('Black_thresh', 'Trackbars', 30, 200, empty)

speed = 60
deviation = 0
last_deviation = 0
# allow the camera or video file to warm up
time.sleep(1)
# keep looping
counter = 0
start_time = time.time()
yellow_point = 0

while True:
    counter += 1
    route = cv2.imread('mapa3.png')
    route_green = route.copy()
    grab, mo_cap_image = mo_cap.read()
    # mo_cap_image = mo_cap.read()
    result = cv2.add(mo_cap_image, route)

    blurred = cv2.GaussianBlur(mo_cap_image, (11, 11), 0)
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
    
    erode = cv2.getTrackbarPos('Erode', 'Trackbars')
    dilate = cv2.getTrackbarPos('Dilate', 'Trackbars')

    # green mask:
    # mask = cv2.inRange(hsv, greenLower, greenUpper)
    # mask = cv2.erode(mask, None, iterations=erode)
    # mask = cv2.dilate(mask, None, iterations=dilate)

    # dark green mask:
    darkgreenmask = cv2.erode(hsv, None, iterations=erode)
    darkgreenmask = cv2.dilate(hsv, None, iterations=dilate)
    darkgreenmask = cv2.inRange(darkgreenmask, darkgreenLower, darkgreenUpper)


    # pink mask:
    pinkmask = cv2.erode(hsv, None, iterations=erode)
    pinkmask = cv2.dilate(hsv, None, iterations=dilate)
    pinkmask = cv2.inRange(pinkmask, pinkLower, pinkUpper)

    # # orange mask:
    # orangemask = cv2.inRange(hsv, orangeLower, orangeUpper)
    # orangemask = cv2.erode(orangemask, None, iterations=erode)
    # orangemask = cv2.dilate(orangemask, None, iterations=dilate)
    #
    # # purple mask:
    # purplemask = cv2.inRange(hsv, purpleLower, purpleUpper)
    # purplemask = cv2.erode(purplemask, None, iterations=erode)
    # purplemask = cv2.dilate(purplemask, None, iterations=dilate)

    # BLUE TRACK / ROUTE:
    hsvBlue = cv2.cvtColor(route.copy(), cv2.COLOR_BGR2HSV)
    maskBlue = cv2.inRange(hsvBlue, blueLower, blueUpper)
    contours_blk, hierarchy_blk = cv2.findContours(maskBlue, cv2.RETR_EXTERNAL,
                                                   cv2.CHAIN_APPROX_SIMPLE)

    if len(maskBlue) > 0:
        cv2.drawContours(route, contours_blk, -1, (255, 0, 0), 1)

    # find contours in the mask and initialize the current
    # (x, y) center of the ball
    darkgreen_cnts = cv2.findContours(darkgreenmask, cv2.RETR_EXTERNAL,
                            cv2.CHAIN_APPROX_SIMPLE)
    darkgreen_cnts = imutils.grab_contours(darkgreen_cnts)
    center1 = None
    pink_cnts = cv2.findContours(pinkmask, cv2.RETR_EXTERNAL,
                                      cv2.CHAIN_APPROX_SIMPLE)
    pink_cnts = imutils.grab_contours(pink_cnts)
    center2 = None
    # only proceed if at least one contour was found
    if len(darkgreen_cnts) > 0:
        # find the largest contour in the mask, then use
        # it to compute the minimum enclosing circle and
        # centroid
        try:
            c1 = max(darkgreen_cnts, key=cv2.contourArea)
            ((x1, y1), radius1) = cv2.minEnclosingCircle(c1)
            M = cv2.moments(c1)
            center1 = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))

        except ZeroDivisionError:
            pass

        # only proceed if the radius meets a minimum size
        if radius1 > 5:
            # draw the circle and centroid on the frame,
            # then update the list of tracked points
            # cv2.circle(result, (int(x1), int(y1)), int(radius1), (0, 0, 255), 1)
            # cv2.circle(path, (int(x), int(y)), int(radius),(0, 0, 255), 1)
            # cv2.circle(result, center1, 5, (0, 0, 255), -1)
            cv2.circle(route, center1, 1, (255, 0, 0), -1)

            # # update the points queue
            # pts.appendleft(center)
    # if len(pink_cnts) > 0:
    #     # find the largest contour in the mask, then use
    #     # it to compute the minimum enclosing circle and
    #     # centroid
    #     try:
    #         c2 = max(pink_cnts, key=cv2.contourArea)
    #         ((x2, y2), radius2) = cv2.minEnclosingCircle(c2)
    #         M = cv2.moments(c2)
    #         center2 = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
    #     except ZeroDivisionError:
    #         pass
    #     # only proceed if the radius meets a minimum size
    #     if radius2 > 5:
    #         # draw the circle and centroid on the frame,
    #         # then update the list of tracked points
    #         # cv2.circle(result, (int(x2), int(y2)), int(radius2), (0, 0, 255), 1)
    #         # cv2.circle(path, (int(x), int(y)), int(radius),(0, 0, 255), 1)
    #         # cv2.circle(result, center2, 5, (0, 0, 255), -1)
    #         cv2.circle(route, center2, 1, (255,20,147), -1)
    #         # # update the points queue
    #         # pts.appendleft(center)

    # if center1 and center2:
    #     cv2.line(route, center1, center2, (0, 250, 0), 3, cv2.LINE_AA)
    #
    #     # hsvGreen = cv2.cvtColor(route.copy(), cv2.COLOR_BGR2HSV)
    #     mask = cv2.inRange(route, (0,245,0), (0,255,0))
    #     contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    #     cv2.drawContours(mask, contours, -1, (255, 0, 0), 1)
    #     if len(contours) > 0:
    #         (x1, y1), (MA1, ma1), angle = cv2.fitEllipse(contours[0])
    #
    #         length = 70
    #         angle = angle - 90
    #
    #         if center1[0] == center2[0] and center1[1] > center2[1]:
    #             length = -length
    #         if center1[0] < center2[0]:
    #             length = -length
    #
    #         x2 = int(center1[0] + length * cos(angle * pi / 180.0))
    #         y2 = int(center1[1] + length * sin(angle * pi / 180.0))
    #
    #         yellow_point = (x2, y2)
    #
    #         cv2.circle(result, (x2, y2), 1, (0, 255, 255), thickness=-2, lineType=8, shift=0)
    #         cv2.circle(route, (x2, y2), 1, (0, 255, 255), thickness=-2, lineType=8, shift=0)

        deviation = (cv2.pointPolygonTest(contours_blk[0], center1, True))
        deviation = -deviation

        error_thresh = 80

        kp = cv2.getTrackbarPos('Kp', 'Trackbars') / 10
        kd = cv2.getTrackbarPos('Kd', 'Trackbars') / 5

        # print(kp)

        total_error = round((deviation * kp) + (kd * (deviation - last_deviation)))
        # print(total_error)
        if total_error > error_thresh:
            total_error = error_thresh
        if total_error < -error_thresh:
            total_error = -error_thresh
        centertext = "Total Error = " + str(total_error)
        cv2.putText(route, centertext, (0, 470), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

        if abs(total_error) > 30:
            current_error = total_error
            autowheelturn = True
        # cv2.putText(image, error_data, (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

        # speed = autospeed
        # cv2.putText(route, "Auto: ON ", (240, 470), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        clientsocket.send(bytes(msgmaker.msgmaker3(total_error, speed) + "\n", "utf-8"))
        # print(msgmaker.msgmaker3(total_error, speed))

        last_deviation = deviation

    # cv2.imshow("DarkGreenMask", darkgreenmask)
    # cv2.imshow("PinkMask", pinkmask)
    cv2.imshow("Result", result)
    cv2.imshow("Route", route)
    # cv2.imshow("Blured", blurred)
    # cv2.imshow("Mask", mask)
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        msg = '<110001000>'
        clientsocket.send(bytes(msg + "\n", "utf-8"))
        clientsocket.send(bytes('<200>' + "\n", "utf-8"))
        msg = 'koniec'
        clientsocket.send(bytes(msg + "\n", "utf-8"))
        clientsocket.close()
        s.close()
        break
finish_time = time.time()
fps = counter / (finish_time - start_time)
print('Frames per second: ' + str(fps))
print(f'Liczba klatek: {counter}')
mo_cap.release()
# close all windows
cv2.destroyAllWindows()