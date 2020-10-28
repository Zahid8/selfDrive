import socket
import cv2
import time
from imutils.video import VideoStream

cameraFeed = True
cameraNo = 1
# frameWidth = 720
# frameHeight = 480


# s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
# s.bind((socket.gethostname(), 1234))
# s.listen(0)
# clientsocket, address = s.accept()
# print(f"Connection from {address} has been established.")

# capture = cv2.VideoCapture('http://192.168.0.11:8080/video')
# capture = cv2.VideoCapture('http://192.168.0.3:4747/video')
capture = cv2.VideoCapture(0)
# capture = VideoStream(src=0).start()



# capture.set(3, frameWidth)
# capture.set(4, frameHeight)
# capture.set(10, 20) #set brightness
# capture.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
# print(capture.get(cv2.CAP_PROP_FPS))

# fourcc = cv2.VideoWriter_fourcc(*'H264')  # DZIAŁA: XVID, DIVX, H264
# out = cv2.VideoWriter(filename, fourcc, 28, (frameHeight, frameWidth))

successful = True

start_time = time.time()
counter = 0

cv2.namedWindow("preview", cv2.WINDOW_NORMAL)
while successful:
    counter += 1
    # print(capture.get(cv2.CAP_PROP_FPS))
    # print(capture.get(cv2.VIDEOWRITER_PROP_FRAMEBYTES))

    successful, image = capture.read()
    # image = capture.read()

    # print(capture.get(cv2.CAP_PROP_FPS))
    # print(capture.get(cv2.VIDEOWRITER_PROP_FRAMEBYTES))

    cv2.imshow("preview", image)

    # KONIEC:
    keys = cv2.waitKey(1) & 0xFF
    if keys == ord("q"):
        # msg = '<110001000>'
        # clientsocket.send(bytes(msg + "\n", "utf-8"))
        # clientsocket.send(bytes('<200>' + "\n", "utf-8"))
        # msg = 'koniec'
        # clientsocket.send(bytes(msg + "\n", "utf-8"))
        # clientsocket.close()
        # s.close()
        break
finish_time = time.time()
fps = counter / (finish_time-start_time)
print('Frames per second: ' + str(fps))
capture.release()
# out.release()
cv2.destroyAllWindows()
