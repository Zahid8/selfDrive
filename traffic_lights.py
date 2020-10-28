import numpy as np
import cv2
import socket

s = socket.socket(socket.AF_INET, socket.SOCK_STREAM,0)
s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
s.bind(('192.168.0.2', 1234))
s.listen(0)
clientsocket, address = s.accept()
print(f"Connection from {address} has been established.")


capture = cv2.VideoCapture('http://192.168.0.7:4747/video')
# capture = WebcamVideoStream(src='http://192.168.0.07:4747/video').start()


while True:
    succesfull, image = capture.read()
    hsv = int(cv2.cvtColor(image, cv2.COLOR_BGR2HSV)[..., 2].mean())
    print(hsv)

    cv2.imshow("preview", image)
    # cv2.imshow("preview2", image2)
    # cv2.imshow("usb_cam", mo_cap_image)
    # time.sleep(0.05)
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
      