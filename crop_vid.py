import numpy as np
import cv2

cap = cv2.VideoCapture('/home/justin/Downloads/EPM_10.mp4')
output = cv2.VideoWriter(
    "output.mp4", cv2.VideoWriter_fourcc(*'mp4v'),
    25, (1280, 960))
i = 0
while(cap.isOpened()):
    i += 1
    ret, frame = cap.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    cv2.imshow('frame',frame)
    if i > 80:
        output.write(frame)
        print("cap")
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
output.release()
cap.release()
cv2.destroyAllWindows()