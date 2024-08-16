import numpy as np
import cv2

cap = cv2.VideoCapture('/home/justin/Downloads/mice_2_1.mp4')
output = cv2.VideoWriter(
    "output.mp4", cv2.VideoWriter_fourcc(*'mp4v'),
    50, (984, 556))
i = 0
while(cap.isOpened()):
    i += 1
    ret, frame = cap.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    cv2.imshow('frame',frame)
    if i > 100:
        output.write(frame)
        print("cap")
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
output.release()
cap.release()
cv2.destroyAllWindows()