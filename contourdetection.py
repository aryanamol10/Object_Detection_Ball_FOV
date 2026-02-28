import cv2
import numpy as np

cap = cv2.VideoCapture(2, cv2.CAP_DSHOW)


#def graph_mapped_function():
    #Still need to figure out
     
while True:
    ret, video = cap.read()
    if not ret:
        break

# hsv mask
    blurred = cv2.GaussianBlur(video, (11, 11), 0)
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

# colors
    lower_yellow = np.array([20, 100, 100], dtype=np.uint8)
    upper_yellow = np.array([40, 255, 255], dtype=np.uint8)

# mask creation and morphology
    mask = cv2.inRange(hsv, lower_yellow, upper_yellow)
    kernel=cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask = cv2.dilate(mask, None, iterations=1)

    contours, _ = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        cv2.drawContours(video, contours, -1, (255,255,255), thickness=-1)
        for contour in contours:
            hull = cv2.convexHull(contour, returnPoints=True)
            perimeter = cv2.arcLength(hull, True)
            epsilon = 0.1 * perimeter  # Adjust the 0.1 factor as needed
            approx_hull = cv2.approxPolyDP(hull, epsilon, True)
            cv2.drawContours(video, [hull], -1, (0, 255, 0), 3)


    cv2.imshow("video",video)
    if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    

cap.release()
cv2.destroyAllWindows()