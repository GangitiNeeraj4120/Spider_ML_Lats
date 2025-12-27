import cv2
import numpy as np

cap = cv2.VideoCapture(0)

trajectory = []

bg_sub = cv2.createBackgroundSubtractorMOG2(
    history=200,
    varThreshold=25,
    detectShadows=False
)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    fg_mask = bg_sub.apply(frame)
    fg_mask = cv2.medianBlur(fg_mask, 5)

    edge = cv2.Canny(fg_mask, 50, 50)

    kernel = np.ones((5, 5), np.uint8)

    dilated = cv2.dilate(edge, kernel, iterations=2)
    closed = cv2.morphologyEx(edge, cv2.MORPH_CLOSE, kernel, iterations=2)
    processed = cv2.erode(closed, kernel, iterations=1)

    contours, _ = cv2.findContours(
        closed, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
    )

    valid_contours = [
        c for c in contours if 1000 < cv2.contourArea(c) < 50000
    ]

    if valid_contours:
        merged_cnt = max(valid_contours, key=cv2.contourArea)
        
        x, y, w, h = cv2.boundingRect(merged_cnt)

        prev_box = (x, y, w, h)

        if prev_box is not None:
            px, py, pw, ph = prev_box
            if abs(x - px) > 40 or abs(y - py) > 40:
                continue
            
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

        hull = cv2.convexHull(merged_cnt)
        cv2.drawContours(frame, [hull], -1, (255, 0, 0), 2)

        M = cv2.moments(merged_cnt)
        if M["m00"] != 0:
            cx = int(M["m10"]/M["m00"])
            cy = int(M["m01"]/M["m00"])

            trajectory.append((cx, cy))

    cv2.imshow("Object Boundary Tracking", frame)
    cv2.imshow("Edges", edge)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()