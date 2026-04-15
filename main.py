import cv2
import numpy as np

# Minimum size for detection
largura_min = 80
altura_min = 80

# Line position
pos_linha = 550
offset = 6

# Variables
detec = []
carros = 0

# Function to get center
def pega_centro(x, y, w, h):
    cx = int(x + w / 2)
    cy = int(y + h / 2)
    return cx, cy

# Video
cap = cv2.VideoCapture('video.mp4')

# Better background subtractor
subtracao = cv2.createBackgroundSubtractorMOG2(history=100, varThreshold=40)

while True:
    ret, frame1 = cap.read()
    if not ret:
        break

    # Preprocessing
    grey = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(grey, (5, 5), 0)

    img_sub = subtracao.apply(blur)

    # Morphological operations
    dilat = cv2.dilate(img_sub, np.ones((5, 5)))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    dilatada = cv2.morphologyEx(dilat, cv2.MORPH_CLOSE, kernel)
    dilatada = cv2.morphologyEx(dilatada, cv2.MORPH_CLOSE, kernel)

    # Find contours
    contornos, _ = cv2.findContours(dilatada, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Draw counting line
    cv2.line(frame1, (25, pos_linha), (1200, pos_linha), (255, 127, 0), 3)

    # Loop through contours
    for (i, c) in enumerate(contornos):
        (x, y, w, h) = cv2.boundingRect(c)

        # Filter small objects
        if w < largura_min or h < altura_min:
            continue

        # Draw rectangle
        cv2.rectangle(frame1, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Get center
        centro = pega_centro(x, y, w, h)
        detec.append(centro)

        # Draw center
        cv2.circle(frame1, centro, 4, (0, 0, 255), -1)

    # Check crossing line
    for (cx, cy) in detec:
        if (pos_linha - offset) < cy < (pos_linha + offset):
            carros += 1
            cv2.line(frame1, (25, pos_linha), (1200, pos_linha), (0, 255, 0), 3)
            detec.remove((cx, cy))
            print("Car Count:", carros)

    # Display count
    cv2.putText(frame1, f"VEHICLE COUNT: {carros}", (400, 70),
                cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 5)

    # Show frames
    cv2.imshow("Original", frame1)
    # cv2.imshow("Detection", dilatada)

    # Exit
    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()