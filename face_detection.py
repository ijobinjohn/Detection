import cv2
cap = cv2.VideoCapture(0)
face_cascade=cv2.CascadeClassifier('haarcascade_frontalcatface.xml')

while True:
    ret, frame = cap.read()
    gray =cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray2 = np.array(gray, dtype='uint8')

    faces = face_cascade.detectMultiScale(gray2, 1.3, 5,0 )
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    cv2.imshow('Input', frame)
    c = cv2.waitKey(1)

cap.release()
cv2.destroyAllWindows()