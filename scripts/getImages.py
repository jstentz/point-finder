import cv2

# url = "http://172.16.0.7:4747/video"
url = "http://192.168.1.148:4747/video"
cap = cv2.VideoCapture(url)

num = 0

while cap.isOpened():

    succes, img = cap.read()

    k = cv2.waitKey(5)

    if k == 27:
        break
    elif k == ord('s'): # wait for 's' key to save and exit
        cv2.imwrite('../calibration-images2/img' + str(num) + '.png', img)
        print(f"image {num} saved!")
        num += 1

    cv2.imshow('Img',img)

# Release and destroy all windows before termination
cap.release()

cv2.destroyAllWindows()