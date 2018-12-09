import cv2

face_cascade = cv2.CascadeClassifier('opencv/haarcascade_frontalcatface.xml')
face_cascade_alt = cv2.CascadeClassifier('opencv/haarcascade_frontalface_alt.xml')
eye_cascade = cv2.CascadeClassifier('opencv/haarcascade_eye.xml')

def detect_faces(f_cascade, colored_img, scaleFactor = 1.2):
    img_copy = colored_img.copy()
    gray = cv2.cvtColor(img_copy, cv2.COLOR_BGR2GRAY)
    faces = f_cascade.detectMultiScale(gray, scaleFactor=scaleFactor, minNeighbors=5);
    for (x, y, w, h) in faces:
        cv2.rectangle(img_copy, (x, y), (x+w, y+h), (0, 255, 0), 2)
    return img_copy

class VideoCamera(object):
    def __init__(self):
        self.video = cv2.VideoCapture(0)

    def __del__(self):
        self.video.release()

    def get_frame(self):
        success, image = self.video.read()
        #detectar caras
        ret, jpeg = cv2.imencode('.jpg', detect_faces(face_cascade_alt, image))
        # reconocer ojos
        # ret, jpeg = cv2.imencode('.jpg', detect_faces(eye_cascade, image))

        return jpeg.tobytes()
