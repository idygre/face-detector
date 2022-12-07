import cv2
face_cascade = cv2.CascadeClassifier("cascade_frontalface.xml")


class VideoCamera(object):

    def __init__(self):
        self.video = cv2.VideoCapture(0)

    def __del__(self):
        self.video.release()

    def get_frame(self):

        check, frame = self.video.read()

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray,
                                              scaleFactor=1.3,
                                              minNeighbors=5)

        for x, y, w, h in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 3)
            break

        # Convert to jpg
        check, jpeg = cv2.imencode('.jpg', frame)
        return jpeg.tobytes()
