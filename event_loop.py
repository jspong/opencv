import cv2 as cv
import string
import threading

_PRINTABLE = tuple(ord(x) for x in string.printable)

class EventLoop:

    def __init__(self, camera=2):
        self._camera = camera
        self._message = ""
        self._started = False
        self._frameCount = 0
        self.keyPress = None

    def start(self):
        if self._started:
            raise AssertionError("already started")
        self._cap = cv.VideoCapture(self._camera)

        while self._cap.isOpened():
            success, frame = self._cap.read()
            if not success:
                raise Exception("Failed to read from camera {}".format(self._camera))

            self._get_input()
            if self.keyPress == 'q':
                break
            self.step(frame)

        self._cap.release()
        cv.destroyAllWindows()

    def _get_input(self):
        key = cv.waitKey(10)
        if key in _PRINTABLE:
            self.keyPress = chr(key)
        else:
            self.keyPress = None

    def step(self, frame):
        pass
