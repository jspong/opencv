import functools
import numpy as np
import cv2 as cv
import os
import string
import sys
import unittest
import event_loop

def takesOdd(fn):
    @functools.wraps(fn)
    def wrapper(i):
        return fn(makeOdd(i))
    return wrapper

def makeOdd(i):
    if i % 2 == 0:
        i += 1
    if i < 3:
        i = 3
    return i

def convolution(i, fn):
    return np.array([[fn(x, y, i // 2) for x in range(i)] for y in range(i)])

def toList(arr):
    return [[x for x in row] for row in arr]

@takesOdd
def nothing(i):
    def fn(x, y, mid):
        return int(x == mid and y == mid)
    return convolution(i, fn)

@takesOdd
def edgeDetect(i):
    def fn(x, y, mid):
        if x == mid and y == mid:
            return i ** 2 - 1
        else:
            return -1
    return convolution(i, fn)

@takesOdd
def blur(i):
    def fn(*args):
        return 1 / i / i
    return convolution(i, fn)

@takesOdd
def sharpen(i):
    def fn(x, y, mid):
        if x == mid and y == mid:
            curr = 1
            for y in range(i // 2):
                curr += 4 * (y+1) * (i // 2 - y)
            return curr
        else:
            if abs(x - mid) + abs(y - mid) > mid:
                return 0
            else:
                return abs(x-mid) + abs(y-mid) - mid - 1

    return convolution(i, fn)

def horizontalEdge(_):
    return np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])

def verticalEdge(_):
    return np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])

class Tests(unittest.TestCase):

    def testEdgeDetect(self):
        self.assertEqual(toList(edgeDetect(3)), [[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]])

    def testBlur(self):
        self.assertEqual(toList(blur(3)), [[1/9 for _ in range(3)] for _ in range(3)])

    def testSharpen(self):
        self.assertEqual(toList(sharpen(3)), [[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
        self.assertEqual(toList(sharpen(5)), [[0, 0, -1, 0, 0], [0, -1, -2, -1, 0], [-1, -2, 17, -2, -1], [0, -1, -2, -1, 0], [0, 0, -1, 0, 0]])
        self.assertEqual(toList(sharpen(7)), [
                        [ 0,  0,  0, -1,  0,  0,  0],
                        [ 0,  0, -1, -2, -1,  0,  0],
                        [ 0, -1, -2, -3, -2, -1,  0],
                        [-1, -2, -3, 41, -3, -2, -1],
                        [ 0, -1, -2, -3, -2, -1,  0],
                        [ 0,  0, -1, -2, -1,  0,  0],
                        [ 0,  0,  0, -1,  0,  0,  0]])


class Playground(event_loop.EventLoop):
    def __init__(self, camera=2):
        super().__init__(camera)

        self.size = 3
        self.transforms = (nothing, edgeDetect, sharpen, blur, horizontalEdge, verticalEdge)
        self.current_transform = 0

    def step(self, frame):
        cv.imshow('original', frame)

        transformFunction = self.transforms[self.current_transform]
        kernel = transformFunction(self.size)
        frame = cv.filter2D(src=frame, ddepth=-1, kernel=kernel)

        cv.putText(frame, '{}({})'.format(transformFunction.__name__, makeOdd(self.size)), (50, 50), cv.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 3, cv.LINE_AA)

        cv.imshow('transformed', frame)

        if self.keyPress is not None:
            if self.keyPress in string.digits:
                self.size = int(self.keyPress)
            elif self.keyPress == '+':
                self.size += 1
            elif self.keyPress == '-':
                self.size -= 1
            elif self.keyPress == ' ':
                self.current_transform = (self.current_transform + 1) % len(self.transforms)
            elif self.keyPress == '\r':
                self.current_transform = (self.current_transform + len(self.transforms) - 1) % len(self.transforms)

if __name__ == '__main__':
    if os.getenv('RUN_TESTS'):
        unittest.main()
    else:
        loop = Playground()
        loop.start()
