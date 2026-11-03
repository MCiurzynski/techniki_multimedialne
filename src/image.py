import cv2
import numpy as np

class Image:

    def __init__(self, path):
        self.image = cv2.imread(path)
        self.path = path

    def histogram(self):
        return

    def compare_histogram(self, image: Image):
        histogram = self.histogram()
        other_histogram = image.histogram()
        