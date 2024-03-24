import pygame
import numpy as np
from Colors import *


class Object:
    def __init__(self, x, y, width, height):
        self.x = x
        self.y = y
        self.width = width
        self.height = height

    def return_coordinate(self):  # return (top_left.x, bottom_right.x, top_left.y, bottom_right.y)
        return self.x - self.width / 2, self.x + self.width / 2, self.y - self.height / 2, self.y + self.height / 2

    def get_corners(self):
        return [(self.x - self.width / 2, self.y - self.height / 2),
                (self.x + self.width / 2, self.y - self.height / 2),
                (self.x - self.width / 2, self.y + self.height / 2),
                (self.x + self.width / 2, self.y + self.height / 2)]

    def get_area(self):
        return self.width * self.height

    def get_intersect_area(self, otherAABB):
        x1, x2, y1, y2 = self.return_coordinate()
        x_1, x_2, y_1, y_2 = otherAABB.return_coordinate()
        top_left_x = max(x1, x_1)
        top_left_y = max(y1, y_1)
        bottom_right_x = min(x2, x_2)
        bottom_right_y = min(y2, y_2)
        return max(0, bottom_right_x - top_left_x) * max(0, bottom_right_y - top_left_y)

    def get_intersect_percentage(self, otherObject):
        return self.get_intersect_area(otherObject) / self.get_area()

    def get_quadrant(self, point):
        point_x, point_y = point
        if point_x <= self.x:
            if point_y <= self.y:
                return self.NW
            else:
                return self.SW
        else:
            if point_y <= self.y:
                return self.NE
            else:
                return self.SE

    def draw(self, window):
        pygame.draw.rect(window, BLACK, (self.x - self.width / 2, self.y - self.height / 2, self.width, self.height), 1)
