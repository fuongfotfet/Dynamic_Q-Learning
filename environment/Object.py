import pygame
from Colors import *


class Object:
    def __init__(self, x, y, width, height):
        self.x = x
        self.y = y
        self.width = width
        self.height = height

    def return_coordinate(self) -> tuple:  # return (top_left.x, bottom_right.x, top_left.y, bottom_right.y)
        return self.x - self.width / 2, self.x + self.width / 2, self.y - self.height / 2, self.y + self.height / 2

    def draw(self, window) -> None:
        pygame.draw.rect(window, BLACK, (self.x - self.width / 2, self.y - self.height / 2, self.width, self.height), 1)
