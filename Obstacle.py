from Object import Object
import numpy as np
from Colors import *
import pygame


class Obstacle(Object):
    def __init__(self, x, y, width, height, static, v, x_bound=(30, 30), y_bound=(100, 100), followPath=False, path=None):
        super().__init__(x, y, width, height)
        self.static = static
        self.v = np.array(v)
        if self.static:
            self.v = np.array([0, 0])
        self.x_bound = (x - x_bound[0], x + x_bound[1])
        self.y_bound = (y - y_bound[0], y + y_bound[1])
        self.original_x = x if not followPath else path[0][0]
        self.original_y = y if not followPath else path[0][1]
        self.counter = 1
        self.history = []
        self.followPath = followPath
        self.path = path

    def draw(self, window, with_past=True) -> None:
        if with_past:
            for pos_x, pos_y in self.history[-5:]:
                pygame.draw.rect(window, GREY,
                                 (pos_x - self.width / 2, pos_y - self.height / 2, self.width, self.height))
        color = BLACK if self.static else CYAN
        pygame.draw.rect(window, color, (self.x - self.width / 2, self.y - self.height / 2, self.width, self.height))

    # Move the obstacle within the boundaries
    def move(self) -> None:
        self.history.append((self.x, self.y))

        # If the obstacle is following a path, move to the next point in the path
        # If the obstacle is not following a path, oscillate within the boundaries
        if self.followPath:
            if self.counter < len(self.path):
                # Calculate and normalize the direction vector
                v = (self.path[self.counter][0] - self.x, self.path[self.counter][1] - self.y)
                v_x, v_y = [float(i) / np.linalg.norm(v) for i in v]

                self.x += v_x * self.v[0]
                self.y += v_y * self.v[0]

                # If the obstacle is close enough to the next point in the path, move to the next point
                if abs(self.x - self.path[self.counter][0]) < 1 and abs(self.y - self.path[self.counter][1]) < 1:
                    self.counter += 1
        else:
            if not self.static:
                v_x, v_y = self.v
                self.x += v_x
                if self.x < self.x_bound[0]:
                    self.x = self.x_bound[0]
                    self.v = (-v_x, v_y)
                elif self.x > self.x_bound[1]:
                    self.x = self.x_bound[1]
                    self.v = (-v_x, v_y)
                self.y += v_y
                if self.y < self.y_bound[0]:
                    self.y = self.y_bound[0]
                    self.v = (v_x, -v_y)
                elif self.y > self.y_bound[1]:
                    self.y = self.y_bound[1]
                    self.v = (v_x, -v_y)

    def reset(self) -> None:
        if self.followPath:
            self.x, self.y = self.path[0]
            self.counter = 1

    def undo_move(self) -> None:
        if len(self.history) > 0:
            self.x, self.y = self.history.pop()

    def __str__(self) -> str:
        return f"Obstacle({self.x}, {self.y}, {self.width}, {self.height}, {self.static}, [{self.v[0]}, {self.v[1]}])"
