import pygame
from Colors import *
from controller.Controller import Controller, decision_movement


class Robot:
    def __init__(self, start: tuple, cell_size, decisionMaker: Controller, vision=40, r=8, numsOfSteps=8):
        self.pos = start
        self.vision = vision
        self.r = r
        self.numsOfSteps = numsOfSteps
        self.currentStep = 0
        self.decisionMaker = decisionMaker
        self.cell_size = cell_size
        self.moveDirection = (0, 0)

    # Move the robot according to the direction vector
    def move(self) -> None:
        x, y = self.pos
        dx = self.moveDirection[0] * self.cell_size
        dy = self.moveDirection[1] * self.cell_size

        # Check if the decision is invalid (out of the boundary), if so, make a new decision
        if (x + dx < 30 or x + dx > 542 or y + dy < 30 or y + dy > 542) and self.currentStep == 0:
            # Set current step to 10 to make new decision
            self.currentStep = 10
            return

        # Move the robot one step towards the direction
        self.pos = (x + dx / self.numsOfSteps, y + dy / self.numsOfSteps)
        self.currentStep += 1

    def resetPosition(self, start: tuple) -> None:
        self.pos = start
        self.currentStep = 0
        self.moveDirection = (0, 0)

    def draw(self, window, draw_sr=True) -> None:
        if draw_sr:
            # draw the sensor range
            pygame.draw.circle(window,
                               RED,
                               self.pos,
                               self.vision,
                               1)
        # draw the robot
        pygame.draw.circle(window, RED, self.pos, self.r, 0)

    # Check if the robot has reached the goal
    def reach(self, goal, epsilon=16) -> bool:
        robotX, robotY = self.pos
        goalX, goalY = goal
        return ((robotX - goalX) ** 2 + (robotY - goalY) ** 2) <= epsilon ** 2

    # Get the closest dynamic obstacles within the vision range
    def detect(self, obstacles_list) -> list or None:
        # Get index of the closest dynamic obstacle
        index = -1
        min_distance = float('inf')
        for i, obstacle in enumerate(obstacles_list):
            if obstacle.static:
                continue

            x1, x2, y1, y2 = obstacle.return_coordinate()
            closest_x = max(x1, min(self.pos[0], x2))
            closest_y = max(y1, min(self.pos[1], y2))
            distance = (closest_x - self.pos[0]) ** 2 + (closest_y - self.pos[1]) ** 2
            if distance <= self.vision ** 2 and distance < min_distance:
                index = i
                min_distance = distance

        if index != -1:
            # Get the current and next position of the closest obstacle
            closest_obstacle = obstacles_list[index]
            current_position = (closest_obstacle.x, closest_obstacle.y)

            closest_obstacle.move()
            next_position = (closest_obstacle.x, closest_obstacle.y)
            closest_obstacle.undo_move()

            return [current_position, next_position]
        else:
            return None

    def getDistanceToClosestObstacle(self, obstacles_list) -> float:
        # Get distance of the closest obstacle
        min_distance = float('inf')
        for i, obstacle in enumerate(obstacles_list):
            x1, x2, y1, y2 = obstacle.return_coordinate()
            closest_x = max(x1, min(self.pos[0], x2))
            closest_y = max(y1, min(self.pos[1], y2))
            distance = (closest_x - self.pos[0]) ** 2 + (closest_y - self.pos[1]) ** 2
            min_distance = min(distance, min_distance)

        return min(self.vision, min_distance ** 0.5)

    # Check if the robot has collided with the obstacles
    def checkCollision(self, obstacles_list) -> bool:
        for obstacle in obstacles_list:
            x1, x2, y1, y2 = obstacle.return_coordinate()
            closest_x = max(x1, min(self.pos[0], x2))
            closest_y = max(y1, min(self.pos[1], y2))
            distance = (closest_x - self.pos[0]) ** 2 + (closest_y - self.pos[1]) ** 2
            if distance < self.r ** 2:
                self.setDiscount()
                return True

        return False

    def setDiscount(self) -> None:
        self.decisionMaker.setCollision()

    def nextPosition(self, goal) -> tuple:
        return goal

    def makeDecision(self, obstacles_list) -> None:
        # After the robot has moved for a certain number of steps, make a new decision and reset the step counter
        if self.currentStep >= self.numsOfSteps:
            obstacle = self.detect(obstacles_list)
            if obstacle is not None:
                decision = self.decisionMaker.makeObstacleDecision(self, obstacle)
            else:
                decision = self.decisionMaker.makeDecision(self)

            self.moveDirection = decision_movement[decision]

            self.currentStep = 0

    def updateQ(self) -> None:
        self.decisionMaker.updateAll(self)

    def outputPolicy(self, scenario, current_map, run_index) -> None:
        self.decisionMaker.outputPolicy(scenario, current_map, run_index)

    def resetController(self) -> None:
        self.decisionMaker.reset()
