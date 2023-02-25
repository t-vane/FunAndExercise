import os
os.environ["OPENBLAS_NUM_THREADS"] = "1"
import math
import random

import robot_cleaner_visualize
import pylab

import numpy as np

from robot_cleaner_verify_movement38 import testRobotMovement


#================================
# Cleaning room with robot
#================================


# Class Position
class Position(object):
    """
    A Position represents a location in a two-dimensional room.
    """
    def __init__(self, x, y):
        """
        Initializes a position with coordinates (x, y).
        """
        self.x = x
        self.y = y
        
    def getX(self):
        return self.x
    
    def getY(self):
        return self.y
    
    def getNewPosition(self, angle, speed):
        """
        Computes and returns the new Position after a single clock-tick has
        passed, with this object as the current position, and with the
        specified angle and speed.

        Does not test whether the returned position fits inside the room.

        angle: number representing angle in degrees, 0 <= angle < 360
        speed: positive float representing speed

        Returns: a Position object representing the new position.
        """
        old_x, old_y = self.getX(), self.getY()
        angle = float(angle)
        # Compute the change in position
        delta_y = speed * math.cos(math.radians(angle))
        delta_x = speed * math.sin(math.radians(angle))
        # Add that to the existing position
        new_x = old_x + delta_x
        new_y = old_y + delta_y
        return Position(new_x, new_y)

    def __str__(self):  
        return "(%0.2f, %0.2f)" % (self.x, self.y)


# Class RectangularRoom
class RectangularRoom(object):
    """
    A RectangularRoom represents a rectangular region containing clean or dirty
    tiles.

    A room has a width and a height and contains (width * height) tiles. At any
    particular time, each of these tiles is either clean or dirty.
    """
    def __init__(self, width, height):
        """
        Initializes a rectangular room with the specified width and height.

        Initially, no tiles in the room have been cleaned.

        width: an integer > 0
        height: an integer > 0
        """
        self.width = width
        self.height = height
        self.tiles = np.zeros((height, width), dtype=int)


    def cleanTileAtPosition(self, pos):
        """
        Marks the tile under the position POS as cleaned.
        1 is cleaned. 0 is not cleaned. 
        Origin is top left of tiles grid.

        Assumes that pos represents a valid position inside this room.

        pos: a Position
        """
        self.tiles[int(pos.y)][int(pos.x)] = 1

        
    def isTileCleaned(self, m, n):
        """
        Returns True if the tile (m, n) has been cleaned.

        Assumes that (m, n) represents a valid tile inside the room.

        m: an integer
        n: an integer
        returns: True if (m, n) is cleaned, False otherwise
        """
        return(bool(self.tiles[n][m]))


    def getNumTiles(self):
        """
        Returns the total number of tiles in the room.

        returns: an integer
        """
        return self.width * self.height
        

    def getNumCleanedTiles(self):
        """
        Returns the total number of clean tiles in the room.

        returns: an integer
        """
        count=0
        for i in range(self.height):
            for j in range(self.width):
                if self.tiles[i][j] == 1:
                    count += 1
        return count


    def getRandomPosition(self):
        """
        Returns a random position inside the room.

        returns: a Position object.
        """
        x = random.uniform(0, self.width)
        y = random.uniform(0, self.height)
        rand_pos = Position(x, y)
        return rand_pos
        

    def isPositionInRoom(self, pos):
        """
        Returns True if POS is inside the room.

        pos: a Position object.
        returns: True if pos is in the room, False otherwise.
        """
        if 0 <= pos.x < self.width and 0 <=pos.y < self.height:
            return True
        else:
            return False
        


# Class Robot
class Robot(object):
    """
    Represents a robot cleaning a particular room.

    At all times the robot has a particular position and direction in the room.
    The robot also has a fixed speed.

    Subclasses of Robot should provide movement strategies by implementing
    updatePositionAndClean(), which simulates a single time-step.
    """
    def __init__(self, room, speed):
        """
        Initializes a Robot with the given speed in the specified room. The
        robot initially has a random direction and a random position in the
        room. The robot cleans the tile it is on.

        room:  a RectangularRoom object.
        speed: a float (speed > 0)
        """
        self.room = room
        self.position = room.getRandomPosition()
        self.direction = int(random.randint(0,359))
        self.speed = speed
        room.cleanTileAtPosition(self.position)
        

    def getRobotPosition(self):
        """
        Returns the position of the robot.

        returns: a Position object giving the robot's position.
        """
        return self.position
        
    
    def getRobotDirection(self):
        """
        Returns the direction of the robot.

        returns: an integer d giving the direction of the robot as an angle in
        degrees, 0 <= d < 360.
        """
        return self.direction
        

    def setRobotPosition(self, position):
        """
        Sets the position of the robot to POSITION.

        position: a Position object.
        """
        self.position = position
        

    def setRobotDirection(self, direction):
        """
        Sets the direction of the robot to DIRECTION.

        direction: integer representing an angle in degrees
        """
        self.direction = direction
        

    def updatePositionAndClean(self):
        """
        Simulates the passage of a single time-step.

        Moves the robot to a new position and mark the tile it is on as having
        been cleaned.
        """
        raise NotImplementedError


# Class StandardRobot
class StandardRobot(Robot):
    """
    A StandardRobot is a Robot with the standard movement strategy.

    At each time-step, a StandardRobot attempts to move in its current
    direction; when it would hit a wall, it *instead* chooses a new direction
    randomly.
    """
    def updatePositionAndClean(self):
        """
        Simulates the passage of a single time-step.

        Moves the robot to a new position and mark the tile it is on as having
        been cleaned.
        """
        newPosition = self.position.getNewPosition(self.direction, self.speed)
        while self.room.isPositionInRoom(newPosition) == False:
            self.direction = int(random.randint(0,359))
            newPosition = self.position.getNewPosition(self.direction, self.speed)
        self.position = newPosition
        self.room.cleanTileAtPosition(self.position)
        

# Function to run the simulation
def runSimulation(num_robots, speed, width, height, min_coverage, num_trials,
                  robot_type):
    """
    Runs NUM_TRIALS trials of the simulation and returns the mean number of
    time-steps needed to clean the fraction MIN_COVERAGE of the room.

    The simulation is run with NUM_ROBOTS robots of type ROBOT_TYPE, each with
    speed SPEED, in a room of dimensions WIDTH x HEIGHT.

    num_robots: an int (num_robots > 0)
    speed: a float (speed > 0)
    width: an int (width > 0)
    height: an int (height > 0)
    min_coverage: a float (0 <= min_coverage <= 1.0)
    num_trials: an int (num_trials > 0)
    robot_type: class of robot to be instantiated (e.g. StandardRobot or
                RandomWalkRobot)
    """
    time_steps_needed = []

    for t in range(num_trials):
        # anim = ps2_visualize.RobotVisualization(num_robots, width, height)   #only for animation
        room = RectangularRoom(width, height)
        robots = []
        for i in range(num_robots):
            robots.append(robot_type(room, speed))
        count = 0
        while room.getNumCleanedTiles() < min_coverage * room.getNumTiles():
            # anim.update(room, robots) #only for animation
            for j in range(num_robots):
                robots[j].updatePositionAndClean()
            count += 1
        time_steps_needed.append(count)
        # anim.done() #only for animation
    return sum(time_steps_needed) / len(time_steps_needed)
    

# Class RandomWalkRobot
class RandomWalkRobot(Robot):
    """
    A RandomWalkRobot is a robot with the "random walk" movement strategy: it
    chooses a new direction at random at the end of each time-step.
    """
    def updatePositionAndClean(self):
        """
        Simulates the passage of a single time-step.

        Moves the robot to a new position and marks the tile it is on as having
        been cleaned.
        """
        newPosition = self.position.getNewPosition(self.direction, self.speed)
        while self.room.isPositionInRoom(newPosition) == False:
            self.direction = int(random.randint(0, 359))
            newPosition = self.position.getNewPosition(self.direction, self.speed)
        self.position = newPosition
        self.direction = int(random.randint(0,359))
        self.room.cleanTileAtPosition(self.position)


# Plotting functions
def showPlot1(title, x_label, y_label):
    num_robot_range = range(1, 11)
    times1 = []
    times2 = []
    for num_robots in num_robot_range:
        print("Plotting", num_robots, "robots...")
        times1.append(runSimulation(num_robots, 1.0, 20, 20, 0.8, 20, StandardRobot))
        times2.append(runSimulation(num_robots, 1.0, 20, 20, 0.8, 20, RandomWalkRobot))
    pylab.plot(num_robot_range, times1)
    pylab.plot(num_robot_range, times2)
    pylab.title(title)
    pylab.legend(('StandardRobot', 'RandomWalkRobot'))
    pylab.xlabel(x_label)
    pylab.ylabel(y_label)
    pylab.show()

    
def showPlot2(title, x_label, y_label):
    aspect_ratios = []
    times1 = []
    times2 = []
    for width in [10, 20, 25, 50]:
        height = 300//width
        print("Plotting cleaning time for a room of width:", width, "by height:", height)
        aspect_ratios.append(float(width) / height)
        times1.append(runSimulation(2, 1.0, width, height, 0.8, 200, StandardRobot))
        times2.append(runSimulation(2, 1.0, width, height, 0.8, 200, RandomWalkRobot))
    pylab.plot(aspect_ratios, times1)
    pylab.plot(aspect_ratios, times2)
    pylab.title(title)
    pylab.legend(('StandardRobot', 'RandomWalkRobot'))
    pylab.xlabel(x_label)
    pylab.ylabel(y_label)
    pylab.show()
    

# Test movement of robots
testRobotMovement(StandardRobot, RectangularRoom)
testRobotMovement(RandomWalkRobot, RectangularRoom)

# Test how long simulations take
print(runSimulation(1, 1.0, 10, 10, 0.75, 30, StandardRobot))
print(runSimulation(2, 1.0, 10, 10, 0.75, 30, RandomWalkRobot))

# Plot
showPlot1("titletest", "x", "y")
showPlot2("titletest", "x", "y")
