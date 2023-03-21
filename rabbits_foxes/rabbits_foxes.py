import random
import pylab


#================================
# Rabbit and fox population growth
#================================


# Global Variables
MAXRABBITPOP = 1000
CURRENTRABBITPOP = 500
CURRENTFOXPOP = 30

# Function to simulate rabbit population growth
def rabbitGrowth():
    """ 
    rabbitGrowth is called once at the beginning of each time step.

    It makes use of the global variables: CURRENTRABBITPOP and MAXRABBITPOP.

    The global variable CURRENTRABBITPOP is modified by this procedure.

    For each rabbit, based on the probabilities in the problem set write-up, 
      a new rabbit may be born.
    Nothing is returned.
    """
    global CURRENTRABBITPOP
    additionals = 0
    for i in range(CURRENTRABBITPOP):
        if random.uniform(0,1) <= CURRENTRABBITPOP * (1 - CURRENTRABBITPOP / MAXRABBITPOP):
            additionals += 1
    helper = 0
    if CURRENTRABBITPOP + additionals <= MAXRABBITPOP:
        helper = additionals
    CURRENTRABBITPOP += helper


# Function to simulate fox population growth
def foxGrowth():
    """ 
    foxGrowth is called once at the end of each time step.

    It makes use of the global variables: CURRENTFOXPOP and CURRENTRABBITPOP,
        and both may be modified by this procedure.

    Each fox, based on the probabilities in the problem statement, may eat 
      one rabbit (but only if there are more than 10 rabbits).

    If it eats a rabbit, then with a 1/3 prob it gives birth to a new fox.

    If it does not eat a rabbit, then with a 1/10 prob it dies.

    Nothing is returned.
    """
    global CURRENTRABBITPOP
    global CURRENTFOXPOP
    rabbits_lost = 0
    additional_foxes = 0
    foxes_lost = 0
    for i in range(CURRENTFOXPOP):
        if random.uniform(0,1) <= (CURRENTRABBITPOP / MAXRABBITPOP):
            rabbits_lost += 1
            if random.uniform(0,1) <= 1/3:
                additional_foxes += 1
        else:
            if random.uniform(0,1) <= 0.1:
                foxes_lost += 1
    helper1 = 0
    if CURRENTRABBITPOP - rabbits_lost >= 10:
        helper1 = rabbits_lost
    CURRENTRABBITPOP -= helper1
    helper2 = 0
    if CURRENTFOXPOP + additional_foxes - foxes_lost >= 10:
        helper2 = additional_foxes - foxes_lost
    CURRENTFOXPOP += helper2


# Function to simulate rabbit and fox populations simultaneously
def runSimulation(numSteps):
    """
    Runs the simulation for `numSteps` time steps.

    Returns a tuple of two lists: (rabbit_populations, fox_populations)
      where rabbit_populations is a record of the rabbit population at the 
      END of each time step, and fox_populations is a record of the fox population
      at the END of each time step.

    Both lists should be `numSteps` items long.
    """
    rabbitPop = []
    foxPop = []
    for i in range(numSteps):
        rabbitGrowth()
        foxGrowth()
        rabbitPop.append(CURRENTRABBITPOP)
        foxPop.append(CURRENTFOXPOP)
    return rabbitPop, foxPop
    pass


# Run simulation for 1000 steps
runSimulation(1000)