from space_cows_helper import get_partitions
import time

#================================
# Transporting Space Cows
#================================

# Function to create dictionary of cow name and weight
def load_cows(filename):
    """
    Reads the contents of the given file, assumes the file contents contain
    data in the form of comma-separated cow name, weight pairs, and returns a
    dictionary containing cow names as keys and corresponding weights as values.

    Parameters:
    filename - the name of the data file as a string

    Returns:
    a dictionary of cow name (string), weight (int) pairs
    """

    cow_dict = dict()

    f = open(filename, 'r')
    
    for line in f:
        line_data = line.split(',')
        cow_dict[line_data[0]] = int(line_data[1])
    return cow_dict


# Function for greedy heuristic algorithm for cow allocation to spaceships
def greedy_cow_transport(cows,limit=10):
    """
    Uses a greedy heuristic to determine an allocation of cows that attempts to
    minimize the number of spaceship trips needed to transport all the cows. The
    returned allocation of cows may or may not be optimal.
    The greedy heuristic follows the following method:
    1. As long as the current trip can fit another cow, it adds the largest cow that will fit
       to the trip.
    2. Once the trip is full, it begins a new trip to transport the remaining cows.

    Does not mutate the given dictionary of cows.

    Parameters:
    cows - a dictionary of name (string), weight (int) pairs
    limit - weight limit of the spaceship (an int)
    
    Returns:
    A list of lists, with each inner list containing the names of cows
    transported on a particular trip and the overall list containing all the
    trips
    """
    all_trips = []
    cows_sorted = sorted(cows.items(), key=lambda x: x[1], reverse=True)

    while cows_sorted != []:
        single_trip = []
        total_weight= 0
        single_trip_dropped = []
        for i in range(len(cows_sorted)):
            if (total_weight + cows_sorted[i][1]) <= limit:
                single_trip.append(cows_sorted[i][0])
                total_weight += cows_sorted[i][1]
                single_trip_dropped.append(i)
        all_trips.append(single_trip)
        #remove elements in single_trip from cows_sorted
        for i in sorted(single_trip_dropped, reverse=True):
            cows_sorted.pop(i)
    return(all_trips)


# Function for brute force algorithm for cow allocation to spaceships
def brute_force_cow_transport(cows,limit=10):
    """
    Finds the allocation of cows that minimizes the number of spaceship trips
    via brute force.  The brute force algorithm follows the following method:
    1. It enumerates all possible ways that the cows can be divided into separate trips.
    2. It selects the allocation that minimizes the number of trips without making any trip
       that does not obey the weight limitation.
            
    Does not mutate the given dictionary of cows.

    Parameters:
    cows - a dictionary of name (string), weight (int) pairs
    limit - weight limit of the spaceship (an int)
    
    Returns:
    A list of lists, with each inner list containing the names of cows
    transported on a particular trip and the overall list containing all the
    trips
    """
    suitable_partitions = []
    for partition in get_partitions(cows):
        suitable_partitions.append(partition)
        for ship in partition:
            ship_cargo = 0
            for individual in ship:
                ship_cargo += cows[individual]
            if ship_cargo > limit:
                if (partition in suitable_partitions):
                    suitable_partitions.remove(partition)
    result=min(suitable_partitions, key=len)
    return(result)

     
# Function to compare runtime of algorithms
def compare_cow_transport_algorithms(function, limit=10):
    """
    Using the data from ps1_cow_data.txt and the specified weight limit,
    it runs the given function and prints out how long it took to complete in seconds.
    
    Parameters:
    function - an algorithm for allocating cows to spaceships
    limit - weight limit of the spaceship (an int)

    Returns:
    Does not return anything.
    """
    cows = load_cows("ps1_cow_data.txt")
    start = time.time()
    function(cows,limit)
    end = time.time()
    print(end - start)
    

# Compare the two algorithms
compare_cow_transport_algorithms(greedy_cow_transport)
compare_cow_transport_algorithms(brute_force_cow_transport)


