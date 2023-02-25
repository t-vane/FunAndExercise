import random
import pylab
from disease_spread_precompiled_38 import *


#================================
# Simulate spread of disease
#================================


# Class NoChildException
class NoChildException(Exception):
    """
    NoChildException is raised by the reproduce() method in the SimpleVirus
    and ResistantVirus classes to indicate that a virus particle does not
    reproduce.
    """

# Class SimpleVirus
class SimpleVirus(object):
    """
    Representation of a simple virus (does not model drug effects/resistance).
    """
    def __init__(self, maxBirthProb, clearProb):
        """
        Initializes a SimpleVirus instance, saves all parameters as attributes
        of the instance.  
        
        maxBirthProb: Maximum reproduction probability (a float between 0-1)        
        clearProb: Maximum clearance probability (a float between 0-1).
        """
        self.maxBirthProb = maxBirthProb
        self.clearProb = clearProb

    def getMaxBirthProb(self):
        """
        Returns the maximum birth probability.
        """
        return self.maxBirthProb

    def getClearProb(self):
        """
        Returns the clear probability.
        """
        return self.clearProb

    def doesClear(self):
        """ 
        Stochastically determines whether this virus particle is cleared from the
        patient's body at a time step. 
        
        returns: True with probability self.getClearProb and otherwise returns False.
        """
        number = random.uniform(0,1)
        if number <= self.getClearProb():
            return True
        else:
            return False
        
    def reproduce(self, popDensity):
        """
        Stochastically determines whether this virus particle reproduces at a
        time step. Called by the update() method in the Patient and
        TreatedPatient classes. The virus particle reproduces with probability
        self.maxBirthProb * (1 - popDensity).
        
        If this virus particle reproduces, then reproduce() creates and returns
        the instance of the offspring SimpleVirus (which has the same
        maxBirthProb and clearProb values as its parent).         

        popDensity: the population density (a float), defined as the current
        virus population divided by the maximum population.         
        
        returns: a new instance of the SimpleVirus class representing the
        offspring of this virus particle. The child should have the same
        maxBirthProb and clearProb values as this virus. Raises a
        NoChildException if this virus particle does not reproduce.               
        """
        number = random.uniform(0,1)
        if number <= self.getMaxBirthProb() * (1 - popDensity):
            newVirus = SimpleVirus(self.getMaxBirthProb(), self.getClearProb())
        else:
            raise NoChildException
        return newVirus
        

# Class Patient
class Patient(object):
    """
    Representation of a simplified patient. The patient does not take any drugs
    and his/her virus populations have no drug resistance.
    """    
    def __init__(self, viruses, maxPop):
        """
        Initialization function, saves the viruses and maxPop parameters as
        attributes.

        viruses: the list representing the virus population (a list of
        SimpleVirus instances)

        maxPop: the maximum virus population for this patient (an integer)
        """
        self.viruses = viruses
        self.maxPop = maxPop

    def getViruses(self):
        """
        Returns the viruses in this Patient.
        """
        return self.viruses

    def getMaxPop(self):
        """
        Returns the maximum population.
        """
        return self.maxPop

    def getTotalPop(self):
        """
        Gets the size of the current total virus population. 
        
        returns: The total virus population (an integer)
        """
        return len(self.getViruses())
        
    def update(self):
        """
        Updates the state of the virus population in this patient for a single
        time step. update() executes the following steps in this order:
        - Determines whether each virus particle survives and updates the list
        of virus particles accordingly.   
        - The current population density is calculated. This population density
          value is used until the next call to update() 
        - Based on this value of population density, determine whether each 
          virus particle should reproduce and add offspring virus particles to 
          the list of viruses in this patient.                    

        returns: The total virus population at the end of the update (an integer)
        """
        helperList = []
        for item in self.viruses:
            helperList.append(item)

        for item in helperList:
            if item.doesClear():
                self.viruses.remove(item)

        helperList2 = []
        for item in self.viruses:
            helperList2.append(item)

        density = self.getTotalPop() / self.getMaxPop()

        for item in helperList2:
            try:
                self.viruses.append(item.reproduce(density))
            except NoChildException:
                continue

        return len(self.getViruses())


# Function to simulate spread without drug treatment
def simulationWithoutDrug(numViruses, maxPop, maxBirthProb, clearProb,
                          numTrials):
    """
    Runs the simulation and plots the graph without drug treatment and resistance.   
    For each of numTrials trial, instantiates a patient, runs a simulation
    for 300 timesteps, and plots the average virus population size as a
    function of time.

    numViruses: number of SimpleVirus to create for patient (an integer)
    maxPop: maximum virus population for patient (an integer)
    maxBirthProb: Maximum reproduction probability (a float between 0-1)        
    clearProb: Maximum clearance probability (a float between 0-1)
    numTrials: number of simulation runs to execute (an integer)
    """
    for i in range(numTrials):
        virusList = []
        for i in range(numViruses):
            virusList.append(SimpleVirus(maxBirthProb, clearProb))

        populationSizeList = []
        for i in range(300):
            populationSizeList.append(0)
        patient = Patient(virusList, maxPop)

        for j in range(300):
            populationSizeList[j] = patient.update()

    y_axis = [i / numTrials for i in populationSizeList]

    pylab.plot(y_axis, label="SimpleVirus")
    pylab.title("SimpleVirus simulation")
    pylab.xlabel("Time Steps")
    pylab.ylabel("Average Virus Population")
    pylab.legend(loc="best")
    pylab.show()



# Class ResistantVirus
class ResistantVirus(SimpleVirus):
    """
    Representation of a virus which can have drug resistance.
    """   
    def __init__(self, maxBirthProb, clearProb, resistances, mutProb):
        """
        Initializes a ResistantVirus instance, saves all parameters as attributes
        of the instance.

        maxBirthProb: Maximum reproduction probability (a float between 0-1)       

        clearProb: Maximum clearance probability (a float between 0-1).

        resistances: A dictionary of drug names (strings) mapping to the state
        of this virus particle's resistance (either True or False) to each drug.
        e.g. {'guttagonol':False, 'srinol':False}, means that this virus
        particle is resistant to neither guttagonol nor srinol.

        mutProb: Mutation probability for this virus particle (a float). This is
        the probability of the offspring acquiring or losing resistance to a drug.
        """
        SimpleVirus.__init__(self, maxBirthProb, clearProb)
        self.resistances = resistances
        self.mutProb = mutProb

    def getResistances(self):
        """
        Returns the resistances for this virus.
        """
        return self.resistances

    def getMutProb(self):
        """
        Returns the mutation probability for this virus.
        """
        return self.mutProb

    def isResistantTo(self, drug):
        """
        Gets the state of this virus particle's resistance to a drug. This method
        is called by getResistPop() in TreatedPatient to determine how many virus
        particles have resistance to a drug.       

        drug: The drug (a string)

        returns: True if this virus instance is resistant to the drug, False
        otherwise.
        """
        if drug in self.getResistances():
            return self.resistances[drug]
        else:
            return False

    def reproduce(self, popDensity, activeDrugs):
        """
        Stochastically determines whether this virus particle reproduces at a
        time step. Called by the update() method in the TreatedPatient class.

        A virus particle will only reproduce if it is resistant to ALL the drugs
        in the activeDrugs list. For example, if there are 2 drugs in the
        activeDrugs list, and the virus particle is resistant to 1 or no drugs,
        then it will NOT reproduce.

        Hence, if the virus is resistant to all drugs
        in activeDrugs, then the virus reproduces with probability:      

        self.maxBirthProb * (1 - popDensity).                       

        If this virus particle reproduces, then reproduce() creates and returns
        the instance of the offspring ResistantVirus (which has the same
        maxBirthProb and clearProb values as its parent). The offspring virus
        will have the same maxBirthProb, clearProb, and mutProb as the parent.

        For each drug resistance trait of the virus (i.e. each key of
        self.resistances), the offspring has probability 1-mutProb of
        inheriting that resistance trait from the parent, and probability
        mutProb of switching that resistance trait in the offspring.       

        popDensity: the population density (a float), defined as the current
        virus population divided by the maximum population       

        activeDrugs: a list of the drug names acting on this virus particle
        (a list of strings).

        returns: a new instance of the ResistantVirus class representing the
        offspring of this virus particle. The child should have the same
        maxBirthProb and clearProb values as this virus. Raises a
        NoChildException if this virus particle does not reproduce.
        """
        willReproduce = True
        for drug in activeDrugs:
            if self.isResistantTo(drug) == False:
                willReproduce = False

        newVirusResistances = self.getResistances().copy()

        number = random.uniform(0, 1)
        if willReproduce and number <= self.getMaxBirthProb() * (1 - popDensity):
            for i in self.getResistances():
                number2 = random.uniform(0,1)
                if number2 > 1 - self.getMutProb():
                    newVirusResistances[i] = not self.getResistances()[i]

            return ResistantVirus(self.getMaxBirthProb(), self.getClearProb(), newVirusResistances, self.getMutProb())
        else:
            raise NoChildException


# Class TreatedPatient
class TreatedPatient(Patient):
    """
    Representation of a patient. The patient is able to take drugs and his/her
    virus population can acquire resistance to the drugs he/she takes.
    """
    def __init__(self, viruses, maxPop):
        """
        Initialization function, saves the viruses and maxPop parameters as
        attributes. Also initializes the list of drugs being administered
        (which should initially include no drugs).              

        viruses: The list representing the virus population (a list of
        virus instances)

        maxPop: The  maximum virus population for this patient (an integer)
        """
        Patient.__init__(self, viruses, maxPop)
        self.prescriptions = []

    def addPrescription(self, newDrug):
        """
        Administers a drug to this patient. After a prescription is added, the
        drug acts on the virus population for all subsequent time steps. If the
        newDrug is already prescribed to this patient, the method has no effect.

        newDrug: The name of the drug to administer to the patient (a string).

        postcondition: The list of drugs being administered to a patient is updated
        """
        if newDrug not in self.prescriptions:
            self.prescriptions.append(newDrug)

    def getPrescriptions(self):
        """
        Returns the drugs that are being administered to this patient.

        returns: The list of drug names (strings) being administered to this patient.
        """
        return self.prescriptions

    def getResistPop(self, drugResist):
        """
        Gets the population of virus particles resistant to the drugs listed in
        drugResist.       

        drugResist: Which drug resistances to include in the population (a list
        of strings - e.g. ['guttagonol'] or ['guttagonol', 'srinol'])

        returns: The population of viruses (an integer) with resistances to all
        drugs in the drugResist list.
        """
        count = 0
        for virus in self.getViruses():
            resistance = True
            for drug in drugResist:
                if virus.isResistantTo(drug) == False:
                    resistance = False
            if resistance == True:
                count += 1

        return count

    def update(self):
        """
        Updates the state of the virus population in this patient for a single
        time step. update() should execute these actions in order:
        - Determines whether each virus particle survives and updates the list of
          virus particles accordingly
        - The current population density is calculated. This population density
          value is used until the next call to update().
        - Based on this value of population density, determines whether each 
          virus particle should reproduce and add offspring virus particles to 
          the list of viruses in this patient.
          The list of drugs being administered should be accounted for in the
          determination of whether each virus particle reproduces.

        returns: The total virus population at the end of the update (an
        integer)
        """
        helperList = self.getViruses().copy()
        for item in helperList:
            if item.doesClear():
                self.viruses.remove(item)

        helperList2 = self.getViruses().copy()

        density = self.getTotalPop() / self.getMaxPop()

        for item in helperList2:
            try:
                self.viruses.append(item.reproduce(density, self.getPrescriptions()))
            except NoChildException:
                continue

        return len(self.getViruses())


# Function to simulate spread with drug treatment
def simulationWithDrug(numViruses, maxPop, maxBirthProb, clearProb, resistances,
                       mutProb, numTrials):
    """
    Runs simulations and plots graphs with drug treatment and resistance.

    For each of numTrials trials, instantiates a patient, runs a simulation for
    150 timesteps, adds guttagonol, and runs the simulation for an additional
    150 timesteps.  At the end plots the average virus population size
    (for both the total virus population and the guttagonol-resistant virus
    population) as a function of time.

    numViruses: number of ResistantVirus to create for patient (an integer)
    maxPop: maximum virus population for patient (an integer)
    maxBirthProb: Maximum reproduction probability (a float between 0-1)        
    clearProb: maximum clearance probability (a float between 0-1)
    resistances: a dictionary of drugs that each ResistantVirus is resistant to
                 (e.g., {'guttagonol': False})
    mutProb: mutation probability for each ResistantVirus particle
             (a float between 0-1). 
    numTrials: number of simulation runs to execute (an integer)
    """
    finalListTotal = []
    finalListResistant = []
    for i in range(300):
        finalListTotal.append(0)
    for i in range(300):
        finalListResistant.append(0)

    for i in range(numTrials):
        virusList = []
        for i in range(numViruses):
            virusList.append(ResistantVirus(maxBirthProb, clearProb, resistances, mutProb))

        patient = TreatedPatient(virusList, maxPop)

        for j in range(150):
            finalListTotal[j] += patient.update()
            finalListResistant[j] += patient.getResistPop(["guttagonol"])

        patient.addPrescription("guttagonol")

        for j in range(150):
            finalListTotal[150 + j] += patient.update()
            finalListResistant[150 + j] += patient.getResistPop(["guttagonol"])

    y_axis_total = [i / numTrials for i in finalListTotal]
    y_axis_resistant = [i / numTrials for i in finalListResistant]

    pylab.plot(y_axis_total, label="TotalViruses")
    pylab.plot(y_axis_resistant, label="GuttagonolResistantViruses")
    pylab.title("ResistantVirus simulation")
    pylab.xlabel("Time Steps")
    pylab.ylabel("Average Virus Population")
    pylab.legend(loc="best")
    pylab.show()
    

# Run simulation without drug treatment
simulationWithoutDrug(100, 1000, 0.1, 0.05, 10)
# Run simulation with drug treatment
simulationWithDrug(100, 1000, 0.1, 0.05,{"guttagonol": False}, 0.005, 1)
