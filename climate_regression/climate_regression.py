import numpy as np
import pylab
import re
import os
os.environ["OPENBLAS_NUM_THREADS"] = "1"
import numpy as np


#================================
# Climate regression
#================================


# Cities in weather data
CITIES = [
    'BOSTON',
    'SEATTLE',
    'SAN DIEGO',
    'PHILADELPHIA',
    'PHOENIX',
    'LAS VEGAS',
    'CHARLOTTE',
    'DALLAS',
    'BALTIMORE',
    'SAN JUAN',
    'LOS ANGELES',
    'MIAMI',
    'NEW ORLEANS',
    'ALBUQUERQUE',
    'PORTLAND',
    'SAN FRANCISCO',
    'TAMPA',
    'NEW YORK',
    'DETROIT',
    'ST LOUIS',
    'CHICAGO'
]

INTERVAL_1 = list(range(1961, 2006))
INTERVAL_2 = list(range(2006, 2016))


# Class Climate
class Climate(object):
    """
    The collection of temperature records loaded from given csv file
    """
    def __init__(self, filename):
        """
        Initializes a Climate instance, which stores the temperature records
        loaded from a given csv file specified by filename.

        Args:
            filename: name of the csv file (str)
        """
        self.rawdata = {}

        f = open(filename, 'r')
        header = f.readline().strip().split(',')
        for line in f:
            items = line.strip().split(',')

            date = re.match('(\d\d\d\d)(\d\d)(\d\d)', items[header.index('DATE')])
            year = int(date.group(1))
            month = int(date.group(2))
            day = int(date.group(3))

            city = items[header.index('CITY')]
            temperature = float(items[header.index('TEMP')])
            if city not in self.rawdata:
                self.rawdata[city] = {}
            if year not in self.rawdata[city]:
                self.rawdata[city][year] = {}
            if month not in self.rawdata[city][year]:
                self.rawdata[city][year][month] = {}
            self.rawdata[city][year][month][day] = temperature
            
        f.close()

    def get_yearly_temp(self, city, year):
        """
        Gets the daily temperatures for the given year and city.

        Args:
            city: city name (str)
            year: the year to get the data for (int)

        Returns:
            a numpy 1-d array of daily temperatures for the specified year and
            city
        """
        temperatures = []
        assert city in self.rawdata, "provided city is not available"
        assert year in self.rawdata[city], "provided year is not available"
        for month in range(1, 13):
            for day in range(1, 32):
                if day in self.rawdata[city][year][month]:
                    temperatures.append(self.rawdata[city][year][month][day])
        return np.array(temperatures)

    def get_daily_temp(self, city, month, day, year):
        """
        Gets the daily temperature for the given city and time (year + date).

        Args:
            city: city name (str)
            month: the month to get the data for (int, where January = 1,
                December = 12)
            day: the day to get the data for (int, where 1st day of month = 1)
            year: the year to get the data for (int)

        Returns:
            a float of the daily temperature for the specified time (year +
            date) and city
        """
        assert city in self.rawdata, "provided city is not available"
        assert year in self.rawdata[city], "provided year is not available"
        assert month in self.rawdata[city][year], "provided month is not available"
        assert day in self.rawdata[city][year][month], "provided day is not available"
        return self.rawdata[city][year][month][day]


# Function to create regression models
def generate_models(x, y, degs):
    """
    Generates regression models by fitting a polynomial for each degree in degs
    to points (x, y).
    Args:
        x: a list with length N, representing the x-coords of N sample points
        y: a list with length N, representing the y-coords of N sample points
        degs: a list of degrees of the fitting polynomial
    Returns:
        a list of numpy arrays, where each array is a 1-d array of coefficients
        that minimizes the squared error of the fitting polynomial
    """
    xVals=np.asarray(x)
    yVals=np.asarray(y)
    modelList=[]
    for i in degs:
        model=np.polyfit(xVals, yVals, i)
        modelList.append(model)
    return modelList
    

# Function to estimate R-squared error term
def r_squared(y, estimated):
    """
    Calculates the R-squared error term.
    Args:
        y: list with length N, representing the y-coords of N sample points
        estimated: a list of values estimated by the regression model
    Returns:
        a float for the R-squared error term
    """
    numerator = 0
    for i in range(len(y)):
        numerator += (y[i] - estimated[i])**2

    mean = sum(y) / len(y)
    denominator = 0
    for i in range(len(y)):
        denominator += (y[i] - mean)**2

    if denominator != 0:
        return 1 - (numerator/denominator)
    else:
        return "No valid data!"


# Function to evaluate fit of different models
def evaluate_models_on_training(x, y, models):
    """
    For each regression model, computes the R-square for this model with the
    standard error over slope of a linear regression line (only if the model is
    linear), and plots the data along with the best fit curve.

    Args:
        x: a list of length N, representing the x-coords of N sample points
        y: a list of length N, representing the y-coords of N sample points
        models: a list containing the regression models you want to apply to
            your data. Each model is a numpy array storing the coefficients of
            a polynomial.
    Returns:
        None
    """
    xVals=np.asarray(x)
    yVals=np.asarray(y)
    pylab.plot(xVals, yVals, 'bo', label='Measured points')
    for i in models:
        pylab.plot(xVals, yVals, 'bo', label = 'Measured points')
        estYVals=pylab.polyval(i, xVals)
        error= r_squared(yVals, estYVals)
        pylab.plot(xVals, estYVals, 'r', label = 'Fit for model: R2 = ' \
                   + str(round(error, 5)))
        pylab.legend(loc='best')
        pylab.show()


# Run
raw_data = Climate('data.csv')
y = []

for year in INTERVAL_1:
    y.append(np.mean(raw_data.get_yearly_temp('BOSTON', year)))
models = generate_models(INTERVAL_1, y, [1])
evaluate_models_on_training(INTERVAL_1, y, models)

for year in INTERVAL_2:
    y.append(np.mean(raw_data.get_yearly_temp('BOSTON', year)))
models = generate_models(INTERVAL_2, y, [1])
evaluate_models_on_training(INTERVAL_2, y, models)
