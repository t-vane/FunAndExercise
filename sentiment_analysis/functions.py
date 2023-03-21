from string import punctuation, digits
import numpy as np
import random

# Helper functions
def get_order(n_samples):
    try:
        with open(str(n_samples) + '.txt') as fp:
            line = fp.readline()
            return list(map(int, line.split(',')))
    except FileNotFoundError:
        random.seed(1)
        indices = list(range(n_samples))
        random.shuffle(indices)
        return indices
        
def extract_words(text):
    """
    Helper function for `bag_of_words()`.
    Args:
        a string `text`.
    Returns:
        a list of lowercased words in the string, where punctuation and digits
        count as their own words.
    """
    for c in punctuation + digits:
        text = text.replace(c, ' ' + c + ' ')
    return text.lower().split()
    
def accuracy(preds, targets):
    """
    Given length-N vectors containing predicted and target labels,
    returns the fraction of predictions that are correct.
    """
    return (preds == targets).mean()
    

# Hinge loss function for single data point
def hinge_loss_single(feature_vector, label, theta, theta_0):
    """
    Finds the hinge loss on a single data point given specific classification
    parameters.

    Args:
        `feature_vector` - numpy array describing the given data point.
        `label` - float, the correct classification of the data point.
        `theta` - numpy array describing the linear classifier.
        `theta_0` - float representing the offset parameter.
    Returns:
        the hinge loss, as a float, associated with the given data point and
        parameters.
    """
    y = np.dot(feature_vector, theta) + theta_0
    loss = np.maximum(0.0, 1 - y * label)
    return loss


# Hinge loss function for dataset
def hinge_loss_full(feature_matrix, labels, theta, theta_0):
    """
    Finds the hinge loss for given classification parameters averaged over a
    given dataset

    Args:
        `feature_matrix` - numpy matrix describing the given data. Each row
            represents a single data point.
        `labels` - numpy array where the kth element of the array is the
            correct classification of the kth row of the feature matrix.
        `theta` - numpy array describing the linear classifier.
        `theta_0` - real valued number representing the offset parameter.
    Returns:
        the hinge loss, as a float, associated with the given dataset and
        parameters.  This number is the average hinge loss across data points.
    """

    y = np.dot(feature_matrix, theta) + theta_0
    losses = np.maximum(0.0, 1 - y * labels)
    return np.mean(losses)


# Perceptron algorithm (single step)
def perceptron_single_step_update(
        feature_vector,
        label,
        current_theta,
        current_theta_0):
    """
    Updates the classification parameters `theta` and `theta_0` via a single
    step of the perceptron algorithm.  Returns new parameters rather than
    modifying in-place.

    Args:
        feature_vector - A numpy array describing a single data point.
        label - The correct classification of the feature vector.
        current_theta - The current theta being used by the perceptron
            algorithm before this update.
        current_theta_0 - The current theta_0 being used by the perceptron
            algorithm before this update.
    Returns a tuple containing two values:
        the updated feature-coefficient parameter `theta` as a numpy array.
        the updated offset parameter `theta_0` as a floating point number.
    """
    error = label * (np.dot(feature_vector, current_theta) + current_theta_0)
    
    if error <= 0:
        theta = current_theta + label * feature_vector
        theta_0 = current_theta_0 + label
        return theta, theta_0
    else:
        return current_theta, current_theta_0


# Perceptron algorithm (full)
def perceptron(feature_matrix, labels, T):
    """
    Runs the full perceptron algorithm on a given set of data. Runs T
    iterations through the data set.

    Args:
        `feature_matrix` - numpy matrix describing the given data. Each row
            represents a single data point.
        `labels` - numpy array where the kth element of the array is the
            correct classification of the kth row of the feature matrix.
        `T` - integer indicating how many times the perceptron algorithm
            should iterate through the feature matrix.

    Returns a tuple containing two values:
        the feature-coefficient parameter `theta` as a numpy array
            (found after T iterations through the feature matrix).
        the offset parameter `theta_0` as a floating point number
            (found also after T iterations through the feature matrix).
    """
    nsamples = feature_matrix.shape[0]

    theta = np.zeros(feature_matrix.shape[1])
    theta_0 = 0

    for t in range(T):
        for i in get_order(nsamples):
            theta, theta_0 = perceptron_single_step_update(feature_matrix[i, :], labels[i], theta, theta_0)
            
    return theta, theta_0


# Average perceptron algorithm
def average_perceptron(feature_matrix, labels, T):
    """
    Runs the average perceptron algorithm on a given dataset.  Runs `T`
    iterations through the dataset and therefore
    averages over `T` many parameter values.

    Args:
        `feature_matrix` -  A numpy matrix describing the given data. Each row
            represents a single data point.
        `labels` - A numpy array where the kth element of the array is the
            correct classification of the kth row of the feature matrix.
        `T` - An integer indicating how many times the perceptron algorithm
            should iterate through the feature matrix.

    Returns a tuple containing two values:
        the average feature-coefficient parameter `theta` as a numpy array
            (averaged over T iterations through the feature matrix).
        the average offset parameter `theta_0` as a floating point number
            (averaged also over T iterations through the feature matrix).
    """
    nsamples = feature_matrix.shape[0]

    theta = np.zeros(feature_matrix.shape[1])
    theta_0 = 0
    sum_theta = theta
    sum_theta_0 = theta_0

    for t in range(T):
        for i in get_order(nsamples):
            theta, theta_0 = perceptron_single_step_update(feature_matrix[i, :], labels[i], theta, theta_0)
            sum_theta += theta
            sum_theta_0 += theta_0

    return sum_theta /(T * nsamples), sum_theta_0 /(T * nsamples)


# Pegasos algorithm (single step)
def pegasos_single_step_update(
        feature_vector,
        label,
        L,
        eta,
        theta,
        theta_0):
    """
    Updates the classification parameters `theta` and `theta_0` via a single
    step of the Pegasos algorithm.  Returns new parameters rather than
    modifying in-place.

    Args:
        `feature_vector` - A numpy array describing a single data point.
        `label` - The correct classification of the feature vector.
        `L` - The lamba value being used to update the parameters.
        `eta` - Learning rate to update parameters.
        `theta` - The old theta being used by the Pegasos
            algorithm before this update.
        `theta_0` - The old theta_0 being used by the
            Pegasos algorithm before this update.
    Returns:
        a tuple where the first element is a numpy array with the value of
        theta after the old update has completed and the second element is a
        real valued number with the value of theta_0 after the old updated has
        completed.
    """
    error = label * (np.dot(feature_vector, theta) + theta_0)
    
    if error <= 1:
        theta = (1 - eta * L) * theta + eta * label * feature_vector
        theta_0 = theta_0 + eta * label
    else:
        theta = (1 - eta * L) * theta
        theta_0 = theta_0
        
    return theta, theta_0


# Pegasos algorithm (full)
def pegasos(feature_matrix, labels, T, L):
    """
    Runs the Pegasos algorithm on a given set of data. Runs T iterations
    through the data set.  For each update, learning rate = 1/sqrt(t), 
    where t is a counter for the number of updates performed so far 
    (between 1 and nT inclusive).

    Args:
        `feature_matrix` - A numpy matrix describing the given data. Each row
            represents a single data point.
        `labels` - A numpy array where the kth element of the array is the
            correct classification of the kth row of the feature matrix.
        `T` - An integer indicating how many times the algorithm
            should iterate through the feature matrix.
        `L` - The lamba value being used to update the Pegasos
            algorithm parameters.

    Returns:
        a tuple where the first element is a numpy array with the value of the
        theta, the linear classification parameter, found after T iterations
        through the feature matrix and the second element is a real number with
        the value of the theta_0, the offset classification parameter, found
        after T iterations through the feature matrix.
    """
    nsamples = feature_matrix.shape[0]

    theta = np.zeros(feature_matrix.shape[1])
    theta_0 = 0
    counter = 1

    for t in range(T):
        for i in get_order(nsamples):
            eta = 1/np.sqrt(counter)
            theta, theta_0 = pegasos_single_step_update(feature_matrix[i, :], labels[i], L, eta, theta, theta_0)
            counter += 1
            
    return theta, theta_0


# Classifier function
def classify(feature_matrix, theta, theta_0):
    """
    A classification function that uses given parameters to classify a set of
    data points.

    Args:
        `feature_matrix` - numpy matrix describing the given data. Each row
            represents a single data point.
        `theta` - numpy array describing the linear classifier.
        `theta_0` - real valued number representing the offset parameter.

    Returns:
        a numpy array of 1s and -1s where the kth element of the array is the
        predicted classification of the kth row of the feature matrix using the
        given theta and theta_0. If a prediction is GREATER THAN zero, it
        is considered a positive classification.
    """
    prediction = np.dot(feature_matrix, theta) + theta_0
    count = 0
    
    for i in prediction:
        if np.isclose(i, 0) or i <0:
            prediction[count] = -1
        else:
            prediction[count] = 1
        count += 1
        
    return prediction

# Function to estimate accuracy of classifier after training
def classifier_accuracy(
        classifier,
        train_feature_matrix,
        val_feature_matrix,
        train_labels,
        val_labels,
        **kwargs):
    """
    Trains a linear classifier and computes accuracy.  The classifier is
    trained on the train data.  The classifier's accuracy on the train and
    validation data is then returned.

    Args:
        `classifier` - A learning function that takes arguments
            (feature matrix, labels, **kwargs) and returns (theta, theta_0).
        `train_feature_matrix` - A numpy matrix describing the training
            data. Each row represents a single data point.
        `val_feature_matrix` - A numpy matrix describing the validation
            data. Each row represents a single data point.
        `train_labels` - A numpy array where the kth element of the array
            is the correct classification of the kth row of the training
            feature matrix.
        `val_labels` - A numpy array where the kth element of the array
            is the correct classification of the kth row of the validation
            feature matrix.
        `kwargs` - Additional named arguments to pass to the classifier
            (e.g. T or L).

    Returns:
        a tuple in which the first element is the (scalar) accuracy of the
        trained classifier on the training data and the second element is the
        accuracy of the trained classifier on the validation data.
    """
    # Train classifier
    theta, theta_0 = classifier(train_feature_matrix, train_labels, **kwargs)
    
    # Make predictions
    prediction_train = classify(train_feature_matrix, theta, theta_0)
    prediction_val = classify(val_feature_matrix, theta, theta_0)
    
    # Estimate accuracy
    accuracy_train = accuracy(prediction_train, train_labels)
    accuracy_val = accuracy(prediction_val, val_labels)
    
    return accuracy_train, accuracy_val


# Function to create dictionary of words
def bag_of_words(texts, remove_stopword=True):
    """
    Creates dictionary of words from `texts`.
    
    Args:
        `texts` - a list of natural language strings.
    Returns:
        a dictionary that maps each word appearing in `texts` to a unique
        integer `index`.
    """
    stopword = np.loadtxt('stopwords.txt', dtype='str')
    indices_by_word = {}
    
    for text in texts:
        word_list = extract_words(text)
        for word in word_list:
            if word in indices_by_word:
                continue
            if remove_stopword and word in stopword:
                continue
            indices_by_word[word] = len(indices_by_word)

    return indices_by_word


# Function to create matrix of feature vectors
def extract_bow_feature_vectors(reviews, indices_by_word, binarize=False):
    """
    Creates matrix with feature vectors for reviews.
    
    Args:
        `reviews` - a list of natural language strings
        `indices_by_word` - a dictionary of uniquely-indexed words.
    Returns:
        a matrix representing each review via bag-of-words features.  This
        matrix thus has shape (n, m), where n counts reviews and m counts words
        in the dictionary.
    """
    feature_matrix = np.zeros([len(reviews), len(indices_by_word)], dtype=np.float64)

    for i, text in enumerate(reviews):
        word_list = extract_words(text)
        for word in word_list:
            if word not in indices_by_word:
                continue
            feature_matrix[i, indices_by_word[word]] = word_list.count(word)

    if binarize:
        for i, text in enumerate(reviews):
            word_list = extract_words(text)
            for word in word_list:
                if word not in indices_by_word:
                    continue
                feature_matrix[i, indices_by_word[word]] = 1

    return feature_matrix
