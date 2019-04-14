# --------------------------------------------------------------------------
# ------------  Metody Systemowe i Decyzyjne w Informatyce  ----------------
# --------------------------------------------------------------------------
#  Zadanie 2: k-NN i Naive Bayes
#  autorzy: A. Gonczarek, J. Kaczmar, S. Zareba
#  2019
# --------------------------------------------------------------------------

from __future__ import division

import numpy as np


def hamming_distance(X, X_train):
    """
    :param X: zbior porownwanych obiektow N1xD
    :param X_train: zbior obiektow do ktorych porownujemy N2xD
    Funkcja wyznacza odleglosci Hamminga obiektow ze zbioru X od
    obiektow X_train. Odleglosci obiektow z jednego i drugiego
    zbioru zwrocone zostana w postaci macierzy
    :return: macierz odleglosci pomiedzy obiektami z X i X_train N1xN2
    """
    # macierze normalne z rzadkich
    X = X.toarray()
    X_train = X_train.toarray()

    X_train = X_train.transpose()

    outArr = X.astype(np.uint8) @ X_train.astype(np.uint8)
    outArr2 = (~X).astype(np.uint8) @ (~X_train).astype(np.uint8)
    arr = outArr + outArr2

    return np.subtract(np.uint8(X_train.shape[0]), arr)
    pass

def sort_train_labels_knn(Dist, y):
    """
    Funkcja sortujaca etykiety klas danych treningowych y
    wzgledem prawdopodobienstw zawartych w macierzy Dist.
    Funkcja zwraca macierz o wymiarach N1xN2. W kazdym
    wierszu maja byc posortowane etykiety klas z y wzgledem
    wartosci podobienstw odpowiadajacego wiersza macierzy
    Dist
    :param Dist: macierz odleglosci pomiedzy obiektami z X
    i X_train N1xN2
    :param y: wektor etykiet o dlugosci N2
    :return: macierz etykiet klas posortowana wzgledem
    wartosci podobienstw odpowiadajacego wiersza macierzy
    Dist. Uzyc algorytmu mergesort.
    """
    w = Dist.argsort(kind='mergesort')
    return y[w]
    pass

def p_y_x_knn(y, k):
    """
    Funkcja wyznacza rozklad prawdopodobienstwa p(y|x) dla
    kazdej z klas dla obiektow ze zbioru testowego wykorzystujac
    klasfikator KNN wyuczony na danych trenningowych
    :param y: macierz posortowanych etykiet dla danych treningowych N1xN2
    :param k: liczba najblizszuch sasiadow dla KNN
    :return: macierz prawdopodobienstw dla obiektow z X
    """
    # resized = np.delete(y, range(k, y.shape[1]), axis=1)
    # output = np.vstack(np.apply_along_axis(np.bincount, axis=1, arr=resized, minlength=k - 1))
    # # output = np.delete(output, 0, axis=1)
    # output = np.divide(output, k)
    # return output

    points_number = 4
    result_matrix = []
    for i in range(np.shape(y)[0]):
        helper = []
        for j in range(k):
            helper.append(y[i][j])
        line = np.bincount(helper, None, points_number)
        result_matrix.append([line[0] / k, line[1] / k, line[2] / k, line[3] / k])
    return result_matrix
    pass


def classification_error(p_y_x, y_true):
    """
    Wyznacz blad klasyfikacji.
    :param p_y_x: macierz przewidywanych prawdopodobienstw
    :param y_true: zbior rzeczywistych etykiet klas 1xN.
    Kazdy wiersz macierzy reprezentuje rozklad p(y|x)
    :return: blad klasyfikacji
    """
    n = len(p_y_x)
    m = len(p_y_x[0])
    res = 0
    for i in range(0, n):
        if (m - np.argmax(p_y_x[i][::-1]) - 1) != y_true[i]:
            res += 1
    return res/n
    pass

def model_selection_knn(Xval, Xtrain, yval, ytrain, k_values):
    """
    :param Xval: zbior danych walidacyjnych N1xD
    :param Xtrain: zbior danych treningowych N2xD
    :param yval: etykiety klas dla danych walidacyjnych 1xN1
    :param ytrain: etykiety klas dla danych treningowych 1xN2
    :param k_values: wartosci parametru k, ktore maja zostac sprawdzone
    :return: funkcja wykonuje selekcje modelu knn i zwraca krotke (best_error,best_k,errors), gdzie best_error to najnizszy
    osiagniety blad, best_k - k dla ktorego blad byl najnizszy, errors - lista wartosci bledow dla kolejnych k z k_values
    """
    k = len(k_values)
    errors = []
    Dist = hamming_distance(Xval, Xtrain)
    ksort = sort_train_labels_knn(Dist, ytrain)
    for i in range(0, k):
        error = classification_error(p_y_x_knn(ksort, k_values[i]), yval)
        errors.append(error)
    best_error = min(errors)
    best_k = k_values[np.argmin(errors)]
    return best_error, best_k,errors
    pass


def estimate_a_priori_nb(ytrain):
    """
    :param ytrain: etykiety dla dla danych treningowych 1xN
    :return: funkcja wyznacza rozklad a priori p(y) i zwraca p_y - wektor prawdopodobienstw a priori 1xM
    """
    res = np.bincount(ytrain)/len(ytrain)
    return res
    pass


def estimate_p_x_y_nb(Xtrain, ytrain, a, b):
    """
    :param Xtrain: dane treningowe NxD
    :param ytrain: etykiety klas dla danych treningowych 1xN
    :param a: parametr a rozkladu Beta
    :param b: parametr b rozkladu Beta
    :return: funkcja wyznacza rozklad prawdopodobienstwa p(x|y) zakladajac, ze x przyjmuje wartosci binarne i ze elementy
    x sa niezalezne od siebie. Funkcja zwraca macierz p_x_y o wymiarach MxD.
    """

    points_number = 4
    result_matrix = []
    vector = np.bincount(ytrain)
    denominator = (vector + a + b - 2)

    def numerator(Xtrain, ytrain, d):
        Xtrain = Xtrain.toarray()
        occurences = [0, 0, 0, 0]
        for i in range(np.shape(ytrain)[0]):
            if Xtrain[i][d] == 1:
                occurences[ytrain[i]] += 1
        occurences_add = [(x + (a - 1)) for x in occurences]
        return np.squeeze(occurences_add)

    for k in range(points_number):
        line = []
        for d in range(np.shape(Xtrain)[1]):
            line.append((numerator(Xtrain, ytrain, d)[k]) / denominator[k])
        result_matrix.append(line)

    return np.array(result_matrix)
    pass


def p_y_x_nb(p_y, p_x_1_y, X):
    """
    :param p_y: wektor prawdopodobienstw a priori o wymiarach 1xM
    :param p_x_1_y: rozklad prawdopodobienstw p(x=1|y) - macierz MxD
    :param X: dane dla ktorych beda wyznaczone prawdopodobienstwa, macierz NxD
    :return: funkcja wyznacza rozklad prawdopodobienstwa p(y|x) dla kazdej z klas z wykorzystaniem klasyfikatora Naiwnego
    Bayesa. Funkcja zwraca macierz p_y_x o wymiarach NxM.
    """
    X = X.toarray()
    p_x_1_y_rev = 1 - p_x_1_y
    X_rev = 1 - X
    res = []
    for i in range(X.shape[0]):
        success = p_x_1_y ** X[i, ]
        fail = p_x_1_y_rev ** X_rev[i, ]
        a = np.prod(success * fail, axis=1) * p_y
        # suma p(x|y') * p(y')
        sum = np.sum(a)
        # prawdopodobieństwo każdej z klas podzielone przez sumę
        res.append(a / sum)
    return np.array(res)
    pass


def model_selection_nb(Xtrain, Xval, ytrain, yval, a_values, b_values):
    """
    :param Xtrain: zbior danych treningowych N2xD
    :param Xval: zbior danych walidacyjnych N1xD
    :param ytrain: etykiety klas dla danych treningowych 1xN2
    :param yval: etykiety klas dla danych walidacyjnych 1xN1
    :param a_values: lista parametrow a do sprawdzenia
    :param b_values: lista parametrow b do sprawdzenia
    :return: funkcja wykonuje selekcje modelu Naive Bayes - wybiera najlepsze wartosci parametrow a i b. Funkcja zwraca
    krotke (error_best, best_a, best_b, errors) gdzie best_error to najnizszy
    osiagniety blad, best_a - a dla ktorego blad byl najnizszy, best_b - b dla ktorego blad byl najnizszy,
    errors - macierz wartosci bledow dla wszystkich par (a,b)
    """
    errors = np.ones((len(a_values), len(b_values)))
    estimated_p_y = estimate_a_priori_nb(ytrain)
    best_a = 0
    best_b = 0
    best_error = np.inf
    for i in range(len(a_values)):
        for j in range(len(b_values)):
            error = classification_error(p_y_x_nb(estimated_p_y, estimate_p_x_y_nb(Xtrain, ytrain, a_values[i], b_values[j]), Xval), yval)
            errors[i][j] = error
            if error<best_error:
                best_a = a_values[i]
                best_b = b_values[j]
                best_error = error
    return best_error, best_a, best_b, errors
    pass
