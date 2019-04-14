# --------------------------------------------------------------------------
# ------------  Metody Systemowe i Decyzyjne w Informatyce  ----------------
# --------------------------------------------------------------------------
#  Zadanie 1: Regresja liniowa
#  autorzy: A. Gonczarek, J. Kaczmar, S. Zareba
#  2017
# --------------------------------------------------------------------------

import numpy as np
from utils import polynomial


def mean_squared_error(x, y, w):
    '''
    :param x: ciag wejsciowy Nx1
    :param y: ciag wyjsciowy Nx1
    :param w: parametry modelu (M+1)x1
    :return: blad sredniokwadratowy pomiedzy wyjsciami y
    oraz wyjsciami uzyskanymi z wielowamiu o parametrach w dla wejsc x
    '''
    return np.sum((y - polynomial(x,w))**2)/np.shape(x)[0]

def design_matrix(x_train, M):
    '''
    :param x_train: ciag treningowy Nx1
    :param M: stopien wielomianu 0,1,2,...
    :return: funkcja wylicza Design Matrix Nx(M+1) dla wielomianu rzedu M
    '''
    return np.transpose([x_train ** i for i in range(M + 1)])[0]


def least_squares(x_train, y_train, M):
    '''
    :param x_train: ciag treningowy wejscia Nx1
    :param y_train: ciag treningowy wyjscia Nx1
    :param M: rzad wielomianu
    :return: funkcja zwraca krotke (w,err), gdzie w sa parametrami dopasowanego wielomianu, a err blad sredniokwadratowy
    dopasowania
    '''
    fi = design_matrix(x_train, M)
    fiT = fi.transpose()
    w = np.linalg.inv(fiT @ fi) @ fiT @ y_train
    return w, mean_squared_error(x_train, y_train, w)


def regularized_least_squares(x_train, y_train, M, regularization_lambda):
    '''
    :param x_train: ciag treningowy wejscia Nx1
    :param y_train: ciag treningowy wyjscia Nx1
    :param M: rzad wielomianu
    :param regularization_lambda: parametr regularyzacji
    :return: funkcja zwraca krotke (w,err), gdzie w sa parametrami dopasowanego wielomianu zgodnie z kryterium z regularyzacja l2,
    a err blad sredniokwadratowy dopasowania
    '''
    fi = design_matrix(x_train, M)
    fiT = fi.transpose()
    I = np.eye(fi.shape[1])
    w = np.linalg.inv(fiT @ fi + regularization_lambda * I) @ fiT @ y_train
    return w, mean_squared_error(x_train, y_train, w)


def model_selection(x_train, y_train, x_val, y_val, M_values):
    '''
    :param x_train: ciag treningowy wejscia Nx1
    :param y_train: ciag treningowy wyjscia Nx1
    :param x_val: ciag walidacyjny wejscia Nx1
    :param y_val: ciag walidacyjny wyjscia Nx1
    :param M_values: tablica stopni wielomianu, ktore maja byc sprawdzone
    :return: funkcja zwraca krotke (w,train_err,val_err), gdzie w sa parametrami modelu, ktory najlepiej generalizuje dane,
    tj. daje najmniejszy blad na ciagu walidacyjnym, train_err i val_err to bledy na sredniokwadratowe na ciagach treningowym
    i walidacyjnym
    '''
    tablica1 = []
    tablica2 = []
    for M in M_values:
        tablica1.append(least_squares(x_train,y_train,M))
        tablica2.append(mean_squared_error(x_val,y_val,tablica1[M][0]))
    index = np.argsort(tablica2)[0]
    best = tablica1[index]
    return (best[0],best[1],tablica2[index])
    
def regularized_model_selection(x_train, y_train, x_val, y_val, M, lambda_values):
    '''
    :param x_train: ciag treningowy wejscia Nx1
    :param y_train: ciag treningowy wyjscia Nx1
    :param x_val: ciag walidacyjny wejscia Nx1
    :param y_val: ciag walidacyjny wyjscia Nx1
    :param M: stopien wielomianu
    :param lambda_values: lista ze wartosciami roznych parametrow regularyzacji
    :return: funkcja zwraca krotke (w,train_err,val_err,regularization_lambda), gdzie w sa parametrami modelu, ktory najlepiej generalizuje dane,
    tj. daje najmniejszy blad na ciagu walidacyjnym. Wielomian dopasowany jest wg kryterium z regularyzacja. train_err i val_err to
    bledy na sredniokwadratowe na ciagach treningowym i walidacyjnym. regularization_lambda to najlepsza wartosc parametru regularyzacji
    '''
    tablica1 = []
    tablica2 = []
    for i in lambda_values:
        tablica1.append(regularized_least_squares(x_train,y_train,M,i))
    for i in range(len(lambda_values)):
        tablica2.append(mean_squared_error(x_val,y_val,tablica1[i][0]))
    index = np.argsort(tablica2)[0]
    best = tablica1[index]
    return(best[0],best[1],tablica2[index],lambda_values[index])