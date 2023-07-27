#!/usr/bin/env python
# -*- coding: utf-8 -*-

import math
from .medida import Medida, MCarlo,montecarlo

__all__ = ["cos", "sin", "tan", "arc_cos", "arc_sin", "arc_tan", "log", "log10", "log2", "ln", "sqrt", "cbrt", "dam", "mean"]


from sys import modules
numpy_import=False
if "numpy" in modules: 
    import numpy as np
    numpy_import=True

def soma(x):
    try:
        return sum(x)
    except:
        m = Medida(
            (sum(list(map(lambda x: x.nominal, x))),
            sum(list(map(lambda x: x.incerteza, x)))), x[0].unidades_originais)
        return m

def torna_medida(x):
    if not isinstance(x, Medida):
        return Medida(x)
    return x

def cos(x):
    if not isinstance(x,MCarlo):
        x    = torna_medida(x)
        nom  = math.cos(x.nominal)
        err  = math.sin(x.nominal)
        err *= x.incerteza
        return Medida((nom, err),"")
    else:
        if numpy_import==True:
            calculo=montecarlo(lambda x: np.cos(x),x)
        else:
            calculo=montecarlo(lambda x: math.cos(x),x)
        return MCarlo((calculo.nominal,calculo.incerteza),x.unidade())

def sin(x):
    if not isinstance(x,MCarlo):
        x    = torna_medida(x)
        nom  = math.sin(x.nominal)
        err  = math.cos(x.nominal)
        err *= x.incerteza
        return Medida((nom, err))
    else:
        if numpy_import==True:
            calculo=montecarlo(lambda x: np.sin(x),x)
        else:
            calculo=montecarlo(lambda x: math.sin(x),x)
        return MCarlo((calculo.nominal,calculo.incerteza),x.unidade())

def tan(x):
    if not isinstance(x,MCarlo):
        x    = torna_medida(x)
        nom  = math.tan(x.nominal)
        err  = (1.0/math.cos(x.nominal))**2
        err *= x.incerteza
        return Medida((nom, err))
    else:
        if numpy_import==True:
            calculo=montecarlo(lambda x: np.tan(x),x)
        else:
            calculo=montecarlo(lambda x: math.tan(x),x)
        return MCarlo((calculo.nominal,calculo.incerteza),x.unidade())
    

def arc_cos(x):
    if not isinstance(x,MCarlo):
        x    = torna_medida(x)
        nom  = math.acos(x.nominal)
        err  = 1/math.sqrt(1 - x.nominal**2)
        err *= x.incerteza
        return Medida((nom, err), "")
    else:
        if numpy_import==True:
            calculo=montecarlo(lambda x: np.arccos(x),x)
        else:
            calculo=montecarlo(lambda x: math.acos(x),x)
        return MCarlo((calculo.nominal,calculo.incerteza),x.unidade())

def arc_sin(x):
    if not isinstance(x,MCarlo):
        x    = torna_medida(x)
        nom  = math.asin(x.nominal)
        err  = 1/math.sqrt(1 - x.nominal**2)
        err *= x.incerteza
        return Medida((nom, err), "")
    else:
        if numpy_import==True:
            calculo=montecarlo(lambda x: np.arcsin(x),x)
        else:  
            calculo=montecarlo(lambda x: math.asin(x),x)
        return MCarlo((calculo.nominal,calculo.incerteza),x.unidade())
def arc_tan(x):
    if not isinstance(x,MCarlo):
        x    = torna_medida(x)
        nom  = math.atan(x.nominal)
        err  = 1/math.sqrt(1 - x.nominal**2)
        err *= x.incerteza
        return Medida((nom, err), "")
    else:
        if numpy_import==True:
            calculo=montecarlo(lambda x: np.arctan(x),x)
        else:
            calculo=montecarlo(lambda x: math.atan(x),x)
        return MCarlo((calculo.nominal,calculo.incerteza),x.unidade())
    
def log(x, b):
    if not isinstance(x,MCarlo):
        x    = torna_medida(x)
        nom  = math.log(x.nominal, b)
        err  = math.log(math.exp(1), b)/x.nominal
        err *= x.incerteza
        return Medida((nom, err), x.unidades_originais)
    else:
        if numpy_import==True:
            calculo=montecarlo(lambda x: np.log(x)/np.log(b),x)
        else:
            calculo=montecarlo(lambda x: math.log(x,b),x)
        return MCarlo((calculo.nominal,calculo.incerteza),x.unidade())
def log2(x):
    return log(x, 2)

def log10(x):
    return log(x, 10)

def ln(x):
    return log(x, math.exp(1))

def sqrt(x):
    return x**0.5

def cbrt(x):
    return x**(1.0/3.0)


def dam(x):
    new_arr = []
    for i in x:
        if type(i).__name__ == 'Medida':
            new_arr.append(i.nominal)
        else:
            new_arr.append(i)
    soma = 0
    media = sum(new_arr)/len(new_arr)
    for i in new_arr:
        soma+=abs(i - media)
    
    return soma/len(new_arr)

def mean(x):
    new_arr = []
    for i in x:
        if type(i).__name__ == 'Medida':
            new_arr.append(i.nominal)
        else:
            new_arr.append(i)
    media = sum(new_arr)/len(new_arr)
    return media
