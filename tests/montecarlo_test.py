import pytest
from LabIFSC import *
import numpy as np


def test_funcoesnativas(): #sem numpy 
  assert montecarlo(lambda x: x**2,Medida((3,0.1),""))
  #operações básicas
  a=Medida((1,0.1),"")
  b=Medida((2,0.05),"")
  assert montecarlo(lambda x,y: x+y,a,b)==a+b
  assert montecarlo(lambda x,y: x-y,a,b)==a-b
  assert montecarlo(lambda x,y: x*y,a,b)==a*b
  assert montecarlo(lambda x,y: x/y,a,b)==a/b
  #funções LabIFSC
  m1=Medida((20,1.5),"")
  assert montecarlo(lambda x: cos(m1),m1)==cos(m1)
  assert montecarlo(lambda x: sin(m1),m1)==sin(m1)
  assert montecarlo(lambda x: tan(m1),m1)==tan(m1)
  m1=Medida((0.4,0.01),"")
  assert montecarlo(lambda x: arc_cos(m1),m1)==arc_cos(m1)
  assert montecarlo(lambda x: arc_sin(m1),m1)==arc_sin(m1)
  assert montecarlo(lambda x: arc_tan(m1),m1)==arc_tan(m1)
  m1=Medida((1.2,0.01),"")
  assert montecarlo(lambda x: ln(m1),m1)==ln(m1)
  assert montecarlo(lambda x: log2(m1),m1)==log2(m1)
  assert montecarlo(lambda x: log10(m1),m1)==log10(m1)
  assert montecarlo(lambda x: sqrt(m1),m1)==sqrt(m1)

def test_funcoesnativas_MCarlo(): #sem numpy 
  m1=Medida((20,1.5),""); m1_c=MCarlo((20,1.5),"")
  assert cos(m1_c)==cos(m1)
  assert sin(m1_c)==sin(m1)
  assert tan(m1_c)==tan(m1)
  m1=Medida((0.4,0.01),"rad"); m1_c=MCarlo((0.4,0.01),"rad")
  assert arc_cos(m1_c)==arc_cos(m1)
  assert arc_sin(m1_c)==arc_sin(m1)
  assert arc_tan(m1_c)==arc_tan(m1)
  m1=Medida((1.2,0.01),""); m1_c=MCarlo((1.2,0.01),"")
  assert ln(m1_c)==ln(m1)
  assert log2(m1_c)==log2(m1)
  assert log10(m1_c)==log10(m1)
  assert sqrt(m1_c)==sqrt(m1)

def test_funcoesnativas_numpy_MCarlo(): #com numpy
  import numpy as np 
  m1=Medida((20,1.5),""); m1_c=MCarlo((20,1.5),"")
  assert cos(m1_c)==cos(m1)
  assert sin(m1_c)==sin(m1)
  assert tan(m1_c)==tan(m1)
  m1=Medida((0.4,0.01),"rad"); m1_c=MCarlo((0.4,0.01),"rad")
  assert arc_cos(m1_c)==arc_cos(m1)
  assert arc_sin(m1_c)==arc_sin(m1)
  assert arc_tan(m1_c)==arc_tan(m1)
  m1=Medida((1.2,0.01),""); m1_c=MCarlo((1.2,0.01),"")
  assert ln(m1_c)==ln(m1)
  assert log2(m1_c)==log2(m1)
  assert log10(m1_c)==log10(m1)
  assert sqrt(m1_c)==sqrt(m1)


def test_wrongvariables():
  a=Medida((7,0.3),"")
  b=Medida((3,0.1),"")
  statements=[lambda:montecarlo(lambda x:x,3),
              lambda:montecarlo(lambda x:x,3.13),  
              lambda:montecarlo(lambda x:x,"string"),
              lambda:montecarlo(lambda x,y:x+y,[a,b]),
              lambda: montecarlo(lambda x:x,a,probabilidade=3),
              lambda: montecarlo(lambda x:x,a,probabilidade=5.1),
              lambda: montecarlo(lambda x:x,a,comparar="True"),
              lambda: montecarlo(lambda x:x,a,comparar="False"),
              lambda: montecarlo(lambda x:x,a,hist="True"),
              lambda: montecarlo(lambda x:x,a,hist="False")] 
  for j in statements:
    with pytest.raises(ValueError):
        j()



def test_importmodules():
    from sys import modules
    import numpy; import matplotlib.pyplot
    assert "numpy" in modules
    assert "matplotlib.pyplot" in modules
    assert montecarlo(lambda x: x**2,Medida((3,0.1),""))
    #operações básicas
    a=Medida((1,0.1),"")
    b=Medida((2,0.05),"")
    assert montecarlo(lambda x,y: x+y,a,b)==a+b
    assert montecarlo(lambda x,y: x-y,a,b)==a-b
    assert montecarlo(lambda x,y: x*y,a,b)==a*b
    assert montecarlo(lambda x,y: x/y,a,b)==a/b
    #funções LabIFSC
    m1=Medida((20,1.5),"")
    assert montecarlo(lambda x: cos(m1),m1)==cos(m1)
    assert montecarlo(lambda x: sin(m1),m1)==sin(m1)
    assert montecarlo(lambda x: tan(m1),m1)==tan(m1)
    m1=Medida((0.4,0.01),"")
    assert montecarlo(lambda x: arc_cos(m1),m1)==arc_cos(m1)
    assert montecarlo(lambda x: arc_sin(m1),m1)==arc_sin(m1)
    assert montecarlo(lambda x: arc_tan(m1),m1)==arc_tan(m1)
    m1=Medida((1.2,0.01),"")
    assert montecarlo(lambda x: ln(m1),m1)==ln(m1)
    assert montecarlo(lambda x: log2(m1),m1)==log2(m1)
    assert montecarlo(lambda x: log10(m1),m1)==log10(m1)
    assert montecarlo(lambda x: sqrt(m1),m1)==sqrt(m1)
def test_probabilidade(): #68-95-99.7 rule
  from math import isclose
  from random import random
  for _ in range(10):
      media, sigma=random(),random() #gaussianas aleatorias
      a=Medida((media,sigma),"")
      assert isclose(montecarlo(lambda x:x, a,N=10000,probabilidade=[media-sigma,media+sigma]),0.68,abs_tol=0.02)
      assert isclose(montecarlo(lambda x:x, a,N=10000,probabilidade=[media-2*sigma,media+2*sigma]),0.95,abs_tol=0.01)
      assert isclose(montecarlo(lambda x:x, a,N=10000,probabilidade=[media-3*sigma,media+3*sigma]),0.997,abs_tol=0.01)
      assert montecarlo(lambda x:x, Medida((1,0.1),""),N=10000,probabilidade=[media-100000*sigma,media+100000*sigma])==1


def equivalencia_MCarlo_Medida(): #incertezas pequenas os metodos são equivalentes
   from random import random
   for _ in range(100):
      media1, sigma1=random(),0.01*random()
      media2, sigma2=random(),0.01*random()
      a_carlo= MCarlo((media1,sigma1),"")
      b_carlo= MCarlo((media2,sigma2),"")
      a=Medida((media1,sigma1),"")
      b=Medida((media2,sigma2),"")
      assert a_carlo+b_carlo==a+b
      assert a_carlo*b_carlo==a*b
      assert a_carlo**b_carlo==a**b
      assert a_carlo/b_carlo==a/b
      assert a_carlo-b_carlo==a-b

