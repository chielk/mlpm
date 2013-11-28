import numpy as np
import numpy.matlib
import math
import matplotlib.pyplot as plt
import scipy.stats
from scipy import integrate

priorList = [lambda x: np.cosh(x)**-1 , lambda x: np.exp(-.5*x**2)*np.cosh(x), lambda x: np.exp(-0.25*x**4), lambda x: (x**2+5)**(-3)]
normFactors = []
normalizedPriors = []

for i in range(len(priorList)):
    norm =  1/integrate.quad(priorList[i], -np.inf, np.inf)[0]
    normFactors.append(norm)

normFactors[1] = 1/4.13273
 
##normalizedPriors.append(lambda x: normFactors[0]*np.cosh(x)**-1)
##normalizedPriors.append(lambda x: normfactors[1]*np.exp(-.5*x**2)*np.cosh(x))
##normalizedPriors.append(lambda x: normFactors[2]*np.exp(-0.25*x**4))
##normalizedPriors.append(lambda x: normFactors[3]*(x**2+5)**(-3))            
##
    
numerator = []
x= lambda x: normFactors[0]*(x**4)*np.cosh(x)**-1
numerator.append(integrate.quad(x, -np.inf, np.inf)[0])
y= lambda x: normFactors[1]*np.exp(-.5*x**2)*np.cosh(x)*x**4
numerator.append(integrate.quad(y, -np.inf, np.inf)[0])
z= lambda x: normFactors[2]*np.exp(-0.25*x**4)*x**4
numerator.append(integrate.quad(z, -np.inf, np.inf)[0])
w =  lambda x: normFactors[3]*(x**2+5)**(-3)*x**4
numerator.append(integrate.quad(w, -np.inf, np.inf)[0])


denominator =[]
x= lambda x: normFactors[0]*(x**2)*np.cosh(x)**-1
denominator.append((integrate.quad(x, -np.inf, np.inf)[0])**2)
y= lambda x: normFactors[1]*np.exp(-.5*x**2)*np.cosh(x)*x**2
denominator.append((integrate.quad(y, -np.inf, np.inf)[0])**2)
z= lambda x: normFactors[2]*np.exp(-0.25*x**4)*x**2
denominator.append((integrate.quad(z, -np.inf, np.inf)[0])**2)
w =  lambda x: normFactors[3]*(x**2+5)**(-3)*x**2
denominator.append((integrate.quad(w, -np.inf, np.inf)[0])**2)

kurtosis = []
for i in range(len(numerator)):
    kurtosis.append(numerator[i]/denominator[i])

kurtosis[1] = 2.5 #derived by Mathematica   
print kurtosis

