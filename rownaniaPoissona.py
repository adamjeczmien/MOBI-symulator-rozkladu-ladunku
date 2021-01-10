import numpy as np


def calculateBoundaryConditionsForMatrix(epsilonSi, epsilonSiO, stalaSiatki, tox):
    firstCond = epsilonSi/stalaSiatki + epsilonSiO/tox
    secondCond = -epsilonSi/stalaSiatki
    return firstCond, secondCond


def calculateBoundaryConditionsForVector(epsilonSi, tox, Vg):
    firstCond = epsilonSi/tox * Vg
    secondCond = 0
    return firstCond, secondCond


def generateEyeMatrix(noOfPoints, vectorOnEye):
    eyeOnes = np.eye(noOfPoints)
    eyeWithoutBoundaryConditions = (eyeOnes-2-vectorOnEye)*eyeOnes
    return eyeWithoutBoundaryConditions


def generateGrid(noOfPoints,tsi):
    h = tsi/noOfPoints
    grid = np.linspace(0,tsi,noOfPoints)
    return h, grid


def createFirstApproximation(fun, grid):
    if fun == 'lin':
        array = np.linspace(0, 1, len(grid))
    elif fun == 'poly2':
        array = np.linspace(0, 1, len(grid))**2
    elif fun == 'log':
        array = np.log(np.linspace(0.000001, 1, len(grid)))
    else:  # const
        array = np.zeros(len(grid))
    return array


def calculateDerivativeOfP(charge, h, eSi, Vt, p, n):
    derP = charge*h**2/(eSi*Vt)*(p + n)
    return derP


def calculateP(charge, h, eSi, p, n, N):
    P = -charge*h**2/eSi * (p - n + N)
    return P


def generateMatrix(noOfPoints, eyeMatrix, condPsi1, condPsi2):
    J = np.zeros([noOfPoints, noOfPoints])
    eyeMatrix2 = np.eye(noOfPoints, k=1)
    eyeMatrix3 = np.eye(noOfPoints, k=-1)
    J = J + eyeMatrix2 + eyeMatrix3 + eyeMatrix
    J[noOfPoints-1][noOfPoints-1] = -1
    J[noOfPoints-1][noOfPoints-2] = 1
    J[0][0] = condPsi1
    J[0][1] = condPsi2
    return J


def generateBArray(funPArray, funDerivativeOfPArray, psiArray, epsilonSi, tox, Vg):
    j = 0
    retVal = np.ones(len(psiArray))
    for psi in psiArray:
        retVal[j] = funPArray[j] - funDerivativeOfPArray[j]*psi
        j = j+1
    retVal[0], retVal[len(retVal)-1] = calculateBoundaryConditionsForVector(epsilonSi, tox, Vg)
    return retVal
