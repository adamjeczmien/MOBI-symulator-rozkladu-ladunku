from matplotlib import pyplot as plt
from MOBI_symulator_rozkladu_ladunku.stale import *
from MOBI_symulator_rozkladu_ladunku.poziomyEnergetyczne import *
from MOBI_symulator_rozkladu_ladunku.rownaniaPoissona import *


def calculateForInputPrediction(funType,noOfPoints,maxError):
    noOfPoints = noOfPoints
    maxPossibleError = maxError
    h, grid = generateGrid(noOfPoints, tsi, tox)
    gridTox = np.linspace(0, tox, 2)
    psiArray = createFirstApproximation(funType, grid, Vg=Vg)
    print(f'Calculating for max epsilon {maxPossibleError} with fun type: {funType}')
    firstPsi = psiArray
    final = []
    N = calculateN(Na, Nd)
    vt = calculateVt(k, T, q)
    p0 = calculateMajorityCarriers(N, ni)
    n0 = calculateMinorityCarriers(ni, p0)
    pArray = np.ones(len(psiArray)) * p0
    nArray = np.ones(len(psiArray)) * n0
    funPArray = np.ones(len(psiArray))
    funDerivativePArray = np.ones(len(psiArray))
    condMatrix1, condMatrix2 = calculateBoundaryConditionsForMatrix(epsilonSi, epsilonSiO, h, tox)
    fip = 0
    fin = 0

    errorList = []
    error = 1
    g = 0

    while error > maxPossibleError:
        i = 0
        for psi in psiArray:
            pArray[i] = calculateConcentrationOfHoles(fip=fip, psi=psi, vt=vt, ni=ni)
            nArray[i] = calculateConcentrationOfElectrons(fin=fin, psi=psi, vt=vt, ni=ni)
            funDerivativePArray[i] = calculateDerivativeOfP(charge=q, h=h, eSi=epsilonSi, Vt=vt, p=pArray[i],
                                                            n=nArray[i])
            funPArray[i] = calculateP(charge=q, h=h, eSi=epsilonSi, p=pArray[i], n=nArray[i], N=N)
            i = i + 1

        updatedEye = generateEyeMatrix(noOfPoints=noOfPoints, vectorOnEye=funDerivativePArray)
        J = generateMatrix(noOfPoints=noOfPoints, eyeMatrix=updatedEye, condPsi1=condMatrix1, condPsi2=condMatrix2)
        b = generateBArray(funPArray=funPArray, funDerivativeOfPArray=funDerivativePArray,
                           psiArray=psiArray, epsilonSi=epsilonSi, tox=tox, Vg=Vg)
        previousPsiArray = psiArray
        psiArray = np.linalg.solve(J, b)
        errors = abs(psiArray - previousPsiArray)
        error = max(errors)
        psiInTox = np.linspace(Vg, psiArray[0], 2)
        temp = np.hstack((psiInTox, psiArray, np.flip(psiArray), np.flip(psiInTox)))
        final.append(temp)
        errorList.append(error)
        g = g + 1
    noOfIterations = g

    gridToPlot = np.hstack((gridTox, grid, tsi + 2 * tox - np.flip(grid), 2 * tox + tsi - np.flip(gridTox)))
    firstAproxTox = np.linspace(Vg, firstPsi[0], 2)
    firstAprox = np.hstack((firstAproxTox, firstPsi, np.flip(firstPsi), np.flip(firstAproxTox)))
    return final, gridToPlot, firstAprox, noOfIterations, errorList


def drawPlots(final, gridToPlot, firstApprox, noOfIterations):
    plt.figure()
    plt.plot(gridToPlot * 1e9, firstApprox, label='Input Approx')
    for i in range(round(noOfIterations / 3)):
        plt.plot(gridToPlot * 1e9, final[i * 3], label=f'Approx {i * 3 + 1}')
    plt.plot(gridToPlot * 1e9, final[len(final) - 1],linestyle='dashed', label=f'Last approx ({noOfIterations})')
    plt.legend()
    plt.title(f'Rozkład potencjału elektrostatycznego w przyrządzie przy początkowym przybliżeniu funkcją {funType}')
    plt.ylabel(f'$\psi$[V]')
    plt.xlabel('nm')
    axes = plt.gca()
    ylim = axes.get_ylim()
    plt.vlines([0, tox * 1e9, (tsi + tox) * 1e9, (tsi + tox * 2) * 1e9], ylim[0], ylim[1], linestyles='dotted',
               colors='k')
    axes.set_ylim(ylim)

########################################################################################################################
noOfPoints = 500
maxError = 1e-14
########################################################################################################################
# linear
funType = 'lin'
final, gridToPlot, firstApprox, noOfIterationsLin, errorListLin = calculateForInputPrediction(funType, noOfPoints, maxError)
drawPlots(final, gridToPlot, firstApprox, noOfIterationsLin)
# poly**2
funType = 'poly2'
final, gridToPlot, firstApprox, noOfIterationsPoly, errorListPoly = calculateForInputPrediction(funType, noOfPoints, maxError)
drawPlots(final, gridToPlot, firstApprox, noOfIterationsPoly)
# const
funType = 'const'
final, gridToPlot, firstApprox, noOfIterationsConst, errorListConst = calculateForInputPrediction(funType, noOfPoints, maxError)
drawPlots(final, gridToPlot, firstApprox, noOfIterationsConst)
# sqrt
funType = 'sqrt'
final, gridToPlot, firstApprox, noOfIterationsSqrt, errorListSqrt = calculateForInputPrediction(funType, noOfPoints, maxError)
drawPlots(final, gridToPlot, firstApprox, noOfIterationsSqrt)
plt.figure()
plt.semilogy(np.linspace(1, noOfIterationsLin, noOfIterationsLin), errorListLin, label='Przybliżenie liniową')
plt.semilogy(np.linspace(1, noOfIterationsPoly, noOfIterationsPoly), errorListPoly, label='Przybliżenie poly2')
plt.semilogy(np.linspace(1,noOfIterationsConst,noOfIterationsConst), errorListConst, label='Przybliżenie stałą')
plt.semilogy(np.linspace(1,noOfIterationsSqrt,noOfIterationsSqrt), errorListSqrt, label='Przybliżenie pierwiastkiem')
plt.title('Maksymalny błąd bezwzględny w porównaniu z poprzednią iteracją')
plt.xlabel('Numer iteracji')
plt.ylabel('Błąd [V]')
plt.legend()
plt.grid()
plt.show()


