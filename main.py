from matplotlib import pyplot as plt
from MOBI_symulator_rozkladu_ladunku.stale import *
from MOBI_symulator_rozkladu_ladunku.poziomyEnergetyczne import *
from MOBI_symulator_rozkladu_ladunku.rownaniaPoissona import *


noOfPoints = 15
noOfIterations = 20
h, grid = generateGrid(noOfPoints, tsi/2)
psiArray = createFirstApproximation('lin', grid)
firstPsi = psiArray
final = np.ones([noOfIterations, len(psiArray)*2])
N = calculateN(Na, Nd)
vt = calculateVt(k, T, q)
p0 = calculateMajorityCarriers(N, ni)
n0 = calculateMinorityCarriers(ni, p0)
pArray = np.ones(len(psiArray)) * p0
nArray = np.ones(len(psiArray)) * n0
funPArray = np.ones(len(psiArray))
funDerivativePArray = np.ones(len(psiArray))
condMatrix1, condMatrix2 = calculateBoundaryConditionsForMatrix(epsilonSi, epsilonSiO, h, tox)


for g in range(noOfIterations):
    i = 0
    for psi in psiArray:
        fip = calculateFermiEnergyP(p=pArray[i], Vt=vt, ni=ni, psi=psi)
        fin = calculateFermiEnergyN(n=nArray[i], Vt=vt, ni=ni, psi=psi)
        pArray[i] = calculateConcentrationOfHoles(fip=fip, psi=psi, vt=vt, ni=ni)
        nArray[i] = calculateConcentrationOfElectrons(fin=fin, psi=psi, vt=vt, ni=ni)
        funDerivativePArray[i] = calculateDerivativeOfP(charge=q, h=h, eSi=epsilonSi, Vt=vt, p=pArray[i], n=nArray[i])
        funPArray[i] = calculateP(charge=q, h=h, eSi=epsilonSi, p=pArray[i], n=nArray[i], N=N)
        i = i + 1

    updatedEye = generateEyeMatrix(noOfPoints=noOfPoints, vectorOnEye=funDerivativePArray)
    J = generateMatrix(noOfPoints=noOfPoints, eyeMatrix=updatedEye, condPsi1=condMatrix1, condPsi2=condMatrix2)
    b = generateBArray(funPArray=funPArray, funDerivativeOfPArray=funDerivativePArray,
                       psiArray=psiArray, epsilonSi=epsilonSi, tox=tox, Vg=Vg)
    psiArray = np.linalg.solve(J, b)
    temp = np.hstack((psiArray, np.flip(psiArray)))
    final[g] = temp

gridToPlot = np.hstack((grid, tsi-np.flip(grid)))
plt.plot(gridToPlot, np.hstack((firstPsi, np.flip(firstPsi))), marker='x', label='input Approx')
for i in range(int(noOfIterations/10)):
    plt.plot(gridToPlot, final[i*10], marker='x', label=f'Approx {i*10+1}')
plt.legend()
plt.title('$\psi$')
plt.show()


