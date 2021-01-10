import math as m


def calculateN(Na,Nd):
    N = Nd - Na
    return N


def calculateMajorityCarriers(N, ni):
    if N < 0:
        retVal = m.sqrt((N/2)**2 + ni**2) - N/2
    else:
        retVal = m.sqrt((N/2)**2 + ni**2) + N/2
    return retVal


def calculateMinorityCarriers(ni, majorityCarriers):
    retVal = (ni**2) / majorityCarriers
    return retVal


def calculateFermiEnergyP(p, Vt, ni, psi):
    fiP = psi + Vt*m.log(p/ni)
    return fiP


def calculateFermiEnergyN(n, Vt, ni, psi):
    fiN = psi - Vt*m.log(n/ni)
    return fiN


def calculateConcentrationOfHoles(fip, psi, vt,  ni):
    p = ni * m.exp((fip-psi)/vt)
    return p


def calculateConcentrationOfElectrons(fin, psi, vt,  ni):
    n = ni * m.exp((psi-fin)/vt)
    return n


def calculatePotentialInGate(q, Vg):
    retVal = -q*Vg
    return retVal


def calculateVt(k, T, q):
    retVal = k*T/q
    return retVal
