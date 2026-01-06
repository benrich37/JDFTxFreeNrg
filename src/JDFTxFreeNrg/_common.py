import numpy as np

def dagger(mat: np.ndarray) -> np.ndarray:
    return mat.conj().T
def invsqrt(mat: np.ndarray) -> np.ndarray:
    eigs, evecs = np.linalg.eigh(mat)
    invsqrt_eigs = np.diag(1.0 / np.sqrt(eigs))
    return evecs @ invsqrt_eigs @ dagger(evecs)
def pow(val, exp):
    return val**exp

def sqrt(val):
    return np.sqrt(val)

def hypot(a, b):
    return np.hypot(a, b)

def cis(val: complex) -> complex:
    return np.cos(val) + 1j * np.sin(val)

def box(a, b, c):
    return np.dot(a, np.cross(b, c))

def dot(a, b):
    return np.dot(a, b)

def remove_phase(N: int, vec: np.ndarray, threshold: float = 1e-300) -> np.ndarray:
    w0 = 0.0
    r1 = 0.0
    r2 = 0.0
    i1 = 0.0
    i2 = 0.0
    for i in range(N):
        c = vec[i]*vec[i]
        w = np.abs(c)
        if w > threshold:
            w0 += w
            r1 += c.real
            i1 += c.imag
            r2 += (c.real)**2/w
            i2 += (c.imag)**2/w
    rMean = r1/w0
    iMean = i1/w0
    rSigma = sqrt(max(0.0, r2/w0 - pow(rMean, 2)))
    iSigma = sqrt(max(0.0, i2/w0 - pow(iMean, 2)))
    meanPhase = 0.5*np.arctan2(iMean, rMean)
    sigmaPhase = 0.5*hypot(iMean*rSigma, rMean*iSigma)/(pow(rMean, 2) + pow(iMean, 2))
    phaseCompensate = cis(-meanPhase)
    imSqSum = 0.0
    reSqSum = 0.0
    for i in range(N):
        c = vec[i]*phaseCompensate
        reSqSum += pow(c.real, 2)
        imSqSum += pow(c.imag, 2)
        vec[i] = c.real
    rmsImagErr = sqrt(imSqSum/(imSqSum + reSqSum))
    return vec