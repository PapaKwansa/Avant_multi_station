# inclusion_parameterization.py
import numpy as np

def reparameterize_inclusion(a, b, c):
    """(a,b,c) → (V, r1, r2)"""
    V = (4/3) * np.pi * a * b * c
    r1 = a / b
    r2 = a / c
    return V, r1, r2

def deparameterize_inclusion(V, r1, r2):
    """(V, r1, r2) → (a,b,c)"""
    b = 1.0
    a = r1 * b
    c = a / r2

    V0 = (4/3) * np.pi * a * b * c
    scale = (V / V0)**(1/3)

    return a * scale, b * scale, c * scale
