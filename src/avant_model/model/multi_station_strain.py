import numpy as np

# -----------------------------
# Geometry / helper functions
# -----------------------------

def C_matrix(x, y, z, a, b, c):
    """Return corners of the cuboid centered at (x,y,z) with semi-axes a,b,c."""
    corners = np.array([
        [x - a, y - b, z - c],
        [x + a, y - b, z - c],
        [x + a, y + b, z - c],
        [x - a, y + b, z - c],
        [x - a, y + b, z + c],
        [x - a, y - b, z + c],
        [x + a, y - b, z + c],
        [x + a, y + b, z + c]
    ])
    return corners

def R_n(n, C):
    return np.linalg.norm(C[n])

def delta(i, j):
    return int(i == j)

deltamod = np.array([[0,1,0],[0,0,1],[1,0,0]])

# -----------------------------
# Displacement-related functions
# -----------------------------

# Numerical note:
# The analytical expressions below contain arctangent terms involving
# ratios of corner-coordinate factors. At exact geometric alignments,
# a denominator can become zero and NumPy may emit a divide-by-zero
# warning while evaluating the intermediate ratio.
#
# A targeted diagnostic over the investigated theta-x0' domain found
# only finite-over-zero cases, with no 0/0 cases and no non-finite
# final strain tensors. The analytical expressions are therefore
# intentionally retained unchanged.
#
# Diagnostic:
# scripts/diagnostics/diagnose_strain_singularities.py

def V0(x, y, z, a, b, c):
    C = C_matrix(x, y, z, a, b, c)
    R_n_list = [R_n(n, C) for n in range(8)]
    result = 0
    for n in range(8):
        for i in range(3):
            result += (
                C[n, i] * C[n, (i+1)%3] * np.log(R_n_list[n] + C[n, (i+2)%3])
                - (C[n, i]**2)/2 * np.arctan((C[n, (i+1)%3]*C[n,(i+2)%3])/(C[n,i]*R_n_list[n]))
            )
        result *= ((-1)**(n+1))
    return result

def v10(x, y, z, a, b, c):
    C = C_matrix(x, y, z, a, b, c)
    R_n_list = [R_n(n, C) for n in range(8)]
    result = np.zeros(3)
    for i in range(3):
        for n in range(8):
            result[i] += ((-1)**(n+1)) * (
                C[n,(i+1)%3]*np.log(R_n_list[n]+C[n,(i+2)%3]) +
                C[n,(i+2)%3]*np.log(R_n_list[n]+C[n,(i+1)%3]) -
                C[n,i]*np.arctan((C[n,(i+1)%3]*C[n,(i+2)%3])/(C[n,i]*R_n_list[n]))
            )
    return result

def v20(x, y, z, a, b, c):
    C = C_matrix(x, y, z, a, b, c)
    R_n_list = [R_n(n, C) for n in range(8)]
    result = np.zeros([3,3])
    for i in range(3):
        for j in range(3):
            for n in range(8):
                result[i,j] += ((-1)**(n+1)) * (
                    deltamod[i,j]*np.log(C[n,(j+1)%3] + R_n_list[n]) +
                    deltamod[j,i]*np.log(C[n,(i+1)%3] + R_n_list[n]) -
                    delta(i,j)*np.arctan((C[n,(i+1)%3]*C[n,(i+2)%3])/(C[n,i]*R_n_list[n]))
                )
    return result

def v30(x, y, z, a, b, c):
    C = C_matrix(x, y, z, a, b, c)
    R_n_list = [R_n(n, C) for n in range(8)]
    result = np.zeros([3,3])
    for i in range(3):
        for j in range(3):
            for n in range(8):
                result[i,j] += ((-1)**(n+1)) * (
                    (deltamod[i,j]*(C[n,2] + delta(i,0)*R_n_list[n]))/(R_n_list[n]*(C[n,(i+2)%3]+R_n_list[n])) +
                    (deltamod[j,i]*(C[n,2] + delta(j,0)*R_n_list[n]))/(R_n_list[n]*(C[n,(j+2)%3]+R_n_list[n])) -
                    delta(i,j)*(C[n,0]*C[n,1]*((1-2*delta(i,2))*(R_n_list[n]**2) - C[n,2]**2)) / 
                    (R_n_list[n]*(C[n,i]**2*R_n_list[n]**2 + C[n,(i+1)%3]**2*C[n,(i+2)%3]**2))
                )
    return result

# -----------------------------
# Material / strain functions
# -----------------------------

def linear_trans(alpha, nu, p, E):
    return ((alpha * (1 - 2 * nu)) * p) / E

def charac_strain(linear_trans_val, nu):
    return (1 / (4 * np.pi)) * ((1 + nu) / (1 - nu)) * linear_trans_val

def disp(x, y, z, a, b, c, ec, h, nu):
    V101 = v10(x, y, z - h, a, b, c)
    V102 = v10(x, y, -z - h, a, b, c)
    v201 = v20(x, y, -z - h, a, b, c)
    result = np.zeros(3)
    for i in range(3):
        result[i] = -ec * (V101[i] + (3 - 4*nu)*V102[i] - 2*(1-2*delta(i,2))*z*v201[i,2])
    return result

def disp_prime(x, y, z, a, b, c, ec, h, nu):
    v201 = v20(x, y, z - h, a, b, c)
    v202 = v20(x, y, -z - h, a, b, c)
    v301 = v30(x, y, -z - h, a, b, c)
    result = np.zeros([3,3])
    for i in range(3):
        for j in range(3):
            result[i,j] = -ec * (
                v201[i,j] +
                ((3-4*nu)*(1-2*delta(j,2)) - 2*(1-2*delta(i,2))*delta(j,2))*v202[i,j] +
                (-2)*(1-2*delta(i,2))*(1-2*delta(j,2))*z*v301[i,j]
            )
    return result

def strain(x, y, z, a, b, c, ec, h, nu):
    local_disp = disp_prime(x, y, z, a, b, c, ec, h, nu)
    result = 0.5 * (local_disp + local_disp.T)
    return result