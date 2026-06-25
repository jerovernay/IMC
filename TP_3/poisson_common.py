"""
TP3 - Ecuacion de Poisson 1D por diferencias finitas.

Problema:
    u''(x) = f(x),  x en (0,1)
    u(0)   = alfa            (Dirichlet)
    u'(1)  = beta            (Neumann)

Esquema (igual idea que el ejemplo de la catedra, pero con Neumann en x=1):
    - Nodos interiores: diferencias centradas de orden 2 para u''
        u''(x[j]) ~ ( u[j-1] - 2 u[j] + u[j+1] ) / h^2 = f(x[j])
    - Borde x=1 (Neumann): diferencia backward de orden 2 a 3 nodos
        u'(1) ~ ( u[M-2] - 4 u[M-1] + 3 u[M] ) / (2h) = beta

Malla:
    n = 2**k            (parametro de refinamiento, EDITABLE)
    M = 2*n + 1
    h = 1 / M
    x[j] = h*j,  j = 0,...,M   ->  x[0] = 0,  x[M] = 1

Como u[0] = alfa es dato, no es incognita. Las incognitas son u[1],...,u[M],
asi que el sistema lineal A u = b tiene M ecuaciones y M incognitas.
Para que el indice de A no quede corrido, usamos U[i] = u[i+1] (i = 0,...,M-1):
    fila i interior  -> ecuacion del nodo j = i+1   (1 <= j <= M-1)
    fila i = M-1     -> condicion de Neumann en j = M
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import solve                 # para matrices densas
from scipy.sparse import diags_array           # para matrices esparzas (ralas)
from scipy.sparse.linalg import spsolve        # solver para matrices esparzas
from scipy.special import erf


# ---------------------------------------------------------------------------
# Malla
# ---------------------------------------------------------------------------
def mesh(k):
    """Devuelve (n, M, h, x) para el indice de refinamiento k.  n = 2**k."""
    n = 2 ** k
    M = 2 * n + 1
    h = 1.0 / M
    x = np.array([h * j for j in range(M + 1)])   # x[0]=0, ..., x[M]=1
    return n, M, h, x


# ---------------------------------------------------------------------------
# Lado derecho b (longitud M)
# ---------------------------------------------------------------------------
def _rhs(M, h, x, f, alpha, beta):
    """
    b[i] corresponde a la ecuacion del nodo j = i+1.
    En j=1 el dato u[0]=alfa pasa al lado derecho: b[0] -= alfa/h^2.
    La ultima fila (i = M-1) es la condicion de Neumann: b[M-1] = beta.
    """
    b = np.zeros(M)
    b[:M - 1] = f(x[1:M])            # f(x[1]),...,f(x[M-1])
    b[0] -= alpha / h ** 2          # u[0] = alfa pasa al RHS en j=1
    b[M - 1] = beta                 # condicion de Neumann en x=1
    return b


# ---------------------------------------------------------------------------
# Ensamblado de A:  tridiagonal de las diferencias centradas + fila de Neumann.
# La fila de Neumann (ultima) rompe la tridiagonal: la sobrescribimos a mano.
# ---------------------------------------------------------------------------
def build_dense(k, f, alpha, beta):
    """Construye (A, b, x) con A densa (np.ndarray M x M)."""
    n, M, h, x = mesh(k)

    # Tridiagonal de diferencias centradas: -2 en la diagonal, +1 en las codiagonales.
    A = (np.diag(-2.0 * np.ones(M))
         + np.diag(np.ones(M - 1), +1)
         + np.diag(np.ones(M - 1), -1)) / h ** 2

    # Ultima fila = Neumann backward de orden 2:  (1, -4, 3)/(2h).
    A[M - 1, :] = 0.0
    A[M - 1, M - 3] = 1.0 / (2.0 * h)
    A[M - 1, M - 2] = -4.0 / (2.0 * h)
    A[M - 1, M - 1] = 3.0 / (2.0 * h)

    b = _rhs(M, h, x, f, alpha, beta)
    return A, b, x


def build_sparse(k, f, alpha, beta):
    """Construye (A, b, x) con A esparza (scipy.sparse, formato CSC para spsolve)."""
    n, M, h, x = mesh(k)

    # Misma tridiagonal pero como matriz esparza. offsets=[0,1,-1] = diag, super, sub.
    # Usamos formato "lil" (List of Lists) inicialmente: SciPy recomienda este formato 
    # fuertemente cuando se necesita construir o modificar una matriz esparsa entrada
    # por entrada (como haremos con la fila de Neumann). Construirla directo en CSR/CSC
    # resulta sumamente ineficiente.
    A = diags_array(
        [-2.0 * np.ones(M), np.ones(M - 1), np.ones(M - 1)],
        offsets=[0, 1, -1], shape=(M, M), format="lil",
    ) / h ** 2

    # Sobrescribimos la ultima fila con la condicion de Neumann (formato lil lo permite).
    A[M - 1, M - 3] = 1.0 / (2.0 * h)
    A[M - 1, M - 2] = -4.0 / (2.0 * h)
    A[M - 1, M - 1] = 3.0 / (2.0 * h)

    b = _rhs(M, h, x, f, alpha, beta)
    return A.tocsc(), b, x


# ---------------------------------------------------------------------------
# Solvers
# ---------------------------------------------------------------------------
def solve_dense(A, b):
    return solve(A, b)


def solve_sparse(A, b):
    return spsolve(A, b)


# ---------------------------------------------------------------------------
# Soluciones exactas y funciones RHS f(x)
# ---------------------------------------------------------------------------
def exact_quadratic(x, alpha, beta):
    """Solucion exacta para f(x)=3:  u(x) = 1.5 x^2 + (beta-3) x + alfa."""
    return 1.5 * x ** 2 + (beta - 3.0) * x + alpha


f_erf = lambda x: np.exp(-x ** 2)

def exact_erf(x, alpha, beta):
    """
    Solucion exacta para f(x)=e^{-x^2}:
        u'(x) = C1 + (sqrt(pi)/2) erf(x),   C1 = beta - (sqrt(pi)/2) erf(1)
        u(x)  = alfa + C1 x + (sqrt(pi)/2)[ x erf(x) + (e^{-x^2}-1)/sqrt(pi) ]
    """
    s = np.sqrt(np.pi) / 2.0
    C1 = beta - s * erf(1.0)
    return alpha + C1 * x + s * (x * erf(x) + (np.exp(-x ** 2) - 1.0) / np.sqrt(np.pi))


f_sin = lambda x: np.sin(128.0 * np.pi * np.asarray(x, dtype=float))

def exact_sin(x, alpha, beta):
    """
    Solucion exacta para f(x)=sen(128*pi*x):
        u(x) = -1/(128*pi)^2 * sen(128*pi*x) + (beta + 1/(128*pi)) * x + alpha
    """
    w = 128.0 * np.pi
    C = beta + 1.0 / w
    return -1.0 / (w**2) * np.sin(w * x) + C * x + alpha


# ---------------------------------------------------------------------------
# Error en norma infinito
# ---------------------------------------------------------------------------
def inf_error(U, x, exact, alpha, beta):
    """
    Error en norma infinito sobre todos los nodos.
    U son las incognitas (u[1]..u[M]); se antepone u[0] = alfa (exacto).
    """
    u_num = np.concatenate([[alpha], U])
    u_ex = exact(x, alpha, beta)
    return np.max(np.abs(u_num - u_ex))


# ---------------------------------------------------------------------------
# Utilidad de Matplotlib
# ---------------------------------------------------------------------------
def save_figure(out_path, title, xlabel, ylabel, invert_x=False):
    """Aplica configuracion estandar de graficos y lo guarda."""
    if invert_x:
        plt.gca().invert_xaxis()
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True, which="both", ls=":")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=130)
    plt.close()
    print(f"  Figura guardada: {out_path}")
