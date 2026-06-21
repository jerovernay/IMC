"""
TP3 - Items 1 y 2 (matriz DENSA).

Item 1: plantear Au = b para f(x)=3, alpha=2, beta=1 y mostrar la matriz.
Item 2: resolver para k=3..14, error en norma infinito vs la solucion exacta,
        y graficar log(error) vs log(h) para estimar el orden.

Complemento: barrido con una solucion no polinomica (u=e^x) para exhibir el
orden 2 real del metodo (en el caso f=3 la solucion exacta es un polinomio de
grado 2 y el esquema la reproduce sin error de truncamiento: lo que se observa
es redondeo amplificado por el condicionamiento ~ h^{-2}).
"""

import os
import time
import numpy as np
import matplotlib

matplotlib.use("Agg")  # backend sin ventana
import matplotlib.pyplot as plt

from poisson_common import (
    mesh, build_dense, solve_dense, exact_quadratic, inf_error,
)

FIGDIR = os.path.join(os.path.dirname(__file__), "figuras")
os.makedirs(FIGDIR, exist_ok=True)

# Datos del problema (items 1-2)
ALPHA, BETA = 2.0, 1.0
f3 = lambda x: np.full_like(np.asarray(x, dtype=float), 3.0)


# ---------------------------------------------------------------------------
# Item 1: mostrar el sistema para un k chico
# ---------------------------------------------------------------------------
def item1(k=3):
    print("=" * 70)
    print(f"ITEM 1 - Sistema Au=b para f=3, alpha={ALPHA}, beta={BETA}, k={k}")
    n, M, h, x = mesh(k)
    A, b, x = build_dense(k, f3, ALPHA, BETA)
    print(f"  n = 2**{k} = {n},  M = 2n+1 = {M},  h = 1/M = {h:.6f}")
    print(f"  A es {A.shape[0]} x {A.shape[1]}  (1/h^2 = {1/h**2:.3f}, 1/(2h) = {1/(2*h):.3f})")
    np.set_printoptions(precision=2, suppress=True, linewidth=160)
    print("  Matriz A:")
    print(A)
    print("  Vector b:")
    print(b)
    # verificacion: la solucion exacta cumple A u = b (esquema exacto p/ grado 2)
    U = solve_dense(A, b)
    err = inf_error(U, x, exact_quadratic, ALPHA, BETA)
    print(f"  Error ||.||_inf vs solucion exacta (k={k}): {err:.3e}")
    print()


# ---------------------------------------------------------------------------
# Item 2: barrido de error vs h
# ---------------------------------------------------------------------------
def sweep_dense(ks, f, exact, alpha, beta, label):
    hs, errs, kept = [], [], []
    for k in ks:
        try:
            t0 = time.perf_counter()
            A, b, x = build_dense(k, f, alpha, beta)
            U = solve_dense(A, b)
            err = inf_error(U, x, exact, alpha, beta)
            dt = time.perf_counter() - t0
        except MemoryError:
            print(f"  [{label}] k={k}: MemoryError -> se detiene el barrido denso")
            break
        n, M, h, _ = mesh(k)
        hs.append(h); errs.append(err); kept.append(k)
        print(f"  [{label}] k={k:2d}  M={M:6d}  h={h:.3e}  err={err:.3e}  ({dt:.2f}s)")
    return np.array(kept), np.array(hs), np.array(errs)


def fit_slope(hs, errs):
    """Pendiente de log(err) vs log(h) (ignora errores no positivos)."""
    m = errs > 0
    if m.sum() < 2:
        return float("nan")
    return np.polyfit(np.log(hs[m]), np.log(errs[m]), 1)[0]


def item2():
    print("=" * 70)
    print("ITEM 2 - Error vs h (f=3, solucion exacta cuadratica)")
    ks, hs, errs = sweep_dense(range(3, 15), f3, exact_quadratic, ALPHA, BETA, "f=3")
    slope = fit_slope(hs, errs)
    print(f"  Pendiente global log(err)/log(h) = {slope:.3f}")
    print("  (pendiente ~ -2: el error CRECE al refinar => dominado por redondeo,")
    print("   no por truncamiento; el esquema es exacto para el polinomio de grado 2)")

    plt.figure(figsize=(7, 5))
    plt.loglog(hs, errs, "o-", label=f"f=3  (pendiente={slope:.2f})")
    plt.gca().invert_xaxis()  # h decreciente hacia la derecha
    plt.xlabel("h"); plt.ylabel(r"$\|u_{num}-u_{ex}\|_\infty$")
    plt.title("Item 2: error vs h  (f=3, sol. exacta cuadratica)")
    plt.grid(True, which="both", ls=":"); plt.legend()
    out = os.path.join(FIGDIR, "item2_error_vs_h.png")
    plt.tight_layout(); plt.savefig(out, dpi=130); plt.close()
    print(f"  Figura guardada: {out}")
    print()
    return ks, hs, errs, slope


def item2_convergencia():
    """Complemento: solucion manufacturada u=e^x (no polinomica) -> orden 2 real."""
    print("=" * 70)
    print("ITEM 2 (complemento) - Convergencia con u(x)=e^x  (f=e^x)")
    fexp = lambda x: np.exp(x)
    a, b_ = 1.0, np.e               # u(0)=1, u'(1)=e
    exact = lambda x, alpha, beta: np.exp(x)
    ks, hs, errs = sweep_dense(range(3, 13), fexp, exact, a, b_, "u=e^x")
    slope = fit_slope(hs, errs)
    print(f"  Pendiente global log(err)/log(h) = {slope:.3f}  (esperado ~ +2)")

    plt.figure(figsize=(7, 5))
    plt.loglog(hs, errs, "s-", color="C1", label=f"u=e^x  (pendiente={slope:.2f})")
    ref = errs[0] * (hs / hs[0]) ** 2
    plt.loglog(hs, ref, "k--", lw=1, label="referencia orden 2")
    plt.gca().invert_xaxis()
    plt.xlabel("h"); plt.ylabel(r"$\|u_{num}-u_{ex}\|_\infty$")
    plt.title("Item 2 (complemento): orden 2 real con sol. no polinomica")
    plt.grid(True, which="both", ls=":"); plt.legend()
    out = os.path.join(FIGDIR, "item2_convergencia.png")
    plt.tight_layout(); plt.savefig(out, dpi=130); plt.close()
    print(f"  Figura guardada: {out}")
    print()
    return ks, hs, errs, slope


if __name__ == "__main__":
    item1(k=3)
    item2()
    item2_convergencia()
