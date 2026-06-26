"""
TP3 - Items 1 y 2 (matriz DENSA).

Item 1: plantear Au = b para f(x)=sen(128*pi*x), alpha=2, beta=1 y mostrar la matriz
        para un n chico (que entre en papel).
Item 2: resolver para n=3..14, error en norma infinito vs la solucion exacta,
        y graficar log(error) vs log(h) para estimar el orden del metodo.
        Graficar tambien perfiles de la solucion para n representativos
        (uno con aliasing y otro con la onda bien resuelta).

Parametro de refinamiento n (el del enunciado): M = 2**n + 1, h = 1/M.
"""

import os
import time
import numpy as np
import matplotlib

matplotlib.use("Agg")  # backend sin ventana
import matplotlib.pyplot as plt

from poisson_common import (
    mesh, build_dense, solve_dense, exact_sin, inf_error, f_sin, save_figure
)

FIGDIR = os.path.join(os.path.dirname(__file__), "figuras")
os.makedirs(FIGDIR, exist_ok=True)

# Datos del problema (items 1-2)
ALPHA, BETA = 2.0, 1.0


# ---------------------------------------------------------------------------
# Item 1: mostrar el sistema para un n chico
# ---------------------------------------------------------------------------
def item1(n=3):
    print("=" * 70)
    print(f"ITEM 1 - Sistema Au=b para f=sen(128*pi*x), alpha={ALPHA}, beta={BETA}, n={n}")
    M, h, x = mesh(n)
    A, b, x = build_dense(n, f_sin, ALPHA, BETA)
    print(f"  M = 2**n + 1 = {M},  h = 1/M = {h:.6f}")
    print(f"  A es {A.shape[0]} x {A.shape[1]}  (1/h^2 = {1/h**2:.3f}, 1/(2h) = {1/(2*h):.3f})")
    np.set_printoptions(precision=2, suppress=True, linewidth=160)
    print("  Matriz A:")
    print(A)
    print("  Vector b:")
    print(b)
    U = solve_dense(A, b)
    err = inf_error(U, x, exact_sin, ALPHA, BETA)
    print(f"  Error ||.||_inf vs solucion exacta (n={n}): {err:.3e}")
    print()


# ---------------------------------------------------------------------------
# Item 2: barrido de error vs h
# ---------------------------------------------------------------------------
def sweep_dense(ns, f, exact, alpha, beta, label):
    hs, errs, kept = [], [], []
    for n in ns:
        try:
            t0 = time.perf_counter()
            A, b, x = build_dense(n, f, alpha, beta)
            U = solve_dense(A, b)
            err = inf_error(U, x, exact, alpha, beta)
            dt = time.perf_counter() - t0
        except MemoryError:
            print(f"  [{label}] n={n}: MemoryError -> se detiene el barrido denso")
            break
        M, h, _ = mesh(n)
        hs.append(h); errs.append(err); kept.append(n)
        print(f"  [{label}] n={n:2d}  M={M:6d}  h={h:.3e}  err={err:.3e}  ({dt:.2f}s)")
    return np.array(kept), np.array(hs), np.array(errs)


def fit_slope(hs, errs):
    """Pendiente de log(err) vs log(h) (ignora errores no positivos)."""
    m = errs > 0
    if m.sum() < 2:
        return float("nan")
    return np.polyfit(np.log(hs[m]), np.log(errs[m]), 1)[0]


def item2():
    print("=" * 70)
    print("ITEM 2 - Error vs h (f=sen(128*pi*x), solucion exacta oscilatoria)")
    ns, hs, errs = sweep_dense(range(3, 15), f_sin, exact_sin, ALPHA, BETA, "f=sen")

    # f=sen(128*pi*x) tiene 64 periodos en [0,1]: por Nyquist hacen falta M>128
    # nodos (n>=7) para no tener aliasing. Los primeros n por encima de Nyquist (n=7,8)
    # todavia estan en transicion; el orden 2 asintotico se mide bien por encima del
    # limite (n>=9, M>=513) hasta el minimo del error (luego subiria por redondeo).
    i0 = int(np.argmax(ns >= 9))         # bien por encima de Nyquist
    imin = int(np.argmin(errs))          # fondo del valle (antes de la subida por redondeo)
    sel = slice(i0, imin + 1)
    slope = fit_slope(hs[sel], errs[sel])
    print(f"  Pendiente en la rama de convergencia (n={ns[i0]}..{ns[imin]}) = {slope:.3f}"
          f"  (esperado ~ +2)")

    plt.figure(figsize=(7, 5))
    plt.loglog(hs, errs, "o-", label="f=sen(128*pi*x)")
    plt.loglog(hs[sel], errs[sel], "o-", color="C2",
               label=f"rama de convergencia (pendiente={slope:.2f})")
    ref = errs[sel][0] * (hs / hs[sel][0]) ** 2
    plt.loglog(hs, ref, "k--", lw=1, label="referencia orden 2")

    out = os.path.join(FIGDIR, "item2_error_vs_h.png")
    save_figure(out, "Item 2: error vs h  (f=sen(128*pi*x))", "h",
                r"$\|u_{num}-u_{ex}\|_\infty$", invert_x=True)
    print()

    # Graficos de soluciones representativas
    plot_solution(n=6, label="aliasing")   # M=65 < 128 (Nyquist): submuestreado
    plot_solution(n=9, label="resuelta")   # M=513: onda bien resuelta

    return ns, hs, errs, slope


def plot_solution(n, label):
    """Grafica la solucion numerica vs exacta para un n especifico."""
    A, b, x = build_dense(n, f_sin, ALPHA, BETA)
    U = solve_dense(A, b)
    u_num = np.concatenate([[ALPHA], U])

    # Para graficar la exacta suavemente, usamos una grilla muy fina
    x_fine = np.linspace(0, 1, 10000)
    u_ex_fine = exact_sin(x_fine, ALPHA, BETA)

    plt.figure(figsize=(8, 4))
    plt.plot(x_fine, u_ex_fine, "-", lw=1, alpha=0.7, label="exacta")
    plt.plot(x, u_num, "o--", ms=3, lw=1, color="red", label=f"numerica (n={n})")

    out = os.path.join(FIGDIR, f"item2_perfil_n{n}.png")
    save_figure(out, f"Perfil u(x) para n={n} ({label})", "x", "u(x)")


if __name__ == "__main__":
    item1(n=3)
    item2()
