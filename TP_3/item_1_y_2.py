"""
TP3 - Items 1 y 2 (matriz DENSA).

Item 1: plantear Au = b para f(x)=sen(128*pi*x), alpha=2, beta=1 y mostrar la matriz.
Item 2: resolver para k=3..14, error en norma infinito vs la solucion exacta,
        y graficar log(error) vs log(h) para estimar el orden.
        Graficar tambien perfiles de la solucion para k representativos (aliasing y onda resuelta).
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
# Item 1: mostrar el sistema para un k chico
# ---------------------------------------------------------------------------
def item1(k=3):
    print("=" * 70)
    print(f"ITEM 1 - Sistema Au=b para f=sen(128*pi*x), alpha={ALPHA}, beta={BETA}, k={k}")
    n, M, h, x = mesh(k)
    A, b, x = build_dense(k, f_sin, ALPHA, BETA)
    print(f"  n = 2**{k} = {n},  M = 2n+1 = {M},  h = 1/M = {h:.6f}")
    print(f"  A es {A.shape[0]} x {A.shape[1]}  (1/h^2 = {1/h**2:.3f}, 1/(2h) = {1/(2*h):.3f})")
    np.set_printoptions(precision=2, suppress=True, linewidth=160)
    print("  Matriz A:")
    print(A)
    print("  Vector b:")
    print(b)
    U = solve_dense(A, b)
    err = inf_error(U, x, exact_sin, ALPHA, BETA)
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
    print("ITEM 2 - Error vs h (f=sen(128*pi*x), solucion exacta oscilatoria)")
    ks, hs, errs = sweep_dense(range(3, 13), f_sin, exact_sin, ALPHA, BETA, "f=sen")
    
    # Pendiente en la zona de convergencia (cuando k es suficiente para evitar aliasing, ej. k>=9)
    valid_mask = ks >= 9
    slope = fit_slope(hs[valid_mask], errs[valid_mask])
    print(f"  Pendiente asintotica log(err)/log(h) (k>=9) = {slope:.3f}")

    plt.figure(figsize=(7, 5))
    plt.loglog(hs, errs, "o-", label=f"f=sen(128*pi*x)")
    if not np.isnan(slope):
        ref = errs[valid_mask][0] * (hs / hs[valid_mask][0]) ** 2
        plt.loglog(hs, ref, "k--", lw=1, label=f"referencia orden 2")
    
    out = os.path.join(FIGDIR, "item2_error_vs_h.png")
    save_figure(out, "Item 2: error vs h  (f=sen(128*pi*x))", "h", r"$\|u_{num}-u_{ex}\|_\infty$", invert_x=True)
    print()

    # Graficos de soluciones representativas
    plot_solution(k=5, label="aliasing") # submuestreado (M=65) < 128 (Nyquist)
    plot_solution(k=8, label="resuelta") # bien resuelto (M=513)

    return ks, hs, errs, slope


def plot_solution(k, label):
    """Grafica la solucion numerica vs exacta para un k especifico."""
    A, b, x = build_dense(k, f_sin, ALPHA, BETA)
    U = solve_dense(A, b)
    u_num = np.concatenate([[ALPHA], U])
    
    # Para graficar la exacta suavemente, usamos una grilla muy fina
    x_fine = np.linspace(0, 1, 10000)
    u_ex_fine = exact_sin(x_fine, ALPHA, BETA)
    
    plt.figure(figsize=(8, 4))
    plt.plot(x_fine, u_ex_fine, "-", lw=1, alpha=0.7, label="exacta")
    plt.plot(x, u_num, "o--", ms=3, lw=1, color="red", label=f"numerica (k={k})")
    
    out = os.path.join(FIGDIR, f"item2_perfil_k{k}.png")
    save_figure(out, f"Perfil u(x) para k={k} ({label})", "x", "u(x)")


if __name__ == "__main__":
    item1(k=3)
    item2()
