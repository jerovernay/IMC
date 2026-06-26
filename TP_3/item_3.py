"""
TP3 - Item 3 (matrices ESPARSAS).

Repetimos el planteo para f(x)=e^{-x^2}, alpha=2, beta=1, pero armando A como
matriz esparsa (scipy.sparse) y resolviendo con spsolve.

Idea (igual que en el ejemplo de la catedra): con almacenamiento esparso A ocupa
O(M) (~3 entradas por fila) en lugar de O(M^2), asi que se puede subir mucho n.
Subimos n de a uno hasta que spsolve falla; ese n-1 es el mayor que resuelve esta
maquina. Conviene correrlo en una maquina local (no en un cluster) para que el
limite aparezca pronto.

Parametro de refinamiento n (el del enunciado): M = 2**n + 1, h = 1/M.
"""

import os
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from poisson_common import (
    mesh, build_sparse, solve_sparse, exact_erf, inf_error, f_erf, save_figure
)

FIGDIR = os.path.join(os.path.dirname(__file__), "figuras")
os.makedirs(FIGDIR, exist_ok=True)

ALPHA, BETA = 2.0, 1.0

NMAX = 26   # hasta donde intentamos subir el refinamiento


def item3():
    print("=" * 70)
    print("ITEM 3 - f=e^{-x^2}, alpha=2, beta=1, matrices esparsas")
    ns, hs, errs = [], [], []
    last_ok = None

    for n in range(3, NMAX + 1):
        M, h, _ = mesh(n)
        try:
            A, b, x = build_sparse(n, f_erf, ALPHA, BETA)
            U = solve_sparse(A, b)
        except (MemoryError, RuntimeError) as e:
            # spsolve/SuperLU se queda sin memoria al factorizar (SUPERLU_MALLOC).
            # OJO: NO es fill-in. La matriz es casi tridiagonal y el factor LU casi no
            # se llena (medido: nnz(L)+nnz(U) ~ 1.33 * nnz(A), constante con n). El limite
            # es la memoria total que SuperLU necesita para armar y guardar la LU mas su
            # bookkeeping: es O(M) pero con una constante grande, y a M~16.7M agota la RAM.
            print(f"  n={n}: spsolve fallo ({type(e).__name__}) -> M={M} es el limite.")
            break
        err = inf_error(U, x, exact_erf, ALPHA, BETA)
        ns.append(n); hs.append(h); errs.append(err); last_ok = n
        print(f"  n={n:2d}  M={M:8d}  nnz={A.nnz:9d}  err={err:.3e}")

    print(f"\n  Mayor n resuelto en esta maquina: n = {last_ok}")
    print("  El cuello de botella es la memoria que spsolve/SuperLU necesita para la")
    print("  factorizacion LU (y su bookkeeping), no el fill-in: A es casi tridiagonal")
    print("  y la LU casi no se llena (nnz(LU) ~ 1.33*nnz(A)). Es O(M) pero con constante")
    print("  grande y a M~16.7M agota la RAM. Ademas, pasado cierto n el error deja de")
    print("  bajar y vuelve a subir por el redondeo amplificado (~h^{-2}).")

    ns = np.array(ns); hs = np.array(hs); errs = np.array(errs)

    # La curva error vs h baja (orden 2) hasta un minimo y luego sube (redondeo).
    # Ajustamos la pendiente SOLO en la rama de convergencia (hasta el minimo).
    nmin_idx = int(np.argmin(errs))
    hc, ec = hs[:nmin_idx + 1], errs[:nmin_idx + 1]
    slope = np.polyfit(np.log(hc), np.log(ec), 1)[0] if len(hc) >= 2 else float("nan")
    print(f"\n  Pendiente en la rama de convergencia (n=3..{ns[nmin_idx]}) = {slope:.2f}"
          f"  (esperado ~ +2)")

    # Figura 1: convergencia (orden 2) y piso/subida de redondeo
    plt.figure(figsize=(7, 5))
    plt.loglog(hs, errs, "o-", label="f=e^(-x^2)")
    plt.loglog(hc, ec, "o-", color="C2",
               label=f"rama de convergencia (pendiente={slope:.2f})")
    ref = ec[0] * (hc / hc[0]) ** 2
    plt.loglog(hc, ref, "k--", lw=1, label="referencia orden 2")
    plt.annotate("redondeo ~h$^{-2}$", xy=(hs[-1], errs[-1]),
                 xytext=(hs[-1] * 6, errs[-1] * 0.4),
                 arrowprops=dict(arrowstyle="->", lw=1))

    out1 = os.path.join(FIGDIR, "item3_error_vs_h.png")
    save_figure(out1, "Item 3: error vs h (esparso, f=e^{-x^2})", "h",
                r"$\|u_{num}-u_{ex}\|_\infty$", invert_x=True)
    print()

    # Figura 2: perfil de la solucion para el mayor n resuelto
    n_p = last_ok
    print(f"\n  Generando perfil de la solucion para n={n_p}...")
    A, b, x = build_sparse(n_p, f_erf, ALPHA, BETA)
    U = solve_sparse(A, b)
    u_num = np.concatenate([[ALPHA], U])

    # Submuestrear para no crashear la memoria al graficar millones de puntos
    step = max(1, len(x) // 5000)
    x_plot = x[::step]
    u_num_plot = u_num[::step]

    plt.figure(figsize=(7, 5))
    plt.plot(x_plot, exact_erf(x_plot, ALPHA, BETA), "-", lw=2, label="exacta (erf)")
    plt.plot(x_plot, u_num_plot, "--", lw=1.5, label=f"numerica (n={n_p}, submuestreo)")

    out2 = os.path.join(FIGDIR, "item3_perfil.png")
    save_figure(out2, "Item 3: perfil de la solucion, f=e^{-x^2}", "x", "u(x)")
    return last_ok, slope


if __name__ == "__main__":
    item3()
