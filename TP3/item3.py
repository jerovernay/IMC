"""
TP3 - Item 3 (matrices ESPARSAS).

Repetimos el planteo para f(x)=e^{-x^2}, alpha=2, beta=1, pero armando A como
matriz esparsa (scipy.sparse) y resolviendo con spsolve.

Idea (igual que en el ejemplo de la catedra): con almacenamiento esparso A ocupa
O(M) (~3 entradas por fila) en lugar de O(M^2), asi que se puede subir mucho k.
Subimos k de a uno hasta que spsolve falla; ese k-1 es el mayor que resuelve esta
maquina. Probar valores mayores y prestar atencion al mensaje de error.
"""

import os
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from poisson_common import (
    mesh, build_sparse, solve_sparse, exact_erf, inf_error,
)

FIGDIR = os.path.join(os.path.dirname(__file__), "figuras")
os.makedirs(FIGDIR, exist_ok=True)

ALPHA, BETA = 2.0, 1.0
fexp = lambda x: np.exp(-x ** 2)

KMAX = 24   # hasta donde intentamos subir el refinamiento


def item3():
    print("=" * 70)
    print("ITEM 3 - f=e^{-x^2}, alpha=2, beta=1, matrices esparsas")
    ks, hs, errs = [], [], []
    last_ok = None

    for k in range(3, KMAX + 1):
        n, M, h, _ = mesh(k)
        try:
            A, b, x = build_sparse(k, fexp, ALPHA, BETA)
            U = solve_sparse(A, b)
        except (MemoryError, RuntimeError) as e:
            # spsolve/SuperLU se queda sin memoria al factorizar (SUPERLU_MALLOC):
            # el factor LU se llena (fill-in) y es mucho mas denso que A.
            print(f"  k={k}: spsolve fallo ({type(e).__name__}) -> M={M} es el limite.")
            break
        err = inf_error(U, x, exact_erf, ALPHA, BETA)
        ks.append(k); hs.append(h); errs.append(err); last_ok = k
        print(f"  k={k:2d}  M={M:8d}  nnz={A.nnz:9d}  err={err:.3e}")

    print(f"\n  Mayor k resuelto en esta maquina: k = {last_ok}")
    print("  El cuello de botella NO es guardar A (es O(M)), sino el fill-in de la")
    print("  factorizacion LU dentro de spsolve. Ademas, a partir de ~k=14 el error")
    print("  deja de bajar y vuelve a subir por el redondeo amplificado (~h^{-2}).")

    ks = np.array(ks); hs = np.array(hs); errs = np.array(errs)

    # La curva error vs h baja (orden 2) hasta un minimo y luego sube (redondeo).
    # Ajustamos la pendiente SOLO en la rama de convergencia (hasta el minimo).
    kmin_idx = int(np.argmin(errs))
    hc, ec = hs[:kmin_idx + 1], errs[:kmin_idx + 1]
    slope = np.polyfit(np.log(hc), np.log(ec), 1)[0] if len(hc) >= 2 else float("nan")
    print(f"\n  Pendiente en la rama de convergencia (k=3..{ks[kmin_idx]}) = {slope:.2f}"
          f"  (esperado ~ +2)")

    # Figura 1: convergencia (orden 2) y piso/subida de redondeo
    plt.figure(figsize=(7, 5))
    plt.loglog(hs, errs, "o-", label="f=e^(-x^2)")
    plt.loglog(hc, ec, "o-", color="C2",
               label=f"rama de convergencia (pendiente={slope:.2f})")
    ref = ec[0] * (hc / hc[0]) ** 2
    plt.loglog(hc, ref, "k--", lw=1, label="referencia orden 2")
    plt.gca().invert_xaxis()
    plt.xlabel("h"); plt.ylabel(r"$\|u_{num}-u_{ex}\|_\infty$")
    plt.title("Item 3: error vs h (esparso, f=e^{-x^2})")
    plt.annotate("redondeo ~h$^{-2}$", xy=(hs[-1], errs[-1]),
                 xytext=(hs[-1] * 6, errs[-1] * 0.4),
                 arrowprops=dict(arrowstyle="->", lw=1))
    plt.grid(True, which="both", ls=":"); plt.legend()
    out1 = os.path.join(FIGDIR, "item3_error_vs_h.png")
    plt.tight_layout(); plt.savefig(out1, dpi=130); plt.close()
    print(f"\n  Figura guardada: {out1}")

    # Figura 2: perfil de la solucion para un k moderado
    kp = 8
    A, b, x = build_sparse(kp, fexp, ALPHA, BETA)
    U = solve_sparse(A, b)
    u_num = np.concatenate([[ALPHA], U])
    plt.figure(figsize=(7, 5))
    plt.plot(x, exact_erf(x, ALPHA, BETA), "-", lw=2, label="exacta (erf)")
    plt.plot(x, u_num, "--", lw=1.5, label=f"numerica (k={kp})")
    plt.xlabel("x"); plt.ylabel("u(x)")
    plt.title("Item 3: perfil de la solucion, f=e^{-x^2}")
    plt.grid(True, ls=":"); plt.legend()
    out2 = os.path.join(FIGDIR, "item3_perfil.png")
    plt.tight_layout(); plt.savefig(out2, dpi=130); plt.close()
    print(f"  Figura guardada: {out2}")
    return last_ok, slope


if __name__ == "__main__":
    item3()
