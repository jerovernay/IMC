import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from matplotlib import animation

# Configuración visual
plt.style.use('default')

# =====================================================================
# --- 1. NÚCLEO MATEMÁTICO  ---
# =====================================================================

def sistema_kepler(theta, y, alpha, delta):
    """Ecuación diferencial: u'' + u = 1/alpha + delta*u^2"""
    
    u, u_dot = y
    
    return [u_dot, -u + 1/alpha + delta * (u**2)]

def integrar_sistema(epsilon, alpha, delta, vueltas, incompleto=False):
    """
    Resuelve la trayectoria. 
    Para epsilon >= 1 (órbitas abiertas), limita el ángulo para evitar r -> inf.
    """
    
    u0 = (1 + epsilon) / alpha
    y0 = [u0, 0.0] # r_dot(0)=0 implica u_dot(0)=0
    
    if epsilon < 1:
        # Órbitas cerradas (elipses/círculos)
        
        theta_lim = 1.5 * np.pi if incompleto else vueltas * 2 * np.pi
        t_span = (0, theta_lim)
    else:
        # Órbitas abiertas (parábolas/hipérbolas)
        # El límite asintótico es arccos(-1/epsilon)
        
        theta_lim = np.arccos(-1.0 / epsilon) - 0.05
        t_span = (-theta_lim if not incompleto else 0, theta_lim)
        
    t_eval = np.linspace(t_span[0], t_span[1], 3000)
    sol = solve_ivp(sistema_kepler, t_span, y0, t_eval=t_eval, 
                    args=(alpha, delta), rtol=1e-9, atol=1e-9)
    
    return sol.t, sol.y[0], sol.y[1] # theta, u, u_dot


def dibujar_quiver(ax, alpha, delta):
    """Dibuja el campo vectorial de fondo para el espacio de fases."""
    
    u_min, u_max = -0.2, 3.5
    v_min, v_max = -2.5, 2.5
    u_grid, v_grid = np.meshgrid(np.linspace(u_min, u_max, 20),
                                 np.linspace(v_min, v_max, 20))
    du = v_grid
    dv = -u_grid + 1.0/alpha + delta * (u_grid**2)
    M = np.hypot(du, dv)
    M[M == 0] = 1.0
    ax.quiver(u_grid, v_grid, du/M, dv/M, color="lightgray", alpha=0.5, pivot="mid", scale=30)
    ax.set_xlim(u_min, u_max)
    ax.set_ylim(v_min, v_max)


# --- PARÁMETROS BASE ---
alpha_base = 1.0
delta_rel = 0.05

# =====================================================================
# --- 2. PUNTO 2: ESTUDIO CLÁSICO (delta = 0) ---
# =====================================================================

# FIGURA A: Desarrollo Inicial (theta < 2pi)
fig_fase1, axes1 = plt.subplots(1, 3, figsize=(15, 5))
fig_fase1.suptitle(r'Punto 2: Evolución en el Espacio de Fases ($\theta < 2\pi$)', fontsize=16)
eps_vals = [0.0, 0.5, 1.2]
colores = ['#2ca02c', '#9467bd', '#8c564b']

for ax, eps, col in zip(axes1, eps_vals, colores):
    dibujar_quiver(ax, alpha_base, 0.0)
    t, u, ud = integrar_sistema(eps, alpha_base, 0.0, vueltas=1, incompleto=True)
    ax.plot(u, ud, color=col, linewidth=2, label=f'$\epsilon={eps}$')
    ax.plot(u[0], ud[0], 'o', color=col)
    ax.set_title(f'Trayectoria $\epsilon = {eps}$')
    ax.plot(1/alpha_base, 0.0, 'ks', markersize=3,label='Equilibrio (Sol)' )
    ax.set_xlabel('$u$'); ax.set_ylabel('$\dot{u}$')
    ax.legend()


 # FIGURA B: Órbitas Consolidadas (4 vueltas)
fig_fase2, axes2 = plt.subplots(1, 3, figsize=(15, 5))
fig_fase2.suptitle('Punto 2: Estabilidad de Órbitas (4 vueltas)', fontsize=16)
eps_cons = [0.2, 0.6, 1.5]
 
for ax, eps in zip(axes2, eps_cons):
     dibujar_quiver(ax, alpha_base, 0.0)
     t, u, ud = integrar_sistema(eps, alpha_base, 0.0, vueltas=4)
     ax.plot(u, ud, color='#1f77b4')
     ax.set_title(f'Consolidada $\epsilon = {eps}$')
     ax.plot(1/alpha_base, 0.0, 'ks', markersize=3,label='Equilibrio (Sol)' )
     ax.set_xlabel('$u$'); ax.set_ylabel('$\dot{u}$')
     ax.legend()


# =====================================================================
# --- 3. PUNTO 3: COMPARATIVA Y PRECESIÓN ---
# =====================================================================

# FIGURA C: Efecto de la corrección relativista
t_cl, u_cl, ud_cl = integrar_sistema(0.5, alpha_base, 0.0, vueltas=10)
t_re, u_re, ud_re = integrar_sistema(0.5, alpha_base, delta_rel, vueltas=10)

fig_precesion = plt.figure(figsize=(14, 6))
fig_precesion.suptitle(f'Punto 3: Precesión del Perihelio ($\delta = {delta_rel}$)', fontsize=16)

ax_f = fig_precesion.add_subplot(1, 2, 1)
dibujar_quiver(ax_f, alpha_base, delta_rel)
ax_f.plot(u_re, ud_re, 'r', linewidth=1.5, label='Relativista')
ax_f.plot(u_cl, ud_cl, color='black', linestyle='--', alpha=0.6, label='Clásica')
ax_f.set_title("Espacio de Fases ($u, \dot{u}$)")
ax_f.set_xlabel('$u$'); ax_f.set_ylabel('$\dot{u}$')
ax_f.legend(loc='upper right')

# Subplot: Plano Polar
ax_p = fig_precesion.add_subplot(1, 2, 2, projection='polar')
ax_p.plot(t_re, 1/u_re, 'r', linewidth=1.5, label='Precesión Relativista')
ax_p.plot(t_cl[:1000], (1/u_cl)[:1000], color='gray', linestyle='--', label='Referencia Clásica')
ax_p.set_title("Trayectoria en el Plano Polar ($r, \\theta$)")

# Leyenda movida fuera del gráfico para que no tape los datos
ax_p.legend(loc='upper left', bbox_to_anchor=(1.05, 1.0))

plt.tight_layout(rect=[0, 0, 0.95, 0.95]) # Ajuste de márgenes para el título
plt.show()

# =====================================================================
# --- VARIACIÓN DE PARÁMETROS (Grilla Polar Separada) ---
# =====================================================================

fig_var = plt.figure(figsize=(16, 10))
fig_var.suptitle('Punto 3: Variación de Parámetros en Coordenadas Polares', fontsize=16)

# 1. Variando Alpha (Epsilon 0.5)
ax1 = fig_var.add_subplot(2, 3, 1, projection='polar')
for al in [0.5, 0.9, 1.4]:
    t, u, _ = integrar_sistema(0.5, al, delta_rel, vueltas=6)
    ax1.plot(t, 1/u, label=f'$\\alpha={al}$')
ax1.set_title('$\\alpha$ Variable ($\\epsilon=0.5$)')
ax1.legend()

# 2. Variando Epsilon Cerradas (Alpha 1.0)
ax2 = fig_var.add_subplot(2, 3, 2, projection='polar')
for ep in [0.4, 0.8]:
    t, u, _ = integrar_sistema(ep, 1.0, delta_rel, vueltas=6)
    ax2.plot(t, 1/u, label=f'$\\epsilon={ep}$')
ax2.set_title('$\\epsilon < 1$ (Cerradas)')
ax2.legend()

# 3. Variando Epsilon Abiertas (Alpha 1.0)
ax3 = fig_var.add_subplot(2, 3, 3, projection='polar')
for ep in [1.0, 1.3]:
    t, u, _ = integrar_sistema(ep, 1.0, delta_rel, vueltas=1)
    ax3.plot(t, 1/u, label=f'$\\epsilon={ep}$')
ax3.set_title('$\\epsilon \geq 1$ (Abiertas)')
ax3.legend(loc='upper left', bbox_to_anchor=(1.05, 1.0))

# 4. Simultáneo (Cerradas)
ax4 = fig_var.add_subplot(2, 3, 4, projection='polar')
for al, ep in [(0.5, 0.4), (0.9, 0.8)]:
    t, u, _ = integrar_sistema(ep, al, delta_rel, vueltas=6)
    ax4.plot(t, 1/u, label=f'$\\alpha={al}, \\epsilon={ep}$')
ax4.set_title('Simultáneo (Cerradas)')
ax4.legend()

# 5. Simultáneo (Abierta)
ax5 = fig_var.add_subplot(2, 3, 5, projection='polar')
t, u, _ = integrar_sistema(1.3, 1.4, delta_rel, vueltas=1)
ax5.plot(t, 1/u, color='brown', label='$\\alpha=1.4, \\epsilon=1.3$')
ax5.set_title('Simultáneo (Abierta)')
ax5.legend()

plt.tight_layout()
plt.show()

# =====================================================================
# --- 5. ANIMACIÓN 3D (Eje Z = Ángulo Theta) ---
# =====================================================================

# NOTA PARA SPYDER: Si la animación no aparece, ve a: 
# Tools -> Preferences -> IPython Console -> Graphics -> Backend: Automatic
# Reinicia el kernel antes de ejecutar.

fig3d = plt.figure(figsize=(10, 8))
ax3d = fig3d.add_subplot(projection='3d')

# Usamos 12 vueltas para que se note el "resorte"
t_a, u_a, _ = integrar_sistema(0.5, 1.0, 0.05, vueltas=12)
x_a = (1/u_a) * np.cos(t_a)
y_a = (1/u_a) * np.sin(t_a)
z_a = t_a      # El tiempo/ángulo desenrollado

curva, = ax3d.plot([], [], [], 'r-', lw=1.5)
ax3d.set(xlim=(-3,3), ylim=(-3,3), zlim=(0, max(z_a)))
ax3d.set_xlabel('X'); ax3d.set_ylabel('Y'); ax3d.set_zlabel('Ángulo $\\theta$')
ax3d.set_title("Animación: Evolución Temporal (Z) de la Precesión")

def update(i):
    step = i * 20
    curva.set_data(x_a[:step], y_a[:step])
    curva.set_3d_properties(z_a[:step])
    return curva,

ani = animation.FuncAnimation(fig3d, update, frames=len(t_a)//20, interval=25, blit=True)
plt.show()
