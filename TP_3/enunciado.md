# Resolución numérica de una ecuación de Poisson unidimensional.

Se desea resolver numéricamente la ecuación de Poisson en una dimensión con condiciones de Dirichlet en $x = 0$ y de Neumann en $x = 1$.
$$
\begin{cases} u_{xx}(x) = f(x) \text{ para } x \in (0,1) \\ u(0) = \alpha \\ u_x(1) = \beta \end{cases} \tag{1}
$$

1. Plantear el problema como un sistema lineal $Au = b$, para $f(x) = \text{sen}(128\pi x)$, $\alpha = 2$, $\beta = 1$ en una malla uniforme $\{x_j = hj\}$ con $j = 0, 1, 2, ..., 2^n + 1$ y $h = \frac{1}{2^n + 1}$. <p>Construir la matriz A de diferenciación utilizando el esquema de diferencias centradas para la derivada segunda y diferencias backward de orden 2 para la condición de Neumann (es la aproximación del ejercicio 2(a)-2(b) para 3 nodos). Mostrar las fórmulas de las aproximaciones usadas para las derivadas. Mostrar algunas matrices $A$ y $b$ con $n$ chico (que entre en papel).


2. Resolver y calcular el error en $\| \cdot \|_\infty$ respecto a la solución exacta para $n = 3, 4, ..., 14$ y graficar el logaritmo del error en función del logaritmo de $h$ para estimar el orden del método. Las soluciones mostrarlas como gráficos para algunos $n$ representativos. Para el orden del error estimar la pendiente del gráfico log-log.

3. Repetir el primer ítem y resolverlo para $f(x) = e^{-x^2}$, $\alpha = 2$, $\beta = 1$ utilizando matrices esparsas. ¿Cuál es el mayor $n$ que puede utilizar usando una máquina local? ¿Por qué? Mostrar un gráfico de la solución para el $n$ más alto.