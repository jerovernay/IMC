TRABAJO PRÁCTICO N°1: TRAYECTORIAS DE KEPLER Y CORRECCIÓN RELATIVISTA
Introducción al Modelado Continuo - 1er Cuatrimestre 2026
Grupo 11

DESCRIPCIÓN
Este proyecto consiste en la resolución numérica y análisis de las ecuaciones de movimiento para el problema de dos cuerpos en el sistema Sol-planeta. Se utiliza una formulación en coordenadas polares u(theta) = 1/r(theta) para estudiar:
1. Órbitas de Kepler (Clásicas): Análisis de trayectorias cerradas (elipses) y abiertas (parábolas/hipérbolas) variando la excentricidad epsilon.
2. Corrección Relativista: Estudio del fenómeno de precesión del perihelio mediante la inclusión del término delta * u^2(theta).
3. Visualización Multidimensional: Gráficos en el espacio de fases (u, u_dot), coordenadas polares y planos cartesianos filtrados.

ESTRUCTURA DEL PROYECTO
- Codigo_Graficos_Grupo11.py: Script principal optimizado para entornos locales (Spyder, VS Code, PyCharm).
- TP1_Grupo11.ipynb: Versión adaptada para Google Colab, incluyendo animaciones 3D.
- graficos/: Carpeta con capturas de los resultados en alta resolución (PNG).
- README.txt: Este archivo informativo.

REQUISITOS Y DEPENDENCIAS
Se requiere Python 3 con las siguientes librerías:
- numpy
- scipy
- matplotlib

INSTRUCCIONES DE EJECUCIÓN
1. Entorno Local: Abrir Codigo_Graficos_Grupo11.py y ejecutar. Asegúrese de que el backend de gráficos permita ventanas emergentes para la animación 3D.
2. Google Colab: Subir el archivo .ipynb y ejecutar las celdas.

NOTA SOBRE EL CONTENIDO DEL ZIP
El archivo comprimido tiene un peso superior al promedio debido a la inclusión de los códigos en dos formatos y la carpeta completa de gráficos exportados para asegurar una validación inmediata de los resultados.