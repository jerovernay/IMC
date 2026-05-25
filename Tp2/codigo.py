"""
TP2 - Identificación de número telefónico por tonos DTMF
Introducción al Modelado Continuo

Estrategia:
  tlfn-a → tonos con silencios claros → segmentación por energía
  tlfn-b → señal continua con ruido   → ventana deslizante + votación
"""

import numpy as np
import matplotlib.pyplot as plt
import soundfile as sf
from scipy.signal import find_peaks

# =============================================================================
# TABLA DTMF
#          1209 Hz  1336 Hz  1477 Hz
#  697 Hz :   1        2        3
#  770 Hz :   4        5        6
#  852 Hz :   7        8        9
#  941 Hz :   *        0        #
# =============================================================================
FILAS = np.array([697,  770,  852,  941])
COLS  = np.array([1209, 1336, 1477])
TABLA = [['1','2','3'],
         ['4','5','6'],
         ['7','8','9'],
         ['*','0','#']]

def digito_dtmf(f1, f2):
    """Mapea dos frecuencias al dígito DTMF más cercano."""
    f_baja = min(f1, f2)
    f_alta = max(f1, f2)
    fila   = int(np.argmin(np.abs(FILAS - f_baja)))
    col    = int(np.argmin(np.abs(COLS  - f_alta)))
    # Validar que las frecuencias estén razonablemente cerca
    if abs(f_baja - FILAS[fila]) > 80 or abs(f_alta - COLS[col]) > 80:
        return None   # ruido, no DTMF
    return TABLA[fila][col]

# =============================================================================
# FFT de un segmento → devuelve (f1, f2, dig) o None si no hay picos claros
# =============================================================================
def detectar_digito_en_segmento(segmento, fs, umbral_pico=0.25):
    N     = len(segmento)
    fft   = np.abs(np.fft.rfft(segmento * np.hanning(N)))  # ventana Hanning
    freqs = np.fft.rfftfreq(N, d=1/fs)

    # Solo rango DTMF
    mask    = (freqs >= 620) & (freqs <= 1550)
    fft_rec = fft.copy()
    fft_rec[~mask] = 0

    if fft_rec.max() == 0:
        return None, None, None

    picos, props = find_peaks(fft_rec,
                              height=fft_rec.max() * umbral_pico,
                              distance=int(50 * N / fs))   # ~50 Hz separación mínima

    if len(picos) < 2:
        return None, None, None

    top2 = picos[np.argsort(fft_rec[picos])[::-1]][:2]
    f1, f2 = freqs[top2[0]], freqs[top2[1]]
    dig = digito_dtmf(f1, f2)
    return f1, f2, dig

# =============================================================================
# MÉTODO A: Segmentación por energía (para tlfn-a con silencios claros)
# =============================================================================
def segmentar_por_energia(audio, fs, tam_ms=20, umbral_rel=0.04):
    tam = int(fs * tam_ms / 1000)
    energia = np.array([
        np.sum(audio[i:i+tam]**2)
        for i in range(0, len(audio)-tam, tam)
    ])
    umbral = energia.max() * umbral_rel
    activo = (energia > umbral).astype(int)
    cambios = np.diff(np.r_[0, activo, 0])
    inicios = np.where(cambios ==  1)[0] * tam
    fines   = np.where(cambios == -1)[0] * tam
    # Filtrar segmentos muy cortos (< 30 ms) → probablemente transitorios
    segs = [(i, f) for i, f in zip(inicios, fines) if (f - i) >= int(fs * 0.03)]
    return segs

# =============================================================================
# MÉTODO B: Ventana deslizante + votación (para tlfn-b con ruido continuo)
# =============================================================================
def detectar_por_ventana_deslizante(audio, fs, tam_ms=60, paso_ms=20):
    """
    Recorre la señal con ventanas solapadas.
    Agrupa ventanas consecutivas que detectan el mismo dígito → un tono.
    Desecha ventanas donde no se detecta DTMF válido (ruido).
    """
    tam  = int(fs * tam_ms  / 1000)
    paso = int(fs * paso_ms / 1000)

    resultados = []  # (t_inicio, digito)
    for i in range(0, len(audio) - tam, paso):
        seg = audio[i:i+tam]
        f1, f2, dig = detectar_digito_en_segmento(seg, fs)
        t = i / fs
        resultados.append((t, dig))

    # Agrupar secuencias del mismo dígito (ignorar None)
    tonos = []
    i = 0
    while i < len(resultados):
        t_ini, dig = resultados[i]
        if dig is None:
            i += 1
            continue
        # Avanzar mientras el dígito sea el mismo
        j = i
        while j < len(resultados) and resultados[j][1] == dig:
            j += 1
        duracion = (j - i) * paso_ms / 1000   # segundos
        # Solo aceptar grupos que duren al menos 50 ms (elimina falsas detecciones)
        if duracion >= 0.05:
            t_fin = resultados[j-1][0] + tam_ms/1000
            tonos.append((t_ini, t_fin, dig))
        i = j

    # Fusionar tonos duplicados consecutivos separados por gaps muy breves
    fusionados = []
    for tono in tonos:
        if fusionados and fusionados[-1][2] == tono[2] and (tono[0] - fusionados[-1][1]) < 0.08:
            fusionados[-1] = (fusionados[-1][0], tono[1], tono[2])
        else:
            fusionados.append(list(tono))

    return fusionados

# =============================================================================
# GRÁFICO: Forma de onda
# =============================================================================
def plot_waveform(audio, fs, nombre, marcas=None):
    t = np.arange(len(audio)) / fs
    fig, ax = plt.subplots(figsize=(13, 3))
    ax.plot(t, audio, linewidth=0.4, color='steelblue')
    if marcas:
        for idx, (t0, t1, dig) in enumerate(marcas):
            ax.axvspan(t0, t1, alpha=0.25, color='orange')
            ax.text((t0+t1)/2, audio.max()*0.85, dig,
                    ha='center', fontsize=9, fontweight='bold', color='darkred')
    ax.set_title(f"Forma de onda — {nombre}", fontsize=13)
    ax.set_xlabel("Tiempo (s)")
    ax.set_ylabel("Amplitud")
    plt.tight_layout()
    plt.savefig(f"waveform_{nombre.replace('.wav','')}.png", dpi=150)
    plt.show()

# =============================================================================
# GRÁFICO: FFT por segmento (paneles apilados)
# =============================================================================
def plot_fft_segmentos(audio, fs, segmentos_info, nombre):
    """
    segmentos_info: lista de (t_inicio, t_fin, digito)  [en segundos o muestras]
    """
    n = len(segmentos_info)
    fig, axes = plt.subplots(n, 1, figsize=(11, 3*n), squeeze=False)

    for i, (t0, t1, dig) in enumerate(segmentos_info):
        # Convertir a muestras si viene en segundos
        ini = int(t0 * fs) if isinstance(t0, float) else t0
        fin = int(t1 * fs) if isinstance(t1, float) else t1
        segmento = audio[ini:fin]

        N     = len(segmento)
        fft   = np.abs(np.fft.rfft(segmento * np.hanning(N)))
        freqs = np.fft.rfftfreq(N, d=1/fs)

        f1, f2, _ = detectar_digito_en_segmento(segmento, fs)

        ax = axes[i][0]
        ax.plot(freqs, fft, color='steelblue', linewidth=0.7, label='Espectro')
        if f1 and f2:
            ax.axvline(min(f1,f2), color='crimson',   linestyle='--', lw=1.5,
                       label=f'{min(f1,f2):.0f} Hz (fila)')
            ax.axvline(max(f1,f2), color='darkorange', linestyle='--', lw=1.5,
                       label=f'{max(f1,f2):.0f} Hz (col.)')
        ax.set_xlim(500, 1700)
        ax.set_title(f"Segmento {i+1}  →  dígito '{dig}'", fontsize=10)
        ax.set_xlabel("Frecuencia (Hz)")
        ax.set_ylabel("|FFT|")
        ax.legend(fontsize=8)

    fig.suptitle(f"Espectro DTMF por dígito — {nombre}",
                 fontsize=13, fontweight='bold', y=1.005)
    plt.tight_layout()
    plt.savefig(f"fft_{nombre.replace('.wav','')}.png", dpi=150, bbox_inches='tight')
    plt.show()

# =============================================================================
# PROCESAR tlfn-a  (silencios claros → energía)
# =============================================================================
def procesar_a(nombre='tlfn-a.wav'):
    audio, fs = sf.read(nombre)
    if audio.ndim > 1:
        audio = audio[:, 0]

    print(f"\n{'='*55}")
    print(f"  Archivo : {nombre}  |  Fs = {fs} Hz  |  Dur = {len(audio)/fs:.2f} s")
    print(f"{'='*55}")

    segs = segmentar_por_energia(audio, fs)
    print(f"  Tonos detectados: {len(segs)}\n")

    numero = []
    segs_info = []

    for i, (ini, fin) in enumerate(segs):
        segmento = audio[ini:fin]
        f1, f2, dig = detectar_digito_en_segmento(segmento, fs)
        if dig is None:
            dig = '?'
        numero.append(dig)
        segs_info.append((ini/fs, fin/fs, dig))
        print(f"  Tono {i+1:>2d} | {min(f1,f2):.1f} Hz  {max(f1,f2):.1f} Hz  →  '{dig}'")

    plot_waveform(audio, fs, nombre, marcas=segs_info)
    plot_fft_segmentos(audio, fs, segs_info, nombre)

    resultado = ''.join(numero)
    print(f"\n  ► Número identificado: {resultado}")
    print(f"{'='*55}\n")
    return resultado

# =============================================================================
# PROCESAR tlfn-b  (ruido continuo → ventana deslizante)
# =============================================================================
def procesar_b(nombre='tlfn-b.wav'):
    audio, fs = sf.read(nombre)
    if audio.ndim > 1:
        audio = audio[:, 0]

    print(f"\n{'='*55}")
    print(f"  Archivo : {nombre}  |  Fs = {fs} Hz  |  Dur = {len(audio)/fs:.2f} s")
    print(f"{'='*55}")
    print("  Estrategia: ventana deslizante (60 ms, paso 20 ms)\n")

    tonos = detectar_por_ventana_deslizante(audio, fs, tam_ms=60, paso_ms=20)

    numero = []
    for i, (t0, t1, dig) in enumerate(tonos):
        numero.append(dig)
        print(f"  Tono {i+1:>2d} | {t0:.2f}s – {t1:.2f}s  →  '{dig}'")

    plot_waveform(audio, fs, nombre, marcas=tonos)
    plot_fft_segmentos(audio, fs, tonos, nombre)

    resultado = ''.join(numero)
    print(f"\n  ► Número identificado: {resultado}")
    print(f"{'='*55}\n")
    return resultado

# =============================================================================
# EJECUCIÓN
# =============================================================================
if __name__ == "__main__":
    num_a = procesar_a('tlfn-a.wav')
    num_b = procesar_b('tlfn-b.wav')
    print("\nResumen final:")
    print(f"  tlfn-a → {num_a}")
    print(f"  tlfn-b → {num_b}")