"""
TP2 - Identificación de número telefónico por tonos DTMF
Introducción al Modelado Continuo

Flujo: .wav -> segmentar digitos -> DFT (via FFT) -> 2 picos -> tabla DTMF -> numero
"""

import numpy as np
import matplotlib.pyplot as plt
import soundfile as sf
from scipy.fft import rfft, rfftfreq     
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
    """
    Dados dos picos, devuelve el digito DTMF mas cercano.
    La frecuencia mas baja va a la fila; la mas alta, a la columna.
    Rechaza detecciones que se alejen mas de 80 Hz de los valores canonicos.
    """
    fb = min(f1, f2)  
    fa = max(f1, f2)
    fi = int(np.argmin(np.abs(FILAS - fb)))
    fc = int(np.argmin(np.abs(COLS  - fa)))
    if abs(fb - FILAS[fi]) > 80 or abs(fa - COLS[fc]) > 80:
        return None   # ruido, no DTMF valido
    return TABLA[fi][fc]

# =============================================================================
# NUCLEO: DFT de un segmento via FFT
#
# La DFT (ec. 14.2.1 de las notas) es:
#   f_hat[k] = sum_{j=0}^{N-1} f[j] * exp(-2*pi*i*j*k/N)
#
# scipy.fft.rfft calcula exactamente esto, pero solo para k = 0..N//2
# (frecuencias positivas, suficientes para senales reales).
# El eje de frecuencias es: xi_k = k * fs / N  (resolucion Delta_xi = fs/N).
# =============================================================================
def detectar_digito_en_segmento(segmento, fs):
    N     = len(segmento)
    fhat  = np.abs(rfft(segmento))       # |f_hat[k]|, frec. positivas
    freqs = rfftfreq(N, d=1/fs)          # xi_k = k * fs / N

    # Solo buscar en rango DTMF (620-1550 Hz); fuera de ese rango, cero
    mask     = (freqs >= 620) & (freqs <= 1550)
    fhat_rec = fhat.copy()
    fhat_rec[~mask] = 0

    if fhat_rec.max() == 0:
        return None, None, None

    # Detectar picos con separacion minima ~50 Hz en muestras
    picos, _ = find_peaks(fhat_rec,
                          height=fhat_rec.max() * 0.25,
                          distance=int(50 * N / fs))

    if len(picos) < 2:
        return None, None, None

    # Los dos picos de mayor |f_hat[k]| son las dos frecuencias DTMF
    top2 = picos[np.argsort(fhat_rec[picos])[::-1]][:2]
    f1, f2 = freqs[top2[0]], freqs[top2[1]]
    return f1, f2, digito_dtmf(f1, f2)

# =============================================================================
# SEGMENTACION POR ENERGIA (para tlfn-a: silencios claros)
# =============================================================================
def segmentar_por_energia(audio, fs, tam_ms=20, umbral_rel=0.04):
    tam    = int(fs * tam_ms / 1000)
    energia = np.array([np.sum(audio[i:i+tam]**2)
                        for i in range(0, len(audio)-tam, tam)])
    activo  = (energia > energia.max() * umbral_rel).astype(int)
    cambios = np.diff(np.r_[0, activo, 0])
    inicios = np.where(cambios ==  1)[0] * tam
    fines   = np.where(cambios == -1)[0] * tam
    # Descartar segmentos muy cortos (< 30 ms) -> transitorios
    return [(i, f) for i, f in zip(inicios, fines)
            if (f - i) >= int(fs * 0.03)]

# =============================================================================
# VENTANA DESLIZANTE (para tlfn-b: ruido continuo, sin silencios)
# =============================================================================
def detectar_por_ventana_deslizante(audio, fs, tam_ms=60, paso_ms=20):
    tam  = int(fs * tam_ms  / 1000)
    paso = int(fs * paso_ms / 1000)

    resultados = [(i/fs, detectar_digito_en_segmento(audio[i:i+tam], fs)[2])
                  for i in range(0, len(audio)-tam, paso)]

    # Agrupar ventanas consecutivas con el mismo digito
    tonos = []; i = 0
    while i < len(resultados):
        t_ini, dig = resultados[i]
        if dig is None: i += 1; continue
        j = i
        while j < len(resultados) and resultados[j][1] == dig:
            j += 1
        dur = (j - i) * paso_ms / 1000
        if dur >= 0.05:   # al menos 50 ms -> no es ruido transitorio
            tonos.append([t_ini, resultados[j-1][0] + tam_ms/1000, dig])
        i = j

    # Fusionar tonos identicos separados por un gap muy breve (< 80 ms)
    fusionados = []
    for t in tonos:
        if fusionados and fusionados[-1][2] == t[2] \
                      and (t[0] - fusionados[-1][1]) < 0.08:
            fusionados[-1][1] = t[1]
        else:
            fusionados.append(t)
    return fusionados

# =============================================================================
# GRAFICOS
# =============================================================================
def plot_waveform(audio, fs, nombre, marcas=None):
    t = np.arange(len(audio)) / fs
    fig, ax = plt.subplots(figsize=(13, 3))
    ax.plot(t, audio, linewidth=0.4, color='steelblue')
    if marcas:
        for t0, t1, dig in marcas:
            ax.axvspan(t0, t1, alpha=0.25, color='orange')
            ax.text((t0+t1)/2, audio.max()*0.85, dig,
                    ha='center', fontsize=9, fontweight='bold', color='darkred')
    ax.set_title(f"Forma de onda — {nombre}", fontsize=13)
    ax.set_xlabel("Tiempo (s)"); ax.set_ylabel("Amplitud")
    plt.tight_layout()
    plt.savefig(f"waveform_{nombre.replace('.wav','')}.png", dpi=150)
    plt.show()

def plot_fft_segmentos(audio, fs, segs_info, nombre):
    n = len(segs_info)
    fig, axes = plt.subplots(n, 1, figsize=(11, 3*n), squeeze=False)
    for i, (t0, t1, dig) in enumerate(segs_info):
        ini = int(t0 * fs) if isinstance(t0, float) else t0
        fin = int(t1 * fs) if isinstance(t1, float) else t1
        seg = audio[ini:fin]
        N     = len(seg)
        fhat  = np.abs(rfft(seg))
        freqs = rfftfreq(N, d=1/fs)
        f1, f2, _ = detectar_digito_en_segmento(seg, fs)
        ax = axes[i][0]
        ax.plot(freqs, fhat, color='steelblue', linewidth=0.7, label='|f_hat[k]|')
        if f1 and f2:
            ax.axvline(min(f1,f2), color='crimson',    linestyle='--', lw=1.5,
                       label=f'{min(f1,f2):.0f} Hz (fila)')
            ax.axvline(max(f1,f2), color='darkorange', linestyle='--', lw=1.5,
                       label=f'{max(f1,f2):.0f} Hz (col.)')
        ax.set_xlim(500, 1700)
        ax.set_title(f"Segmento {i+1}  →  '{dig}'", fontsize=10)
        ax.set_xlabel("Frecuencia ξ (Hz)"); ax.set_ylabel("|f̂[k]|")
        ax.legend(fontsize=8)
    fig.suptitle(f"Espectro DTMF — {nombre}", fontsize=13, fontweight='bold', y=1.00005)
    plt.tight_layout()
    plt.savefig(f"fft_{nombre.replace('.wav','')}.png", dpi=150, bbox_inches='tight')
    plt.show()

# =============================================================================
# PROCESAMIENTO PRINCIPAL
# =============================================================================
def procesar_a(nombre='tlfn-a.wav'):
    audio, fs = sf.read(nombre)
    if audio.ndim > 1: audio = audio[:, 0]
    print(f"\n{'='*55}\n  {nombre}  |  Fs={fs} Hz  |  {len(audio)/fs:.2f} s\n{'='*55}")
    segs = segmentar_por_energia(audio, fs)
    print(f"  Tonos detectados: {len(segs)}\n")
    numero = []; info = []
    for i, (ini, fin) in enumerate(segs):
        f1, f2, dig = detectar_digito_en_segmento(audio[ini:fin], fs)
        if dig is None: dig = '?'
        numero.append(dig); info.append((ini/fs, fin/fs, dig))
        print(f"  Tono {i+1:>2d} | {min(f1,f2):.1f} Hz  {max(f1,f2):.1f} Hz  -> '{dig}'")
    plot_waveform(audio, fs, nombre, marcas=info)
    plot_fft_segmentos(audio, fs, info, nombre)
    print(f"\n  Numero identificado: {''.join(numero)}\n{'='*55}\n")
    return ''.join(numero)

def procesar_b(nombre='tlfn-b.wav'):
    audio, fs = sf.read(nombre)
    if audio.ndim > 1: audio = audio[:, 0]
    print(f"\n{'='*55}\n  {nombre}  |  Fs={fs} Hz  |  {len(audio)/fs:.2f} s")
    print(f"  Estrategia: ventana deslizante (60 ms, paso 20 ms)\n{'='*55}")
    tonos = detectar_por_ventana_deslizante(audio, fs)
    numero = []
    for i, (t0, t1, dig) in enumerate(tonos):
        numero.append(dig)
        print(f"  Tono {i+1:>2d} | {t0:.2f}s – {t1:.2f}s  -> '{dig}'")
    plot_waveform(audio, fs, nombre, marcas=tonos)
    plot_fft_segmentos(audio, fs, tonos, nombre)
    print(f"\n  Numero identificado: {''.join(numero)}\n{'='*55}\n")
    return ''.join(numero)

# =============================================================================
if __name__ == "__main__":
    na = procesar_a('tlfn-a.wav')
    nb = procesar_b('tlfn-b.wav')
    print(f"tlfn-a -> {na}")
    print(f"tlfn-b -> {nb}")
