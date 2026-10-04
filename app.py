import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import signal
import streamlit as st

st.set_page_config(
    page_title="Laboratorio Virtual de Vibraciones y DSP", 
    layout="wide"
)

st.title("🎛️ Laboratorio Virtual de Procesamiento Digital de Señales (DSP)")
st.markdown("""
**Simulador Interactivo de Colector FFT Industrial:** Experimenta en tiempo real con la **Anatomía del Ventaneado**,
la **Fuga de Energía (Leakage) por Discontinuidad** y la **Reducción de Ruido por Promediado Espectral ($K$)**.
""")

# --- BARRA LATERAL DE CONFIGURACIÓN ---
st.sidebar.header("1. ⚙️ Configuración del Colector")
nl_opciones = [100, 200, 400, 800, 1600, 3200, 6400, 12800, 25600, 51200]
NL = st.sidebar.selectbox("Líneas de Resolución (NL)", nl_opciones, index=3)
fs = st.sidebar.slider("Frecuencia de Muestreo (fs en Hz)", 100, 5000, 1000, step=100)

N = int(2.56 * NL)
Fmax = fs / 2.56
delta_f = Fmax / NL
T_bloque = N / fs

# 2. Promediado Espectral
st.sidebar.markdown("---")
st.sidebar.header("2. 📊 Promediado Espectral (K)")
num_promedios = st.sidebar.slider("Número de Promedios (K)", 1, 32, 8, step=1)
T_total_maquina = num_promedios * T_bloque

# 3. Filtro Anti-Aliasing
st.sidebar.markdown("---")
st.sidebar.header("3. 🛡️ Filtro Anti-Aliasing")
modo_filtro = st.sidebar.radio("Modo del Filtro", ["Auto (Modo Colector)", "Manual (Educativo)", "Apagado"])

if modo_filtro == "Auto (Modo Colector)":
    f_corte = Fmax
    usar_filtro = True
elif modo_filtro == "Manual (Educativo)":
    f_corte = st.sidebar.slider("Corte Filtro (Hz)", 10.0, float(fs/2), float(Fmax), step=10.0)
    usar_filtro = True
else:
    usar_filtro = False
    f_corte = Fmax

# 4. Ventana Digital
st.sidebar.markdown("---")
st.sidebar.header("4. 🪟 Ventana Digital (Windowing)")
tipo_ventana = st.sidebar.selectbox(
    "Seleccionar Ventana", 
    ["Rectangular (Sin Ventana)", "Hanning", "Flat-Top", "Hamming", "Blackman"]
)

# 5. Inyección de Ruido
st.sidebar.markdown("---")
st.sidebar.header("5. 🔊 Inyección de Ruido Blanco")
agregar_ruido = st.sidebar.checkbox("Activar Ruido Severo", value=True)
nivel_ruido = st.sidebar.slider("Nivel de Ruido Blanco (RMS)", 0.0, 15.0, 6.0, step=0.5) if agregar_ruido else 0.0

# 6. Mala Muestra / Descalce
st.sidebar.markdown("---")
st.sidebar.header("6. ⚠️ Simulación de Mala Muestra")
activar_mala_muestra = st.sidebar.checkbox("Provocar Descalce de Ciclo (Fuga)", value=True)
descalce_frec = 0.0
if activar_mala_muestra:
    descalce_frec = st.sidebar.slider(
        "Descalce respecto a la línea Δf", 0.0, 0.5, 0.5, step=0.05,
        help="0.5 Δf hace caer la frecuencia exactamente entre dos líneas FFT, causando el máximo descalce."
    )

# 7. Mezclador de Frecuencias
st.sidebar.markdown("---")
st.sidebar.header("7. 🎶 Mezclador de Frecuencias")
num_componentes = st.sidebar.number_input("Diales Activos", 1, 10, 2, step=1)

# --- GENERACIÓN DE SEÑAL CONTINUA Y PROCESAMIENTO ---
factor_oversample = 8
fs_analogo = fs * factor_oversample
t_largo = np.linspace(0, T_total_maquina, int(fs_analogo * T_total_maquina), endpoint=False)
y_largo = np.zeros_like(t_largo)

componentes_info = []
for i in range(int(num_componentes)):
    with st.sidebar.expander(f"🎛 Dial #{i+1}", expanded=(i == 0)):
        f_base = float((i + 1) * 60.0)
        f_aplicada = f_base + (descalce_frec * delta_f if i == 0 else 0.0)

        freq = st.slider(f"Frecuencia {i+1} (Hz)", 5.0, 2500.0, f_aplicada, step=0.5, key=f"f_{i}")
        amp = st.slider(f"Amplitud {i+1} (mm/s)", 0.0, 10.0, float(max(10.0 - i*4, 3.0)), step=0.1, key=f"a_{i}")
        if amp > 0:
            y_largo += amp * np.sin(2 * np.pi * freq * t_largo)
            componentes_info.append((freq, amp))

if nivel_ruido > 0:
    np.random.seed(42)  # Mantiene consistencia temporal
    y_largo += np.random.normal(0, nivel_ruido, len(t_largo))

# Filtrado Anti-Aliasing
if usar_filtro:
    sos = signal.butter(6, f_corte, btype='low', fs=fs_analogo, output='sos')
    y_filtrado = signal.sosfiltfilt(sos, y_largo)
else:
    y_filtrado = y_largo

# Procesamiento por Bloques K y FFT
espectros_acumulados = []
bloques_digitales = []
bloques_ventaneados = []

for k in range(num_promedios):
    t_inicio = k * T_bloque
    t_fin = t_inicio + T_bloque
    indices_bloque = (t_largo >= t_inicio) & (t_largo < t_fin)
    y_bloque_analogo = y_filtrado[indices_bloque]

    paso = factor_oversample
    y_digital = y_bloque_analogo[::paso][:N]
    bloques_digitales.append(y_digital)

    # Construcción de Ventana
    if tipo_ventana == "Hanning":
        win = np.hanning(len(y_digital))
        factor_c = 2.0
    elif tipo_ventana == "Flat-Top":
        win = signal.windows.flattop(len(y_digital))
        factor_c = 4.18
    elif tipo_ventana == "Hamming":
        win = np.hamming(len(y_digital))
        factor_c = 1.85
    elif tipo_ventana == "Blackman":
        win = np.blackman(len(y_digital))
        factor_c = 2.80
    else:  # Rectangular
        win = np.ones(len(y_digital))
        factor_c = 1.0

    y_win = y_digital * win
    bloques_ventaneados.append(y_win)

    fft_k = np.abs(np.fft.rfft(y_win)) * (2.0 / len(y_digital)) * (factor_c / 2.0)
    espectros_acumulados.append(fft_k)

espectro_promediado = np.mean(espectros_acumulados, axis=0)
fft_freqs = np.fft.rfftfreq(N, 1/fs)

# --- TABLERO SUPERIOR DE MÉTRICAS ---
st.markdown("### 📋 Parámetros de Adquisición Digital")
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Líneas FFT (NL)", f"{NL:,}")
c2.metric("Puntos en Tiempo (N)", f"{N:,} pts", delta=f"2.56 × {NL}")
c3.metric("Período Bloque (T)", f"{T_bloque:.3f} s" if T_bloque >= 1 else f"{T_bloque*1000:.1f} ms")
c4.metric("Período Total (K Promedios)", f"{T_total_maquina:.2f} s", delta=f"K = {num_promedios}")
c5.metric("Resolución Espectral (Δf)", f"{delta_f:.3f} Hz")

st.markdown("---")

# ==============================================================================
# MÓDULO 1: ANATOMÍA DEL VENTANEADO DIGITAL (TIEMPO)
# ==============================================================================
st.subheader("1. 🪟 Efecto Físico de la Ventana Digital en el Dominio del Tiempo")
st.markdown("""
Compara la **Señal Bruta** (azul) contra la **Señal Ventaneada** (verde). Observa cómo la curva de la **Ventana** (naranja)
obliga a los bordes inicial y final de la muestra a llegar suavemente a **cero**, eliminando el salto brusco.
""")

y_primer_bloque = bloques_digitales[0]
y_primer_ventaneado = bloques_ventaneados[0]
t_primer_bloque = np.linspace(0, T_bloque, len(y_primer_bloque), endpoint=False)

fig_win = make_subplots(rows=1, cols=2, subplot_titles=(
    "a) Señal Capturada vs Envolvente de Ventana",
    f"b) Señal Resultante Aplicando Ventana {tipo_ventana}"
))

max_amp = np.max(np.abs(y_primer_bloque)) if np.max(np.abs(y_primer_bloque)) > 0 else 1.0
fig_win.add_trace(go.Scatter(x=t_primer_bloque, y=y_primer_bloque, mode='lines', name='Señal Bruta y(t)', line=dict(color='#1f77b4', width=1.2)), row=1, col=1)
fig_win.add_trace(go.Scatter(x=t_primer_bloque, y=win * max_amp, mode='lines', name=f'Ventana {tipo_ventana}', line=dict(color='#ff7f0e', width=2.5, dash='dash')), row=1, col=1)
fig_win.add_trace(go.Scatter(x=t_primer_bloque, y=-win * max_amp, mode='lines', showlegend=False, line=dict(color='#ff7f0e', width=2.5, dash='dash')), row=1, col=1)

fig_win.add_trace(go.Scatter(x=t_primer_bloque, y=y_primer_ventaneado, mode='lines', name='Señal Ventaneada y_win(t)', line=dict(color='#2ca02c', width=1.5)), row=1, col=2)

fig_win.update_layout(template="plotly_white", height=320, margin=dict(l=20, r=20, t=40, b=20))
fig_win.update_xaxes(title_text="Tiempo en el Bloque (s)")
fig_win.update_yaxes(title_text="Amplitud (mm/s)")
st.plotly_chart(fig_win, use_container_width=True)

st.markdown("---")

# ==============================================================================
# MÓDULO 2: CAUSA RAÍZ DE LA FUGA DE ENERGÍA (DISCONTINUIDAD PERIÓDICA)
# ==============================================================================
st.subheader("2. ⚠️ Causa Raíz de la Fuga (Leakage): Discontinuidad al Repetir la Muestra")
st.markdown("""
La FFT asume que la muestra capturada se **repite infinitamente**. Al unir el final del primer bloque con el inicio del segundo ($t = T$),
si no hay un número entero de ciclos (descalce), se genera una **discontinuidad o salto brusco (escalón)**.
""")

col_leak1, col_leak2 = st.columns([1.2, 1.0])

with col_leak1:
    y_repetido = np.concatenate([y_primer_bloque, y_primer_bloque])
    t_repetido = np.linspace(0, 2 * T_bloque, len(y_repetido), endpoint=False)

    fig_rep = go.Figure()
    fig_rep.add_trace(go.Scatter(x=t_repetido[:N], y=y_repetido[:N], mode='lines', name='Bloque #1 Capturado', line=dict(color='#1f77b4', width=1.5)))
    fig_rep.add_trace(go.Scatter(x=t_repetido[N:], y=y_repetido[N:], mode='lines', name='Bloque #2 (Repetición FFT)', line=dict(color='#9467bd', width=1.5)))

    salto = abs(y_primer_bloque[-1] - y_primer_bloque[0])
    fig_rep.add_vline(x=T_bloque, line_width=2.5, line_dash="solid", line_color="red",
                      annotation_text="⚠️ Punto de Unión (t = T)", annotation_position="top left")

    fig_rep.update_layout(
        title=f"<b>Unión de Bloques ($2T$): Salto en el Borde = {salto:.2f} mm/s</b>",
        xaxis_title="Tiempo Extendido (s)", yaxis_title="Amplitud (mm/s)",
        template="plotly_white", height=320
    )
    st.plotly_chart(fig_rep, use_container_width=True)

with col_leak2:
    f_target = componentes_info[0][0] if componentes_info else 60.0
    fig_zoom = go.Figure()
    fig_zoom.add_trace(go.Scatter(x=fft_freqs, y=espectro_promediado, mode='lines+markers', name='Espectro FFT', line=dict(color='#d62728', width=2), marker=dict(size=4)))

    rango_zoom = [max(0, f_target - 15*delta_f), f_target + 15*delta_f]
    fig_zoom.update_layout(
        title=f"<b>Zoom al Pico Principal ({f_target:.1f} Hz) - Faldón de Fuga</b>",
        xaxis_title="Frecuencia (Hz)", yaxis_title="Amplitud Peak (mm/s)",
        xaxis_range=rango_zoom, template="plotly_white", height=320
    )
    st.plotly_chart(fig_zoom, use_container_width=True)

st.markdown("---")

# ==============================================================================
# MÓDULO 3: PROMEDIADO ESPECTRAL (K) Y SUPRESIÓN DE RUIDO
# ==============================================================================
st.subheader("3. 📊 Reducción del Piso de Ruido mediante Promediado Espectral ($K$)")
st.markdown("""
Las **líneas grises tenues** muestran los **$K$ espectros individuales** calculados en cada instante de tiempo.
La **línea roja gruesa** es el **espectro promedio**. Observa cómo al aumentar $K$, la variación aleatoria del ruido se cancela y los picos mecánicos reales emergen con claridad.
""")

fig_prom = go.Figure()

for k_idx, espectro_k in enumerate(espectros_acumulados):
    fig_prom.add_trace(go.Scatter(
        x=fft_freqs, y=espectro_k, mode='lines',
        line=dict(color='rgba(150, 150, 150, 0.35)', width=1),
        name=f'Bloque #{k_idx+1}' if k_idx < 3 else None,
        showlegend=(k_idx < 3)
    ))

fig_prom.add_trace(go.Scatter(
    x=fft_freqs, y=espectro_promediado, mode='lines',
    line=dict(color='#d62728', width=2.5),
    name=f'<b>PROMEDIO FINAL (K={num_promedios})</b>'
))

fig_prom.add_vline(x=Fmax, line_width=1.5, line_dash="dot", line_color="blue", annotation_text=f"Fmax = {Fmax:.1f} Hz")

fig_prom.update_layout(
    title=f"<b>Espectros Individuales (Gris) vs Espectro Promediado (Rojo, K={num_promedios})</b>",
    xaxis_title="Frecuencia (Hz)", yaxis_title="Amplitud Peak (mm/s)",
    xaxis_range=[0, fs/2], template="plotly_white", height=380
)
st.plotly_chart(fig_prom, use_container_width=True)

st.info("""
💡 **Guía de Demostración Práctica para la Clase:**
1. **Para explicar la Ventana:** Deja activado el *Descalce en 0.5 Δf*. Cambia entre **Rectangular** y **Hanning**. Muestra cómo en el Módulo 1 la ventana Hanning atenúa la señal a cero en los bordes, eliminando el escalón del Módulo 2 y reduciendo drásticamente el faldón de fuga en la FFT.
2. **Para explicar el Promediado:** Sube el **Ruido Severo a 8.0 RMS**. Con **K = 1**, el ruido sepulta el pico de vibración. A medida que subes **K a 16 o 32**, las líneas grises (ruido aleatorio) se cancelan entre sí y la línea roja (pico mecánico) se vuelve limpia y estable.
""")