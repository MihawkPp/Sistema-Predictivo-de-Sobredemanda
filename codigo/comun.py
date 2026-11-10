"""Utilidades compartidas: rutas, calendario, umbral de sobredemanda y estilo de gráficas."""
from pathlib import Path
import pandas as pd
import holidays
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RAIZ = Path(__file__).resolve().parent.parent
DATOS = RAIZ / "datos"
RES = RAIZ / "resultados"
RES.mkdir(exist_ok=True)

# Periodos del prototipo
FIN_ENTRENAMIENTO = pd.Timestamp("2026-08-24")   # se entrena del 1 al 24 de agosto
INICIO_PRUEBA = pd.Timestamp("2026-08-25")       # se evalúa del 25 al 31 de agosto
PERCENTIL_UMBRAL = 0.90

FESTIVOS_CO = holidays.Colombia(years=[2026])

# ---- Paleta (instancia de referencia, modo claro) ----
SUPERFICIE = "#fcfcfb"
TEXTO = "#0b0b0b"
TEXTO_2 = "#52514e"
REJILLA = "#e1e0d9"
EJE = "#c3c2b7"
SERIE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]   # azul, naranja, aqua, amarillo
CRITICO = "#d03b3b"                                    # estado: sobredemanda
SECUENCIAL = "Blues"


def cargar_largo() -> pd.DataFrame:
    d = pd.read_csv(DATOS / "validaciones_15min_largo.csv",
                    dtype={"cod_estacion": str}, parse_dates=["fecha", "timestamp"])
    # Cicloparqueaderos (bicicleteros y corrales) no son estaciones de pasajeros
    d = d[~d["estacion"].str.contains("Bicicletero|Corral", case=False)].copy()
    d = agregar_calendario(d)
    return marcar_dias_anomalos(d)


def marcar_dias_anomalos(d: pd.DataFrame, fraccion=0.10) -> pd.DataFrame:
    """Marca estación-día con un total menor al 10 % de la mediana de esa estación
    para el mismo tipo de día (cierres, fallas de registro). Esos días no se usan
    para entrenar, ni como rezago, ni para evaluar."""
    tot = d.groupby(["cod_estacion", "fecha", "tipo_dia"], as_index=False)["validaciones"].sum()
    med = tot.groupby(["cod_estacion", "tipo_dia"])["validaciones"].transform("median")
    tot["dia_anomalo"] = ((tot["validaciones"] < fraccion * med) | (tot["validaciones"] == 0)).astype(int)
    d = d.merge(tot[["cod_estacion", "fecha", "dia_anomalo"]], on=["cod_estacion", "fecha"])
    # Se excluyen puntos sin demanda relevante (nunca superan 100 validaciones en un día)
    max_est = tot.groupby("cod_estacion")["validaciones"].max()
    return d[d["cod_estacion"].isin(max_est[max_est >= 100].index)].copy()


def tipo_dia(fecha: pd.Timestamp) -> str:
    if fecha in FESTIVOS_CO or fecha.dayofweek == 6:
        return "Domingo/festivo"
    if fecha.dayofweek == 5:
        return "Sábado"
    return "Hábil"


def agregar_calendario(d: pd.DataFrame) -> pd.DataFrame:
    fechas = pd.Series(d["fecha"].unique())
    mapa = {f: tipo_dia(f) for f in fechas}
    d["tipo_dia"] = d["fecha"].map(mapa)
    d["festivo"] = d["fecha"].map(lambda f: int(f in FESTIVOS_CO))
    d["dia_semana"] = d["fecha"].dt.dayofweek
    d["hora"] = d["timestamp"].dt.hour
    d["minuto_dia"] = d["timestamp"].dt.hour * 60 + d["timestamp"].dt.minute
    return d


def umbrales_sobredemanda(d: pd.DataFrame) -> pd.Series:
    """Umbral por estación: percentil 90 de las validaciones por intervalo de 15 min
    en días hábiles del periodo de entrenamiento, considerando solo intervalos con operación.
    Funciona como una aproximación a la capacidad crítica, que no está publicada."""
    base = d[(d["fecha"] <= FIN_ENTRENAMIENTO) & (d["tipo_dia"] == "Hábil")
             & (d["validaciones"] > 0) & (d["dia_anomalo"] == 0)]
    return base.groupby("cod_estacion")["validaciones"].quantile(PERCENTIL_UMBRAL).rename("umbral")


def estilo_ejes(ax, titulo=None, xlabel=None, ylabel=None):
    ax.set_facecolor(SUPERFICIE)
    for lado in ["top", "right", "left"]:
        ax.spines[lado].set_visible(False)
    ax.spines["bottom"].set_color(EJE)
    ax.grid(axis="y", color=REJILLA, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(colors=TEXTO_2, labelsize=9, length=0)
    if titulo:
        ax.set_title(titulo, loc="left", color=TEXTO, fontsize=12, fontweight="bold", pad=12)
    if xlabel:
        ax.set_xlabel(xlabel, color=TEXTO_2, fontsize=9)
    if ylabel:
        ax.set_ylabel(ylabel, color=TEXTO_2, fontsize=9)


def nueva_figura(ancho=10, alto=4.2):
    fig, ax = plt.subplots(figsize=(ancho, alto), dpi=150)
    fig.patch.set_facecolor(SUPERFICIE)
    return fig, ax


def miles(x, _=None):
    return f"{x:,.0f}".replace(",", ".")
