"""
Paso 2: Exploración de patrones y definición de sobredemanda.

Genera tres gráficas y la tabla de umbrales por estación.
"""
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import numpy as np
import pandas as pd
from comun import (cargar_largo, umbrales_sobredemanda, estilo_ejes, nueva_figura, miles,
                   RES, SERIE, TEXTO_2, SUPERFICIE, TEXTO, FESTIVOS_CO)

ORDEN_TIPOS = ["Hábil", "Sábado", "Domingo/festivo"]


def fig_totales_diarios(d):
    tot = d.groupby(["fecha", "tipo_dia"], as_index=False)["validaciones"].sum()
    fig, ax = nueva_figura(10, 4)
    for i, t in enumerate(ORDEN_TIPOS):
        s = tot[tot["tipo_dia"] == t]
        ax.bar(s["fecha"], s["validaciones"], width=0.8, color=SERIE[i], label=t)
    estilo_ejes(ax, "Validaciones troncales por día · agosto de 2026", ylabel="Validaciones")
    ax.yaxis.set_major_formatter(FuncFormatter(miles))
    ax.set_xticks(tot["fecha"][::2])
    ax.set_xticklabels([f.strftime("%d") for f in tot["fecha"][::2]])
    ax.set_xlabel("Día de agosto", color=TEXTO_2, fontsize=9)
    for f in tot["fecha"]:
        if f in FESTIVOS_CO:
            v = tot.loc[tot["fecha"] == f, "validaciones"].iloc[0]
            ax.annotate("Festivo", (f, v), xytext=(0, 4), textcoords="offset points",
                        ha="center", fontsize=8, color=TEXTO_2)
    ax.set_ylim(0, tot["validaciones"].max() * 1.18)
    ax.legend(frameon=False, fontsize=9, ncol=3, loc="upper left")
    fig.tight_layout()
    fig.savefig(RES / "fig1_totales_diarios.png", facecolor=SUPERFICIE)
    plt.close(fig)
    return tot


def fig_perfil_tipo_dia(d):
    # Promedio del sistema por intervalo y tipo de día
    sistema = d.groupby(["fecha", "tipo_dia", "minuto_dia"], as_index=False)["validaciones"].sum()
    perfil = sistema.groupby(["tipo_dia", "minuto_dia"], as_index=False)["validaciones"].mean()
    fig, ax = nueva_figura(10, 4.2)
    for i, t in enumerate(ORDEN_TIPOS):
        s = perfil[perfil["tipo_dia"] == t].sort_values("minuto_dia")
        ax.plot(s["minuto_dia"] / 60, s["validaciones"], color=SERIE[i], lw=2, label=t)
    estilo_ejes(ax, "Perfil promedio de validaciones cada 15 minutos (todo el sistema troncal)",
                xlabel="Hora del día", ylabel="Validaciones por intervalo de 15 min")
    ax.yaxis.set_major_formatter(FuncFormatter(miles))
    ax.set_xticks(range(0, 25, 2))
    ax.set_xlim(3.5, 24)
    ax.legend(frameon=False, fontsize=9, loc="upper right")
    fig.tight_layout()
    fig.savefig(RES / "fig2_perfil_tipo_dia.png", facecolor=SUPERFICIE)
    plt.close(fig)
    return perfil


def fig_mapa_calor(d, estacion):
    s = d[d["estacion"] == estacion]
    s = s[(s["hora"] >= 4)]
    tabla = s.pivot_table(index="fecha", columns="minuto_dia", values="validaciones", aggfunc="sum")
    fig, ax = nueva_figura(10, 5.2)
    im = ax.imshow(tabla.values, aspect="auto", cmap="Blues", interpolation="nearest")
    ax.set_title(f"Validaciones por intervalo · {estacion} · agosto de 2026",
                 loc="left", color=TEXTO, fontsize=12, fontweight="bold", pad=12)
    cols = tabla.columns.values
    pos = [i for i, m in enumerate(cols) if m % 120 == 0]
    ax.set_xticks(pos)
    ax.set_xticklabels([f"{cols[i] // 60:02d}:00" for i in pos])
    ax.set_yticks(range(len(tabla.index)))
    dias = ["Lun", "Mar", "Mié", "Jue", "Vie", "Sáb", "Dom"]
    ax.set_yticklabels([f"{dias[f.dayofweek]} {f.day:02d}" for f in tabla.index], fontsize=7)
    ax.tick_params(colors=TEXTO_2, labelsize=8, length=0)
    for lado in ax.spines.values():
        lado.set_visible(False)
    cb = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.01)
    cb.set_label("Validaciones / 15 min", color=TEXTO_2, fontsize=8)
    cb.outline.set_visible(False)
    cb.ax.tick_params(colors=TEXTO_2, labelsize=8)
    fig.tight_layout()
    fig.savefig(RES / "fig3_mapa_calor_estacion.png", facecolor=SUPERFICIE)
    plt.close(fig)


def main():
    d = cargar_largo()
    tot = fig_totales_diarios(d)
    perfil = fig_perfil_tipo_dia(d)

    top = d.groupby("estacion")["validaciones"].sum().sort_values(ascending=False)
    fig_mapa_calor(d, top.index[0])

    umb = umbrales_sobredemanda(d)
    nombres = d.drop_duplicates("cod_estacion").set_index("cod_estacion")[["estacion", "linea"]]
    tabla_umb = nombres.join(umb).sort_values("umbral", ascending=False)
    tabla_umb.to_csv(RES / "umbrales_sobredemanda.csv")

    # Resumen numérico
    print("Promedio diario por tipo de día:")
    print(tot.groupby("tipo_dia")["validaciones"].agg(["mean", "count"]).round(0))
    habil = perfil[perfil["tipo_dia"] == "Hábil"].set_index("minuto_dia")["validaciones"]
    am = habil[habil.index < 720].idxmax(); pm = habil[habil.index >= 720].idxmax()
    print(f"Pico AM hábil: {am // 60:02d}:{am % 60:02d} ({habil[am]:,.0f})  "
          f"Pico PM hábil: {pm // 60:02d}:{pm % 60:02d} ({habil[pm]:,.0f})")
    print("\nTop 10 estaciones por umbral de sobredemanda (validaciones / 15 min):")
    print(tabla_umb.head(10))


if __name__ == "__main__":
    main()
