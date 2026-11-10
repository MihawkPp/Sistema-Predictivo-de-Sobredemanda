"""
Paso 3: Entrenamiento y evaluación de modelos de pronóstico.

Tarea: pronosticar, con un día de anticipación, las validaciones de cada estación
en cada intervalo de 15 minutos del día siguiente. Solo se usa información de días
anteriores (no hay fuga de datos del día que se pronostica).

Esquema de validación temporal:
    Entrenamiento: 1 al 24 de agosto de 2026
    Prueba:        25 al 31 de agosto de 2026 (5 hábiles, 1 sábado, 1 domingo)

Modelos comparados:
    1. Ingenuo semanal: mismo intervalo, mismo día de la semana anterior.
    2. Perfil histórico: promedio de los últimos 3 días del mismo tipo (hábil,
       sábado o domingo/festivo) en ese intervalo.
    3. LightGBM: modelo de gradient boosting global (todas las estaciones) con
       variables de calendario, hora y rezagos.
"""
import json
import numpy as np
import pandas as pd
import lightgbm as lgb
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from comun import (cargar_largo, umbrales_sobredemanda, estilo_ejes, nueva_figura, miles,
                   tipo_dia, FESTIVOS_CO, RES, SERIE, TEXTO, TEXTO_2, CRITICO, SUPERFICIE,
                   FIN_ENTRENAMIENTO, INICIO_PRUEBA)

TIPOS = {"Hábil": 0, "Sábado": 1, "Domingo/festivo": 2}
VARIABLES = ["cod_cat", "minuto_dia", "dia_semana", "tipo_dia_cod", "festivo",
             "lag_1d", "lag_7d", "prom_tipo_3", "max_tipo_3", "nivel_7d", "total_dia_ant"]
PARAMS = dict(objective="tweedie", tweedie_variance_power=1.3, learning_rate=0.05,
              num_leaves=63, min_child_samples=40, subsample=0.8, subsample_freq=1,
              colsample_bytree=0.9, reg_lambda=1.0, verbose=-1, seed=42)


def construir_variables(d: pd.DataFrame) -> pd.DataFrame:
    """Todas las variables usan solo días anteriores a la fecha del registro."""
    d = d.sort_values(["cod_estacion", "minuto_dia", "fecha"]).copy()
    d["tipo_dia_cod"] = d["tipo_dia"].map(TIPOS)
    d["cod_cat"] = d["cod_estacion"].astype("category")
    # Valor "limpio": los días anómalos (cierres) no sirven como referencia histórica
    d["v_ref"] = d["validaciones"].where(d["dia_anomalo"] == 0)

    serie = d.set_index("fecha").groupby(["cod_estacion", "minuto_dia"])["v_ref"]
    # Rezagos por fecha calendario (no por posición) para respetar huecos
    ref = d[["cod_estacion", "minuto_dia", "fecha", "v_ref"]]
    for dias, nombre in [(1, "lag_1d"), (7, "lag_7d")]:
        r = ref.copy(); r["fecha"] = r["fecha"] + pd.Timedelta(days=dias)
        d = d.merge(r.rename(columns={"v_ref": nombre}), on=["cod_estacion", "minuto_dia", "fecha"], how="left")

    # Últimos 3 días del mismo tipo (estrictamente anteriores)
    d = d.sort_values(["cod_estacion", "tipo_dia", "minuto_dia", "fecha"])
    g = d.groupby(["cod_estacion", "tipo_dia", "minuto_dia"])["v_ref"]
    d["prom_tipo_3"] = g.transform(lambda s: s.shift(1).rolling(3, min_periods=1).mean())
    d["max_tipo_3"] = g.transform(lambda s: s.shift(1).rolling(3, min_periods=1).max())

    # Nivel de la estación: promedio diario de los 7 días previos, y total del día anterior
    tot = (d.groupby(["cod_estacion", "fecha"], as_index=False)
             .agg(total=("v_ref", lambda s: s.sum(min_count=1))).sort_values(["cod_estacion", "fecha"]))
    tot["nivel_7d"] = tot.groupby("cod_estacion")["total"].transform(
        lambda s: s.shift(1).rolling(7, min_periods=2).mean())
    tot["total_dia_ant"] = tot.groupby("cod_estacion")["total"].shift(1)
    d = d.merge(tot[["cod_estacion", "fecha", "nivel_7d", "total_dia_ant"]], on=["cod_estacion", "fecha"])
    return d.sort_values(["cod_estacion", "timestamp"]).reset_index(drop=True)


def metricas(y, p):
    y, p = np.asarray(y, float), np.asarray(p, float)
    return {"MAE": float(np.mean(np.abs(y - p))),
            "RMSE": float(np.sqrt(np.mean((y - p) ** 2))),
            "WAPE": float(np.sum(np.abs(y - p)) / np.sum(y))}


def metricas_alerta(real, pred):
    real, pred = np.asarray(real, bool), np.asarray(pred, bool)
    vp = (real & pred).sum(); fp = (~real & pred).sum(); fn = (real & ~pred).sum()
    prec = vp / (vp + fp) if vp + fp else 0.0
    rec = vp / (vp + fn) if vp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return {"Precision": float(prec), "Recall": float(rec), "F1": float(f1),
            "Episodios reales": int(real.sum()), "Detectados": int(vp), "Falsas alarmas": int(fp)}


def entrenar(train, n_iter=None, valid=None):
    ds = lgb.Dataset(train[VARIABLES], train["validaciones"], categorical_feature=["cod_cat"])
    if valid is not None:
        dv = lgb.Dataset(valid[VARIABLES], valid["validaciones"], reference=ds)
        m = lgb.train(PARAMS, ds, num_boost_round=3000, valid_sets=[dv],
                      callbacks=[lgb.early_stopping(100, verbose=False)])
        return m, m.best_iteration
    return lgb.train(PARAMS, ds, num_boost_round=n_iter), n_iter


def es_hora_critica(df):
    """Horas críticas: días hábiles 5:00–8:59 y 16:00–19:59."""
    h = df["minuto_dia"] // 60
    return (df["tipo_dia"] == "Hábil") & (h.between(5, 8) | h.between(16, 19))


def fig_real_vs_pronostico(prueba, estaciones, umb):
    fig, ejes = plt.subplots(len(estaciones), 1, figsize=(10, 2.6 * len(estaciones)), dpi=150, sharex=True)
    fig.patch.set_facecolor(SUPERFICIE)
    for ax, (cod, nombre) in zip(ejes, estaciones):
        s = prueba[prueba["cod_estacion"] == cod].sort_values("timestamp")
        ax.plot(s["timestamp"], s["validaciones"], color=TEXTO_2, lw=1.4, label="Real")
        ax.plot(s["timestamp"], s["p_lgbm"], color=SERIE[0], lw=2, label="Pronóstico LightGBM")
        ax.axhline(umb[cod], color=CRITICO, lw=1.2, ls=(0, (4, 3)), label="Umbral de sobredemanda (P90)")
        estilo_ejes(ax, nombre)
        ax.yaxis.set_major_formatter(FuncFormatter(miles))
        ax.title.set_fontsize(10)
    ejes[0].legend(frameon=False, fontsize=8, ncol=3, loc="upper right", bbox_to_anchor=(1, 1.32))
    dias = ["Lun", "Mar", "Mié", "Jue", "Vie", "Sáb", "Dom"]
    fechas = pd.date_range(INICIO_PRUEBA, periods=7)
    ejes[-1].set_xticks(fechas + pd.Timedelta(hours=12))
    ejes[-1].set_xticklabels([f"{dias[f.dayofweek]} {f.day}" for f in fechas])
    ejes[-1].set_xlabel("Semana de prueba · 25 al 31 de agosto de 2026", color=TEXTO_2, fontsize=9)
    fig.supylabel("Validaciones por intervalo de 15 min", color=TEXTO_2, fontsize=9)
    fig.tight_layout()
    fig.savefig(RES / "fig4_real_vs_pronostico.png", facecolor=SUPERFICIE)
    plt.close(fig)


def fig_comparacion(tabla):
    fig, ax = nueva_figura(10, 3.8)
    modelos = tabla.index.tolist()
    x = np.arange(len(modelos)); w = 0.36
    b1 = ax.bar(x - w / 2 - 0.01, tabla["WAPE total"] * 100, w, color=SERIE[0], label="Todo el día")
    b2 = ax.bar(x + w / 2 + 0.01, tabla["WAPE horas críticas"] * 100, w, color=SERIE[1], label="Horas críticas")
    for b in list(b1) + list(b2):
        ax.annotate(f"{b.get_height():.1f} %".replace(".", ","), (b.get_x() + b.get_width() / 2, b.get_height()),
                    xytext=(0, 3), textcoords="offset points", ha="center", fontsize=8, color=TEXTO)
    estilo_ejes(ax, "Error porcentual (WAPE) por modelo · menor es mejor", ylabel="WAPE (%)")
    ax.set_xticks(x); ax.set_xticklabels(modelos, color=TEXTO, fontsize=9)
    ax.set_ylim(0, max(tabla["WAPE total"].max(), tabla["WAPE horas críticas"].max()) * 100 * 1.25)
    ax.legend(frameon=False, fontsize=9, loc="upper right")
    fig.tight_layout()
    fig.savefig(RES / "fig5_comparacion_modelos.png", facecolor=SUPERFICIE)
    plt.close(fig)


def fig_importancia(modelo):
    nombres = {"cod_cat": "Estación", "minuto_dia": "Hora del día", "dia_semana": "Día de la semana",
               "tipo_dia_cod": "Tipo de día", "festivo": "Festivo", "lag_1d": "Mismo intervalo, día anterior",
               "lag_7d": "Mismo intervalo, semana anterior", "prom_tipo_3": "Promedio últimos 3 días del mismo tipo",
               "max_tipo_3": "Máximo últimos 3 días del mismo tipo", "nivel_7d": "Nivel diario de la estación (7 días)",
               "total_dia_ant": "Total de la estación el día anterior"}
    imp = pd.Series(modelo.feature_importance("gain"), index=VARIABLES)
    imp = (imp / imp.sum() * 100).sort_values()
    fig, ax = nueva_figura(10, 4.4)
    ax.barh([nombres[i] for i in imp.index], imp.values, color=SERIE[0], height=0.6)
    for i, v in enumerate(imp.values):
        ax.annotate(f"{v:.1f} %".replace(".", ","), (v, i), xytext=(4, 0), textcoords="offset points", va="center",
                    fontsize=8, color=TEXTO)
    estilo_ejes(ax, "Importancia de las variables en el modelo LightGBM", xlabel="% de la ganancia total")
    ax.grid(axis="y", visible=False); ax.grid(axis="x", color="#e1e0d9", linewidth=0.8)
    ax.tick_params(axis="y", colors=TEXTO)
    fig.tight_layout()
    fig.savefig(RES / "fig6_importancia_variables.png", facecolor=SUPERFICIE)
    plt.close(fig)
    return imp


def pronostico_dia_siguiente(d, n_iter, umb):
    """Reentrena con todo agosto y pronostica el martes 1 de septiembre de 2026."""
    nueva = pd.Timestamp("2026-09-01")
    ult = d[d["fecha"] == d["fecha"].max()]
    base = ult[["cod_estacion", "estacion", "linea", "intervalo", "minuto_dia"]].copy()
    base["fecha"] = nueva
    base["timestamp"] = pd.to_datetime(nueva.strftime("%Y-%m-%d") + " " + base["intervalo"])
    base["validaciones"] = np.nan
    base["tipo_dia"] = tipo_dia(nueva); base["festivo"] = int(nueva in FESTIVOS_CO)
    base["dia_semana"] = nueva.dayofweek; base["hora"] = base["minuto_dia"] // 60
    base["dia_anomalo"] = 0
    cols = ["cod_estacion", "estacion", "linea", "fecha", "intervalo", "validaciones", "timestamp",
            "tipo_dia", "festivo", "dia_semana", "hora", "minuto_dia", "dia_anomalo"]
    todo = construir_variables(pd.concat([d[cols], base[cols]], ignore_index=True))
    entren = todo[(todo["fecha"] < nueva) & (todo["dia_anomalo"] == 0)]
    modelo, _ = entrenar(entren, n_iter=n_iter)
    fut = todo[todo["fecha"] == nueva].copy()
    fut["pronostico"] = np.clip(modelo.predict(fut[VARIABLES]), 0, None)
    # Estaciones sin operación en los últimos 3 días se consideran cerradas: sin pronóstico
    ult3 = d[d["fecha"] > d["fecha"].max() - pd.Timedelta(days=3)].groupby("cod_estacion")["validaciones"].sum()
    cerradas = ult3[ult3 == 0].index
    fut = fut[~fut["cod_estacion"].isin(cerradas)]
    fut["umbral"] = fut["cod_estacion"].map(umb)
    fut["sobredemanda"] = fut["pronostico"] > fut["umbral"]
    return fut, list(d[d["cod_estacion"].isin(cerradas)]["estacion"].unique())


def fig_alertas(alertas):
    a = alertas[alertas["sobredemanda"]].copy()
    a["exceso"] = a["pronostico"] - a["umbral"]
    top = a.groupby("estacion")["exceso"].sum().sort_values(ascending=True).tail(15)
    fig, ax = nueva_figura(10, 4.8)
    ax.barh(top.index, top.values, color=CRITICO, height=0.6)
    for i, v in enumerate(top.values):
        ax.annotate(miles(v), (v, i), xytext=(4, 0), textcoords="offset points", va="center", fontsize=8, color=TEXTO)
    estilo_ejes(ax, "Pasajeros pronosticados por encima del umbral · martes 1 de septiembre de 2026",
                xlabel="Suma del exceso sobre el umbral en los intervalos con alerta (validaciones)")
    ax.xaxis.set_major_formatter(FuncFormatter(miles))
    ax.grid(axis="y", visible=False); ax.grid(axis="x", color="#e1e0d9", linewidth=0.8)
    ax.tick_params(axis="y", colors=TEXTO)
    fig.tight_layout()
    fig.savefig(RES / "fig7_alertas_1sep.png", facecolor=SUPERFICIE)
    plt.close(fig)


def main():
    d = cargar_largo()
    umb = umbrales_sobredemanda(d)
    X = construir_variables(d)

    entren = X[(X["fecha"] <= FIN_ENTRENAMIENTO) & (X["dia_anomalo"] == 0)]
    prueba = X[(X["fecha"] >= INICIO_PRUEBA) & (X["dia_anomalo"] == 0)].copy()

    # Ajuste del número de árboles con una validación interna (1–17 vs 18–24 de agosto)
    corte = pd.Timestamp("2026-08-17")
    _, n_iter = entrenar(entren[entren["fecha"] <= corte], valid=entren[entren["fecha"] > corte])
    modelo, _ = entrenar(entren, n_iter=n_iter)

    prueba["p_lgbm"] = np.clip(modelo.predict(prueba[VARIABLES]), 0, None)
    prueba["p_ingenuo"] = prueba["lag_7d"].fillna(prueba["prom_tipo_3"]).fillna(0)
    prueba["p_perfil"] = prueba["prom_tipo_3"].fillna(prueba["lag_7d"]).fillna(0)
    prueba["umbral"] = prueba["cod_estacion"].map(umb)
    prueba["critica"] = es_hora_critica(prueba)

    nombres = {"p_ingenuo": "Ingenuo semanal", "p_perfil": "Perfil histórico", "p_lgbm": "LightGBM"}
    filas, alertas = {}, {}
    for col, nom in nombres.items():
        t = metricas(prueba["validaciones"], prueba[col])
        c = metricas(prueba.loc[prueba["critica"], "validaciones"], prueba.loc[prueba["critica"], col])
        filas[nom] = {"MAE": t["MAE"], "RMSE": t["RMSE"], "WAPE total": t["WAPE"],
                      "MAE horas críticas": c["MAE"], "WAPE horas críticas": c["WAPE"]}
        alertas[nom] = metricas_alerta(prueba["validaciones"] > prueba["umbral"], prueba[col] > prueba["umbral"])
    tabla = pd.DataFrame(filas).T
    tabla_al = pd.DataFrame(alertas).T

    # Error por estación (LightGBM)
    por_est = (prueba.groupby(["cod_estacion", "estacion"])
               .apply(lambda g: pd.Series(metricas(g["validaciones"], g["p_lgbm"])), include_groups=False)
               .reset_index().sort_values("WAPE"))
    por_est.to_csv(RES / "error_por_estacion.csv", index=False)

    # Gráficas
    top = (prueba.groupby(["cod_estacion", "estacion"])["validaciones"].sum()
           .sort_values(ascending=False).reset_index())
    ejemplos = [tuple(top.iloc[i][["cod_estacion", "estacion"]]) for i in (0, 1, 7)]
    fig_real_vs_pronostico(prueba, ejemplos, umb)
    fig_comparacion(tabla)
    imp = fig_importancia(modelo)

    alertas_sep, cerradas = pronostico_dia_siguiente(d, n_iter, umb)
    fig_alertas(alertas_sep)
    salida_sep = alertas_sep[alertas_sep["sobredemanda"]][["estacion", "linea", "intervalo", "pronostico", "umbral"]]
    salida_sep = salida_sep.assign(exceso_pct=(salida_sep["pronostico"] / salida_sep["umbral"] - 1) * 100)
    salida_sep.sort_values(["estacion", "intervalo"]).round(1).to_csv(RES / "alertas_1sep2026.csv", index=False)
    cols_num = ["validaciones", "p_ingenuo", "p_perfil", "p_lgbm", "umbral"]
    exp = prueba[["estacion", "timestamp", "tipo_dia"] + cols_num].copy()
    exp[cols_num] = exp[cols_num].round(1)
    exp.to_csv(RES / "pronosticos_semana_prueba.csv", index=False)
    resumen_sep = (salida_sep.assign(exceso=salida_sep["pronostico"] - salida_sep["umbral"])
                   .groupby("estacion").agg(intervalos=("intervalo", "size"), desde=("intervalo", "min"),
                                            hasta=("intervalo", "max"), exceso=("exceso", "sum"),
                                            pico=("pronostico", "max"), umbral=("umbral", "first"))
                   .sort_values("exceso", ascending=False))

    resumen = {
        "n_iter": int(n_iter), "estaciones_prueba": int(prueba["cod_estacion"].nunique()),
        "filas_entrenamiento": int(len(entren)), "filas_prueba": int(len(prueba)),
        "metricas": tabla.round(4).to_dict(orient="index"),
        "alertas": tabla_al.round(4).to_dict(orient="index"),
        "importancia": imp.round(2).sort_values(ascending=False).to_dict(),
        "wape_estacion_mediana": float(por_est["WAPE"].median()),
        "mejores_estaciones": por_est.head(5)[["estacion", "WAPE"]].round(3).values.tolist(),
        "peores_estaciones": por_est.tail(5)[["estacion", "WAPE"]].round(3).values.tolist(),
        "ejemplos_fig4": [e[1] for e in ejemplos],
        "sep1_estaciones_con_alerta": int(salida_sep["estacion"].nunique()),
        "sep1_intervalos_alerta": int(len(salida_sep)),
        "sep1_total_pronosticado": float(alertas_sep["pronostico"].sum()),
        "sep1_cerradas": cerradas,
        "sep1_top10": resumen_sep.head(10).round(0).reset_index().values.tolist(),
    }
    (RES / "resumen_resultados.json").write_text(json.dumps(resumen, ensure_ascii=False, indent=2))

    pd.set_option("display.width", 200); pd.set_option("display.max_columns", 20)
    print(f"Árboles: {n_iter}  Filas entrenamiento: {len(entren):,}  Filas prueba: {len(prueba):,}")
    print("\nMétricas de pronóstico (semana 25–31 ago):"); print(tabla.round(3))
    print("\nDetección de sobredemanda:"); print(tabla_al)
    print("\nImportancia:"); print(imp.sort_values(ascending=False).round(1))
    print("\nMejores y peores estaciones (WAPE):"); print(por_est.head(5)); print(por_est.tail(5))
    print(f"\n1-sep: {salida_sep['estacion'].nunique()} estaciones con alerta, {len(salida_sep)} intervalos. "
          f"Cerradas: {cerradas}")


if __name__ == "__main__":
    main()
