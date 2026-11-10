"""
Paso 1: Preparación de datos.

Convierte el reporte de TransMilenio "Resumen de Validaciones Troncales
Intervalo 15 Min" (formato ancho: una columna por día) en una tabla larga:
    estacion | fecha | intervalo | validaciones

Solo se conservan estaciones del componente troncal (Fase 1, 2 y 3);
se descartan las rutas duales. Los accesos de una misma estación se suman.
Se completa la rejilla estación x día x intervalo con ceros, porque el
reporte omite las combinaciones sin validaciones.
"""
from pathlib import Path
import pandas as pd

RAIZ = Path(__file__).resolve().parent.parent
ENTRADA = RAIZ / "datos" / "validaciones_troncales_ago2026.xlsx"
SALIDA = RAIZ / "datos" / "validaciones_15min_largo.csv"


def leer_reporte(ruta: Path) -> pd.DataFrame:
    crudo = pd.read_excel(ruta, sheet_name="Validaciones Tullave", header=None)
    # La fila de encabezados es la que contiene "Fase" en la columna B
    fila_enc = crudo.index[crudo.iloc[:, 1].astype(str).str.strip() == "Fase"][0]
    enc = crudo.iloc[fila_enc].tolist()
    datos = crudo.iloc[fila_enc + 1:].copy()
    datos.columns = enc
    datos = datos.loc[:, [c for c in datos.columns if not (isinstance(c, float) and pd.isna(c))]]
    datos = datos.drop(columns=["Total general"], errors="ignore")
    return datos


def a_formato_largo(datos: pd.DataFrame) -> pd.DataFrame:
    datos = datos[datos["Fase"].isin(["Fase 1", "Fase 2", "Fase 3"])].copy()
    cols_dia = [c for c in datos.columns if isinstance(c, pd.Timestamp) or hasattr(c, "year")]
    largo = datos.melt(
        id_vars=["Fase", "Línea", "Estación", "Acceso de Estación", "Intervalo"],
        value_vars=cols_dia, var_name="fecha", value_name="validaciones",
    )
    largo["validaciones"] = pd.to_numeric(largo["validaciones"], errors="coerce").fillna(0)
    largo["fecha"] = pd.to_datetime(largo["fecha"])
    largo["intervalo"] = largo["Intervalo"].astype(str).str[:5]

    # Separar código y nombre: "(06001) Modelia" -> codigo 06001, nombre Modelia
    ext = largo["Estación"].str.extract(r"\((\d+)\)\s*(.*)")
    largo["cod_estacion"] = ext[0]
    largo["estacion"] = ext[1].str.strip()
    largo["linea"] = largo["Línea"].str.replace(r"^\(\d+\)\s*", "", regex=True)

    # Sumar accesos -> nivel estación
    est = (largo.groupby(["cod_estacion", "estacion", "linea", "fecha", "intervalo"], as_index=False)
                ["validaciones"].sum())
    return est


def completar_rejilla(est: pd.DataFrame) -> pd.DataFrame:
    estaciones = est[["cod_estacion", "estacion", "linea"]].drop_duplicates("cod_estacion")
    fechas = pd.DataFrame({"fecha": sorted(est["fecha"].unique())})
    intervalos = pd.DataFrame({"intervalo": sorted(est["intervalo"].unique())})
    rejilla = estaciones.merge(fechas, how="cross").merge(intervalos, how="cross")
    completo = rejilla.merge(est[["cod_estacion", "fecha", "intervalo", "validaciones"]],
                             on=["cod_estacion", "fecha", "intervalo"], how="left")
    completo["validaciones"] = completo["validaciones"].fillna(0).astype(int)
    completo["timestamp"] = pd.to_datetime(completo["fecha"].dt.strftime("%Y-%m-%d") + " " + completo["intervalo"])
    return completo.sort_values(["cod_estacion", "timestamp"]).reset_index(drop=True)


def main():
    datos = leer_reporte(ENTRADA)
    est = a_formato_largo(datos)
    completo = completar_rejilla(est)
    completo.to_csv(SALIDA, index=False)
    print(f"Filas: {len(completo):,}  Estaciones: {completo['cod_estacion'].nunique()}  "
          f"Días: {completo['fecha'].nunique()}  Intervalos/día: {completo['intervalo'].nunique()}")
    print(f"Rango horario: {completo['intervalo'].min()} - {completo['intervalo'].max()}")
    print(f"Total validaciones: {completo['validaciones'].sum():,}")
    print(f"Guardado en {SALIDA}")


if __name__ == "__main__":
    main()
