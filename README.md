# Prototipo: predicción de sobredemanda en estaciones troncales de TransMilenio

Prototipo del proyecto *Sistema Predictivo de Sobredemanda en Estaciones de Transporte
Masivo para la Mitigación de Congestión en Horas Críticas mediante Series Temporales de
Validaciones*, entrenado con las validaciones troncales de **agosto de 2026**.

## Qué hace

1. **Prepara los datos** (`codigo/01_preparar_datos.py`): convierte el reporte
   *Resumen de Validaciones Troncales Intervalo 15 Min* (una columna por día) en una
   tabla larga `estación · fecha · intervalo · validaciones`. Suma los accesos de cada
   estación y descarta las rutas duales.
2. **Explora y define la sobredemanda** (`codigo/02_explorar.py`): grafica los patrones
   diarios y horarios. Calcula para cada estación un **umbral de sobredemanda**: el
   percentil 90 de sus validaciones por intervalo de 15 minutos en días hábiles del
   periodo de entrenamiento. Sirve como aproximación a la capacidad, que no está publicada.
3. **Entrena y evalúa** (`codigo/03_modelar.py`): pronostica, con un día de anticipación,
   las validaciones de cada estación en cada intervalo de 15 minutos. Compara tres modelos:
   - *Ingenuo semanal*: el mismo intervalo del mismo día de la semana anterior.
   - *Perfil histórico*: el promedio de los últimos 3 días del mismo tipo (hábil, sábado
     o domingo/festivo).
   - *LightGBM*: un modelo de gradient boosting global, con variables de calendario, hora y rezagos.

   Entrena con el periodo del 1 al 24 de agosto y prueba con el del 25 al 31. Al final
   reentrena con todo el mes y genera el pronóstico y las alertas para el **martes 1 de
   septiembre de 2026**.

## Cómo ejecutarlo

```bash
pip install -r requirements.txt
python ejecutar_todo.py
```

Para usarlo con otro mes, reemplace `datos/validaciones_troncales_ago2026.xlsx` por el
reporte del mes correspondiente y ajuste las fechas en `codigo/comun.py`.

## Tratamiento de datos atípicos

- **Calle 76** no registra validaciones desde el 22 de agosto (probable cierre).
- **Islandia, Los Laureles y Tibanica - Primavera** empiezan a operar el 16-17 de agosto.
- Los días con un total menor al 10 % de lo normal para esa estación, o iguales a cero,
  se marcan como anómalos. No se usan para entrenar, ni como referencia histórica, ni para evaluar.
- Se excluyen los cicloparqueaderos (bicicleteros y corrales) y los puntos que nunca
  superan 100 validaciones diarias.

## Resultados (`resultados/`)

| Archivo | Contenido |
|---|---|
| `fig1` a `fig7` (PNG) | Gráficas de exploración, evaluación y alertas |
| `umbrales_sobredemanda.csv` | Umbral P90 por estación |
| `pronosticos_semana_prueba.csv` | Valores reales y pronosticados por los tres modelos |
| `error_por_estacion.csv` | Error del modelo LightGBM por estación |
| `alertas_1sep2026.csv` | Intervalos con sobredemanda pronosticada para el 1 de septiembre |
| `resumen_resultados.json` | Métricas y resumen numérico |

## Limitaciones

- Con un solo mes de datos, el modelo no aprende estacionalidad anual (vacaciones,
  diciembre, lluvias). Por eso LightGBM apenas supera al perfil histórico.
- El umbral P90 es relativo a cada estación; no mide capacidad física.
- Las validaciones no incluyen transbordos internos ni evasión.
