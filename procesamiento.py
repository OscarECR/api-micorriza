"""Procesamiento one-hot para modelo. Compatible con dataset_limpio.csv."""
import re
import pandas as pd
from config_columnas import (
    FORMA_OPCIONES, FORMA_A_ESPORA, COLOR_OPCIONES, PAREDES_OPCIONES,
    MELZER_OPCIONES, MELZER_A_COL, TEXTURA_OPCIONES, TEXTURA_A_COL,
    CONEXION_HIFAL_OPCIONES
)

def parsear_rango_tamano(texto: str) -> tuple[float, float, float]:
    numeros = re.findall(r'\d+(?:\.\d+)?', (texto or "").replace(',', '.'))
    if len(numeros) >= 2:
        tam_min, tam_max = float(numeros[0]), float(numeros[1])
    elif len(numeros) == 1:
        tam_min = tam_max = float(numeros[0])
    else:
        return 0.0, 0.0, 0.0
    return tam_min, tam_max, (tam_min + tam_max) / 2

def crear_dataframe_entrada(tamano_texto: str, forma: list, color: list, paredes: str, melzer: str, conexion: list, textura: list) -> pd.DataFrame:
    tam_min, tam_max, tam_promedio = parsear_rango_tamano(tamano_texto)
    fila = {
        "tam_min": tam_min,
        "tam_max": tam_max,
        "tam_promedio": tam_promedio,
        "Numero de paredes": float(paredes) if paredes and paredes.isdigit() else 1.0
    }
    
    # Forma -> espora_X
    for opc in FORMA_OPCIONES:
        fila[FORMA_A_ESPORA.get(opc, f"espora_{opc}")] = 1 if opc in forma else 0
    
    # Color
    for opc in COLOR_OPCIONES:
        fila[f"color_{opc}"] = 1 if opc in color else 0
    fila["color_sin_reporte"] = 0 if color else 1
    
    # Textura -> tex_sup_X, tex_est_X, tex_cons_X, tex_orn_X
    for opc in TEXTURA_OPCIONES:
        col = TEXTURA_A_COL.get(opc)
        if col:
            fila[col] = 1 if opc in textura else 0
    # Asegurar todas las columnas de textura (0 si no seleccionada)
    for col in TEXTURA_A_COL.values():
        fila.setdefault(col, 0)
    
    # Melzer
    for opc in MELZER_OPCIONES:
        fila[MELZER_A_COL.get(opc, f"melzer_{opc.replace(' ', '_')}")] = 1 if opc == melzer else 0
    
    # Conexión hifal - crear columnas one-hot si el modelo las necesita
    # Nota: El dataset tiene conexion_token como JSON, pero el modelo podría esperar columnas one-hot
    # Si el modelo no las usa, se agregarán como 0 cuando se alineen las columnas
    for opc in CONEXION_HIFAL_OPCIONES:
        fila[opc] = 1 if opc in conexion else 0
    
    return pd.DataFrame([fila])
