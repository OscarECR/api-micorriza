"""
API para predicción de micorrizas.
Ejecutar desde la raíz del proyecto:
uvicorn api_prediccion:app --reload --host 0.0.0.0 --port 8000
"""

import os
import unicodedata
import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from procesamiento import crear_dataframe_entrada


# ==========================================================
# CONFIGURACIÓN DE RUTAS
# ==========================================================

DIR_RAIZ = os.path.dirname(os.path.abspath(__file__))

RUTA_IMAGENES = os.path.join(DIR_RAIZ, "imagenes_micorrizas")


def _buscar_archivo(*nombres_posibles):
    for nombre in nombres_posibles:
        ruta = os.path.join(DIR_RAIZ, nombre)
        if os.path.exists(ruta):
            return ruta
    return None


RUTA_MODELO = _buscar_archivo(
    "modelo_micorriza.pkl", "modelo.pkl"
) or os.path.join(DIR_RAIZ, "modelo_micorriza.pkl")

RUTA_ENCODER = _buscar_archivo(
    "label_encoder.pkl", "encoder.pkl"
) or os.path.join(DIR_RAIZ, "label_encoder.pkl")

RUTA_DATASET = _buscar_archivo(
    "dataset_limpio.csv", "dataset.csv"
) or os.path.join(DIR_RAIZ, "dataset_limpio.csv")


# ==========================================================
# FASTAPI CONFIG
# ==========================================================

app = FastAPI(title="MicoTax API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# PUBLICAR CARPETA DE IMÁGENES
if os.path.exists(RUTA_IMAGENES):
    app.mount("/imagenes", StaticFiles(directory=RUTA_IMAGENES), name="imagenes")


@app.get("/")
def home():
    return {
        "mensaje": "MicoTax API funcionando correctamente 🚀",
        "docs": "/docs",
        "estado": "/estado",
    }


# ==========================================================
# CARGA DE RECURSOS
# ==========================================================

def _cargar_modelo():
    if not os.path.exists(RUTA_MODELO):
        print(f"Modelo no encontrado en {RUTA_MODELO}")
        return None
    try:
        return joblib.load(RUTA_MODELO)
    except Exception as e:
        print(f"Error cargando modelo: {e}")
        return None


def _cargar_encoder():
    if not os.path.exists(RUTA_ENCODER):
        print(f"Encoder no encontrado en {RUTA_ENCODER}")
        return None
    try:
        return joblib.load(RUTA_ENCODER)
    except Exception as e:
        print(f"Error cargando encoder: {e}")
        return None


def _cargar_dataset():
    if not os.path.exists(RUTA_DATASET):
        print(f"Dataset no encontrado en {RUTA_DATASET}")
        return None

    for enc in ["utf-8", "utf-8-sig", "latin-1", "cp1252"]:
        try:
            df = pd.read_csv(RUTA_DATASET, encoding=enc)
            print(f"Dataset cargado: {len(df)} filas")
            return df
        except Exception:
            continue

    print("No se pudo cargar el dataset")
    return None


modelo = _cargar_modelo()
encoder = _cargar_encoder()
dataset = _cargar_dataset()


# ==========================================================
# MODELO DE ENTRADA
# ==========================================================

class PrediccionRequest(BaseModel):
    tamano: str
    forma: list[str]
    color: list[str]
    paredes: str
    melzer: str
    conexion: list[str]
    textura: list[str]


# ==========================================================
# ENDPOINTS
# ==========================================================

@app.get("/estado")
def estado():
    return {
        "modelo_ok": modelo is not None,
        "encoder_ok": encoder is not None,
        "dataset_ok": dataset is not None and not dataset.empty,
    }


@app.post("/predecir")
def predecir(req: PrediccionRequest):

    if modelo is None:
        raise HTTPException(status_code=500, detail="Modelo no cargado")

    if encoder is None:
        raise HTTPException(status_code=500, detail="Encoder no cargado")

    try:
        df = crear_dataframe_entrada(
            req.tamano,
            req.forma,
            req.color,
            req.paredes,
            req.melzer,
            req.conexion,
            req.textura,
        )

        # Asegurar orden correcto de columnas
        if hasattr(modelo, "feature_names_in_"):
            columnas = list(modelo.feature_names_in_)
            for col in columnas:
                if col not in df.columns:
                    df[col] = 0
            df = df[columnas]

        # ==========================
        # Predicción con probabilidades
        # ==========================
        if hasattr(modelo, "predict_proba"):

            probas = modelo.predict_proba(df)[0]

            pred_idx = int(probas.argmax())
            confianza = float(probas[pred_idx]) * 100

            especie_principal = str(
                encoder.inverse_transform([pred_idx])[0]
            )

            top_indices = probas.argsort()[::-1][:6]

            alternativas = []

            for idx in top_indices:
                idx = int(idx)
                especie_alt = str(
                    encoder.inverse_transform([idx])[0]
                )

                if especie_alt != especie_principal:
                    alternativas.append({
                        "especie": especie_alt,
                        "confianza": round(float(probas[idx]) * 100, 2),
                        "imagen": _obtener_url_imagen(especie_alt),
                        "info": _obtener_info_especie(dataset, especie_alt)
                    })

                if len(alternativas) >= 5:
                    break

        else:
            pred_idx = int(modelo.predict(df)[0])
            confianza = 100.0
            especie_principal = str(
                encoder.inverse_transform([pred_idx])[0]
            )
            alternativas = []

        return {
            "especie": especie_principal,
            "confianza": round(confianza, 2),
            "imagen": _obtener_url_imagen(especie_principal),
            "info": _obtener_info_especie(dataset, especie_principal),
            "especies_alternativas": alternativas,
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ==========================================================
# FUNCIONES AUXILIARES
# ==========================================================

def _obtener_url_imagen(especie):
    if not especie:
        return ""

    nombre = especie.strip()
    posibles_extensiones = [".png", ".jpg", ".jpeg"]

    for ext in posibles_extensiones:
        ruta_local = os.path.join(
            RUTA_IMAGENES,
            nombre,
            f"{nombre}{ext}"
        )

        if os.path.exists(ruta_local):
            return f"/imagenes/{nombre}/{nombre}{ext}"

    return ""


def _normalizar(s):
    if not s:
        return ""
    nfd = unicodedata.normalize("NFD", str(s).strip().lower())
    return "".join(c for c in nfd if unicodedata.category(c) != "Mn")


def _buscar_col(df, *candidatos):
    if df is None or df.empty:
        return None

    columnas_norm = {
        _normalizar(col): col for col in df.columns
    }

    for cand in candidatos:
        cand_norm = _normalizar(cand)
        for col_norm, col_real in columnas_norm.items():
            if cand_norm in col_norm:
                return col_real

    return None


def _valor(row, col):
    if not col or col not in row.index:
        return ""
    val = row[col]
    if pd.isna(val):
        return ""
    return str(val).strip()


def _obtener_info_especie(df, especie):
    if df is None or df.empty:
        return {}

    col_esp = _buscar_col(df, "nombre cientifico", "especie")

    if not col_esp:
        return {}

    df[col_esp] = df[col_esp].astype(str).str.strip()
    especie_limpia = especie.strip()

    match = df[df[col_esp].str.lower() == especie_limpia.lower()]
    if match.empty:
        return {}

    r = match.iloc[0]

    return {
        "vegetacion": _valor(r, _buscar_col(df, "vegetacion")),
        "habitat": _valor(r, _buscar_col(df, "habitat")),
        "pais": _valor(r, _buscar_col(df, "pais")),
        "localidad": _valor(r, _buscar_col(df, "localidad")),
        "informacion": _valor(
            r,
            _buscar_col(
                df,
                "informacion de la especie",
                "informacion",
                "información"
            )
        ),
        "particularidad": _valor(r, _buscar_col(df, "particularidad")),
    }
