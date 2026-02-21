"""
API para predicción de micorrizas.
Ejecutar desde la raíz del proyecto: uvicorn api_prediccion:app --reload --host 0.0.0.0 --port 8000
"""
import os
import unicodedata
import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from procesamiento import crear_dataframe_entrada

# Rutas relativas a la raíz del proyecto (donde están modelo, encoder, dataset)
DIR_RAIZ = os.path.dirname(os.path.abspath(__file__))

def _buscar_archivo(*nombres_posibles):
    """Busca archivo probando varios nombres posibles."""
    for nombre in nombres_posibles:
        ruta = os.path.join(DIR_RAIZ, nombre)
        if os.path.exists(ruta):
            return ruta
    return None

RUTA_MODELO = _buscar_archivo("modelo_micorriza.pkl", "modelo.pkl") or os.path.join(DIR_RAIZ, "modelo_micorriza.pkl")
RUTA_ENCODER = _buscar_archivo("label_encoder.pkl", "encoder.pkl") or os.path.join(DIR_RAIZ, "label_encoder.pkl")
RUTA_DATASET = _buscar_archivo("dataset_limpio.csv", "dataset_limpio", "dataset.csv") or os.path.join(DIR_RAIZ, "dataset_limpio.csv")

app = FastAPI(title="MicoTax API")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

def _cargar_modelo():
    if not os.path.exists(RUTA_MODELO):
        print(f"AVISO: No se encontró modelo en {RUTA_MODELO}")
        print(f"  Directorio actual: {DIR_RAIZ}")
        print(f"  Archivos .pkl encontrados:")
        try:
            for f in os.listdir(DIR_RAIZ):
                if f.endswith('.pkl'):
                    print(f"    - {f}")
        except Exception as e:
            print(f"    Error listando archivos: {e}")
        return None
    try:
        print(f"Cargando modelo desde: {RUTA_MODELO}")
        return joblib.load(RUTA_MODELO)
    except Exception as e:
        print(f"Error cargando modelo: {e}")
        import traceback
        traceback.print_exc()
        return None

def _cargar_encoder():
    if not os.path.exists(RUTA_ENCODER):
        print(f"AVISO: No se encontró label_encoder en {RUTA_ENCODER}")
        return None
    try:
        print(f"Cargando encoder desde: {RUTA_ENCODER}")
        return joblib.load(RUTA_ENCODER)
    except Exception as e:
        print(f"Error cargando encoder: {e}")
        import traceback
        traceback.print_exc()
        return None

modelo = _cargar_modelo()
encoder = _cargar_encoder()

def _cargar_dataset():
    if not os.path.exists(RUTA_DATASET):
        print(f"AVISO: No se encontró dataset en {RUTA_DATASET}")
        print(f"  Archivos .csv encontrados:")
        try:
            for f in os.listdir(DIR_RAIZ):
                if 'dataset' in f.lower() or f.endswith('.csv'):
                    print(f"    - {f}")
        except Exception as e:
            print(f"    Error listando archivos: {e}")
        return None
    print(f"Cargando dataset desde: {RUTA_DATASET}")
    for enc in ["utf-8", "utf-8-sig", "latin-1", "cp1252"]:
        try:
            df = pd.read_csv(RUTA_DATASET, encoding=enc)
            print(f"  Dataset cargado: {len(df)} filas, {len(df.columns)} columnas")
            return df
        except Exception as e:
            print(f"  Error con encoding {enc}: {e}")
            continue
    print("ERROR: No se pudo cargar el dataset con ningún encoding")
    return None

dataset = _cargar_dataset()

# Mostrar estado al iniciar
print("\n" + "="*60)
print("MicoTax API - Estado de carga:")
print(f"  Modelo: {'✓ Cargado' if modelo is not None else '✗ No encontrado en ' + RUTA_MODELO}")
print(f"  Encoder: {'✓ Cargado' if encoder is not None else '✗ No encontrado en ' + RUTA_ENCODER}")
print(f"  Dataset: {'✓ Cargado (' + str(len(dataset)) + ' filas)' if dataset is not None and len(dataset) > 0 else '✗ No encontrado o vacío en ' + RUTA_DATASET}")
print("="*60 + "\n")

class PrediccionRequest(BaseModel):
    tamano: str
    forma: list[str]
    color: list[str]
    paredes: str
    melzer: str
    conexion: list[str]
    textura: list[str]


@app.get("/estado")
def estado():
    """Verifica si el modelo y recursos están cargados."""
    return {
        "modelo_ok": modelo is not None,
        "encoder_ok": encoder is not None,
        "dataset_ok": dataset is not None and len(dataset) > 0,
    }


@app.post("/predecir")
def predecir(req: PrediccionRequest):
    if modelo is None:
        raise HTTPException(status_code=500, detail=f"Modelo no cargado. Buscado en: {RUTA_MODELO}")
    if encoder is None:
        raise HTTPException(status_code=500, detail=f"Encoder no cargado. Buscado en: {RUTA_ENCODER}")
    
    try:
        df = crear_dataframe_entrada(req.tamano, req.forma, req.color, req.paredes, req.melzer, req.conexion, req.textura)
        
        # Asegurar que todas las columnas del modelo estén presentes
        if hasattr(modelo, "feature_names_in_"):
            cols_esperadas = list(modelo.feature_names_in_)
            for c in cols_esperadas:
                if c not in df.columns:
                    df[c] = 0
            df = df[cols_esperadas]
        else:
            # Si no tiene feature_names_in_, usar las columnas del dataset de entrenamiento
            print("AVISO: Modelo sin feature_names_in_. Usando columnas del DataFrame creado.")
        
        # Obtener probabilidades (confianza)
        probas = None
        if hasattr(modelo, "predict_proba"):
            probas = modelo.predict_proba(df)[0]
            pred_idx = probas.argmax()
            confianza = float(probas[pred_idx]) * 100
        else:
            pred_idx = modelo.predict(df)[0]
            confianza = 100.0  # Si no hay predict_proba, asumir 100%
        
        # Especie predicha
        especie = str(encoder.inverse_transform([pred_idx])[0]) if encoder is not None else str(pred_idx)
        info = _obtener_info_especie(dataset, especie)
        
        # Top especies alternativas (top 5)
        especies_alternativas = []
        if probas is not None and encoder is not None:
            # Obtener índices ordenados por probabilidad descendente
            top_indices = probas.argsort()[::-1][:6]  # Top 6 (incluye la predicha)
            for idx in top_indices:
                prob = float(probas[idx]) * 100
                esp = str(encoder.inverse_transform([idx])[0])
                if esp != especie:  # Excluir la especie principal
                    especies_alternativas.append({
                        "especie": esp,
                        "confianza": round(prob, 2),
                        "info": _obtener_info_especie(dataset, esp)
                    })
                if len(especies_alternativas) >= 5:
                    break
        
        return {
            "especie": especie,
            "confianza": round(confianza, 2),
            "info": info,
            "especies_alternativas": especies_alternativas
        }
    except Exception as e:
        import traceback
        error_detail = f"Error en predicción: {str(e)}\n{traceback.format_exc()}"
        print(error_detail)
        raise HTTPException(status_code=500, detail=error_detail)


def _normalizar(s):
    """Quita acentos para búsqueda insensible."""
    if not s:
        return ""
    nfd = unicodedata.normalize("NFD", str(s).strip().lower())
    return "".join(c for c in nfd if unicodedata.category(c) != "Mn")


def _valor(row, col_name):
    """Obtiene valor de fila. Maneja NaN."""
    if not col_name or col_name not in row.index:
        return ""
    v = row[col_name]
    if pd.isna(v) or str(v).strip() == "" or str(v).lower() == "nan":
        return ""
    return str(v).strip()


def _buscar_col(df, *candidatos):
    """Encuentra columna por nombre exacto o similar (sin acentos)."""
    if df is None or df.empty:
        return None
    cols = {str(c).strip(): c for c in df.columns}
    for cand in candidatos:
        c_clean = _normalizar(cand)
        for orig, col in cols.items():
            if _normalizar(orig) == c_clean or c_clean in _normalizar(orig):
                return col
    return None


def _obtener_info_especie(df, especie):
    if df is None or df.empty:
        return {}
    col_esp = _buscar_col(df, "Nombre cientifico", "nombre_cientifico", "especie")
    if not col_esp:
        return {}
    df = df.copy()
    df[col_esp] = df[col_esp].astype(str).str.strip()
    especie_limpia = str(especie).strip()
    m = df[df[col_esp].str.lower() == especie_limpia.lower()]
    if m.empty:
        m = df[df[col_esp].str.lower().str.contains(especie_limpia.lower(), na=False)]
    if m.empty:
        return {}
    r = m.iloc[0]
    return {
        "vegetacion": _valor(r, _buscar_col(df, "vegetacion", "Vegetación")),
        "habitat": _valor(r, _buscar_col(df, "habitat", "Hábitat")),
        "pais": _valor(r, _buscar_col(df, "Pais", "País")),
        "localidad": _valor(r, _buscar_col(df, "Localidad", "localidad")),
        "informacion": _valor(r, _buscar_col(df, "Información de la especie", "Informacion de la especie", "informacion_especie")),
        "particularidad": _valor(r, _buscar_col(df, "Particularidad", "particularidad")),
    }
