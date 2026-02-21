"""
Configuración de columnas para one-hot encoding.
Compatibles con dataset_limpio.csv y modelo entrenado.
"""
FORMA_A_ESPORA = {
    "globosa": "espora_globosa", "subglobosa": "espora_subglobosa", "ovoide": "espora_ovoide",
    "elipsoide": "espora_elipsoide", "piriforme": "espora_piriforme", "clavada": "espora_clavada",
    "cilindrica": "espora_cilindrica", "oblonga": "espora_oblonga", "reniforme": "espora_reniforme",
    "fusiforme": "espora_fusiforme", "triangular": "espora_triangular", "irregular": "espora_irregular",
}
FORMA_OPCIONES = list(FORMA_A_ESPORA.keys())
COLOR_OPCIONES = ["amarillo", "ambar", "blanco", "crema", "dorado", "gris", "hialino", "marron", "miel", "naranja", "negro", "ocre", "rojo", "verdoso"]
PAREDES_OPCIONES = ["1", "2", "3", "4"]
MELZER_A_COL = {"negativo": "melzer_negativo", "positivo": "melzer_positivo", "ninguna": "melzer_ninguna", "paredes internas": "melzer_paredes_internas", "sin reporte": "melzer_sin_reporte"}
MELZER_OPCIONES = list(MELZER_A_COL.keys())
TEXTURA_A_COL = {
    "lisa": "tex_sup_lisa", "rugosa": "tex_sup_rugosa", "aspera": "tex_sup_aspera", "granular": "tex_sup_granular",
    "reticulada": "tex_sup_reticulada", "foveolada": "tex_sup_foveolada", "scrobiculada": "tex_sup_scrobiculada",
    "perforada": "tex_sup_perforada", "alveolada": "tex_sup_alveolada", "laberintiforme": "tex_sup_laberintiforme",
    "verrugosa": "tex_sup_verrugosa", "ornamentada": "tex_sup_ornamentada", "laminada": "tex_est_laminada",
    "membranosa": "tex_est_membranosa", "estratificada": "tex_est_estratificada", "sublaminada": "tex_est_sublaminada",
    "subcapas": "tex_est_subcapas", "rigida": "tex_cons_rigida", "flexible": "tex_cons_flexible",
    "semiflexible": "tex_cons_semiflexible", "fragil": "tex_cons_fragil", "quebradiza": "tex_cons_quebradiza",
    "suave": "tex_cons_suave", "delgada": "tex_cons_delgada", "gruesa": "tex_cons_gruesa",
    "mucilaginosa": "tex_cons_mucilaginosa", "evanescente": "tex_cons_evanescente",
    "verrugas": "tex_orn_verrugas", "espinas": "tex_orn_espinas", "proyecciones": "tex_orn_proyecciones",
    "protuberancias": "tex_orn_protuberancias", "papilas": "tex_orn_papilas", "pustulas": "tex_orn_pustulas",
    "reticulo": "tex_orn_reticulo", "crestas": "tex_orn_crestas", "depresiones": "tex_orn_depresiones",
    "hoyos": "tex_orn_hoyos", "estrias": "tex_orn_estrias",
}
TEXTURA_OPCIONES = list(TEXTURA_A_COL.keys())
CONEXION_HIFAL_OPCIONES = [
    "hifa_suspensora", "hifa_subtendente", "hifa_simple", "pedicelo", "celula_bulbosa", "saco_esporifero",
    "plexo_central", "manto_hifal", "cicatriz", "recta", "curvada", "recurvada", "cilindrica", "acampanada",
    "embudo", "bulbosa", "hinchada", "contraida", "tubular", "ramificada", "multiples", "sin_septo",
    "con_septo", "septo_laminado", "septo_transversal", "septo_curvado", "septo_continuo", "septo_grueso",
    "tapon", "poro_abierto", "poro_cerrado", "poro_ocluido", "poro_septado", "poro_germinacion",
    "sesil", "cicatriz_prominente", "cicatriz_denticulada", "canal", "detritos", "colapso", "desprendimiento"
]
