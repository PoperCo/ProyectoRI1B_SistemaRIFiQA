import json
import gzip
from functools import lru_cache

from clasicos import (
    buscar_bm25,
    buscar_jaccard,
    buscar_tfidf,
)
from configuracion import CARPETA_DATOS
from semantico import archivo_disponible, buscar_semantico, cargar_indice_semantico
from texto import leer_documentos
from configuracion import (MODELO_EMBEDDINGS, REVISION_MODELO, TAMANO_FRAGMENTO,
                           SOLAPAMIENTO, VERSION_TFIDF, REVISION_FIQA,
                           VERSION_PREPROCESAMIENTO)


NOMBRES_MODELOS = ["Jaccard", "TF-IDF", "BM25", "Embeddings"]


@lru_cache(maxsize=1)
def cargar_sistema():
    ruta_meta = CARPETA_DATOS / "metadatos.json"
    ruta_index = CARPETA_DATOS / "indice_clasico.json.gz"
    ruta_sem_meta = CARPETA_DATOS / "metadatos_embeddings.json"
    for ruta in (ruta_meta, ruta_index, ruta_sem_meta,
                 CARPETA_DATOS / "indice_faiss.index", CARPETA_DATOS / "fragmentos.json"):
        if not archivo_disponible(ruta):
            raise RuntimeError(f"Falta el artefacto preparado {ruta.name}; ejecute preparar_datos.py localmente")
    meta = json.loads(ruta_meta.read_text(encoding="utf-8"))
    if meta.get("documentos") != 57638:
        raise RuntimeError("La aplicación requiere artefactos del corpus completo FiQA (57.638 documentos)")
    with gzip.open(ruta_index, "rt", encoding="utf-8") as archivo:
        classic_meta = json.load(archivo)
    if (classic_meta.get("n_documentos") != 57638 or
            classic_meta.get("version_tfidf") != VERSION_TFIDF or
            classic_meta.get("corpus_revision") != REVISION_FIQA or
            classic_meta.get("normalizacion") != VERSION_PREPROCESAMIENTO):
        raise RuntimeError("El índice léxico es antiguo o corresponde a otro corpus/preprocesamiento")
    sem_meta = json.loads(ruta_sem_meta.read_text(encoding="utf-8"))
    expected = {"modelo": MODELO_EMBEDDINGS, "revision": REVISION_MODELO,
                "fragmento_palabras": TAMANO_FRAGMENTO, "solapamiento": SOLAPAMIENTO,
                "pooling": "CLS", "consulta_sin_instruccion": True,
                "normalizacion": "L2"}
    if any(sem_meta.get(k) != v for k, v in expected.items()):
        raise RuntimeError("El índice semántico no coincide con BGE y su configuración actual")
    documentos = leer_documentos(CARPETA_DATOS / "corpus.jsonl.gz")
    if len(documentos) != 57638:
        raise RuntimeError("corpus.jsonl no contiene los 57.638 documentos esperados")
    por_id = {}

    for documento in documentos:
        por_id[documento["id"]] = documento

    return {
        "documentos": documentos,
        "por_id": por_id,
        "clasico": classic_meta,
        "semantico": cargar_indice_semantico(documentos),
    }


def buscar(nombre_modelo, consulta, limite, sistema=None, completo=False):
    if not consulta.strip():
        return []

    if sistema is None:
        sistema = cargar_sistema()

    clasico = sistema["clasico"]
    documentos = sistema["documentos"]

    if nombre_modelo == "Jaccard":
        posiciones = buscar_jaccard(consulta, clasico, limite)
    elif nombre_modelo == "TF-IDF":
        posiciones = buscar_tfidf(consulta, clasico, documentos, limite)
    elif nombre_modelo == "BM25":
        posiciones = buscar_bm25(consulta, clasico, len(documentos), limite)
    else:
        posiciones = buscar_semantico(consulta, sistema["semantico"], limite, completo=completo)

    resultados = []

    for identificador, puntaje in posiciones:
        documento = sistema["por_id"][identificador]
        resultados.append({
            "id": identificador,
            "titulo": documento["titulo"],
            "texto": documento["texto"],
            "puntaje": puntaje,
        })

    return resultados


@lru_cache(maxsize=1)
def cargar_evaluacion():
    meta_path = CARPETA_DATOS / "metadatos.json"
    if not meta_path.exists() or json.loads(meta_path.read_text(encoding="utf-8")).get("documentos") != 57638:
        raise RuntimeError("Las métricas disponibles no corresponden al corpus completo FiQA")
    with gzip.open(CARPETA_DATOS / "metricas.json.gz", "rt", encoding="utf-8") as archivo:
        datos = json.load(archivo)
    if "splits" not in datos or not all(s in datos["splits"] for s in ("train", "dev", "test")):
        raise RuntimeError("Faltan las métricas completas train/dev/test; ejecute preparar_datos.py")
    return datos
