import json
import gzip
import heapq
import math
import os
from collections import Counter, defaultdict
from pathlib import Path

from configuracion import (B_BM25, CARPETA_DATOS, K1_BM25, VERSION_TFIDF,
                           REVISION_FIQA, VERSION_PREPROCESAMIENTO)
from texto import EXCEPCIONES_STOPWORDS, contar_terminos, tokenizar


def _guardar_json_atomico(ruta, contenido):
    ruta = Path(ruta)
    temporal = ruta.with_suffix(ruta.suffix + ".tmp")
    with temporal.open("w", encoding="utf-8") as archivo:
        json.dump(contenido, archivo, ensure_ascii=False)
        archivo.flush()
        os.fsync(archivo.fileno())
    os.replace(temporal, ruta)


def preparar_indice_clasico(documentos):
    ruta_salida = CARPETA_DATOS / "indice_clasico.json"
    configuracion = {"n_documentos": len(documentos), "version_tfidf": VERSION_TFIDF,
                     "bm25": {"k1": K1_BM25, "b": B_BM25},
                     "stemming": "Porter", "stopword_exceptions": sorted(EXCEPCIONES_STOPWORDS),
                     "normalizacion": VERSION_PREPROCESAMIENTO,
                     "corpus_revision": REVISION_FIQA}
    if ruta_salida.exists():
        try:
            with ruta_salida.open(encoding="utf-8") as f:
                previo = json.load(f)
            if all(previo.get(k) == v for k, v in configuracion.items()):
                return previo
        except (OSError, json.JSONDecodeError):
            pass
    indice = defaultdict(dict)
    longitudes, terminos_por_documento = {}, {}
    for documento in documentos:
        identificador = str(documento["id"])
        frecuencias = contar_terminos(documento)
        terminos_por_documento[identificador] = dict(frecuencias)
        longitudes[identificador] = sum(frecuencias.values())
        for termino, frecuencia in frecuencias.items():
            indice[termino][identificador] = frecuencia

    n = len(documentos)
    idf = {termino: math.log((n + 1) / (len(postings) + 1)) + 1
           for termino, postings in indice.items()}
    normas = {}
    for documento in documentos:
        identificador = str(documento["id"])
        normas[identificador] = math.sqrt(sum(
            (frecuencia * idf[termino]) ** 2
            for termino, frecuencia in terminos_por_documento[identificador].items()
        ))

    datos = {
        "indice_invertido": dict(indice),
        "longitudes": longitudes,
        "terminos_por_documento": terminos_por_documento,
        "idf": idf,
        "normas_tfidf": normas,
        "longitud_media": sum(longitudes.values()) / n if n else 0.0,
        **configuracion,
    }
    _guardar_json_atomico(CARPETA_DATOS / "indice_clasico.json", datos)
    return datos


def cargar_indice_clasico():
    with gzip.open(CARPETA_DATOS / "indice_clasico.json.gz", "rt", encoding="utf-8") as archivo:
        return json.load(archivo)


def ordenar_puntajes(puntajes, limite):
    return heapq.nsmallest(limite, puntajes.items(), key=lambda fila: (-fila[1], fila[0]))


def buscar_jaccard(consulta, indice, limite):
    terminos = set(tokenizar(consulta))
    candidatos = set().union(*(set(indice["indice_invertido"].get(t, {})) for t in terminos)) if terminos else set()
    puntajes = {}
    for identificador in candidatos:
        documento = set(indice["terminos_por_documento"][identificador])
        union = terminos | documento
        if union:
            puntajes[identificador] = len(terminos & documento) / len(union)
    return ordenar_puntajes(puntajes, limite)


def buscar_tfidf(consulta, indice, documentos, limite):
    frecuencias = Counter(tokenizar(consulta))
    idf = indice["idf"]
    vector = {t: f * idf[t] for t, f in frecuencias.items() if t in idf}
    norma_consulta = math.sqrt(sum(v * v for v in vector.values()))
    if norma_consulta == 0:
        return []
    puntajes = defaultdict(float)
    postings = indice["indice_invertido"]
    for termino, peso in vector.items():
        for identificador, frecuencia in postings[termino].items():
            puntajes[identificador] += peso * frecuencia * idf[termino]
    normas = indice["normas_tfidf"]
    puntajes = {identificador: producto / (norma_consulta * normas[identificador])
                for identificador, producto in puntajes.items() if normas[identificador]}
    return ordenar_puntajes(puntajes, limite)


def buscar_bm25(consulta, indice, total_documentos, limite):
    terminos = set(tokenizar(consulta))
    inverted = indice["indice_invertido"]
    candidatos = set().union(*(set(inverted.get(t, {})) for t in terminos)) if terminos else set()
    avg = indice["longitud_media"] or 1.0
    puntajes = {}
    for identificador in candidatos:
        longitud = indice["longitudes"][identificador]
        score = 0.0
        for termino in terminos:
            postings = inverted.get(termino, {})
            tf = postings.get(identificador, 0)
            if tf:
                df = len(postings)
                idf = math.log(1 + (total_documentos - df + 0.5) / (df + 0.5))
                ajuste = 1 - B_BM25 + B_BM25 * longitud / avg
                score += idf * tf * (K1_BM25 + 1) / (tf + K1_BM25 * ajuste)
        puntajes[identificador] = score
    return ordenar_puntajes(puntajes, limite)
