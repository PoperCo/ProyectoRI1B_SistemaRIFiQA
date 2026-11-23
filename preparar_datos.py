"""Build all FiQA indexes from local Parquet/TSV files (never samples data)."""
import argparse
import csv
import gzip
import json
import os
import shutil
from pathlib import Path

import pyarrow.parquet as pq

from clasicos import preparar_indice_clasico
from configuracion import (B_BM25, CARPETA_DATOS, MODELO_EMBEDDINGS,
                           REVISION_FIQA, REVISION_MODELO, REVISION_QRELS,
                           SOLAPAMIENTO, TAMANO_FRAGMENTO, K1_BM25, VERSION_TFIDF,
                           VERSION_PREPROCESAMIENTO)
from evaluacion import evaluar_todos
from motor import cargar_sistema
from semantico import preparar_indice_semantico
from texto import EXCEPCIONES_STOPWORDS, PALABRAS_VACIAS, leer_documentos

EXPECTED_DOCUMENTS = 57638
SPLITS = ("train", "dev", "test")


def _read_parquet(path):
    return pq.read_table(path).to_pylist()


def _guardar_gzip(ruta):
    """Conserva el archivo de preparación y crea la copia que lee la aplicación web."""
    destino = Path(str(ruta) + ".gz")
    temporal = Path(str(destino) + ".tmp")
    with ruta.open("rb") as entrada, temporal.open("wb") as archivo:
        with gzip.GzipFile(fileobj=archivo, mode="wb", filename="", mtime=0) as salida:
            shutil.copyfileobj(entrada, salida)
    os.replace(temporal, destino)


def _find_one(root, predicate, description):
    found = [p for p in root.rglob("*") if p.is_file() and predicate(p)]
    if len(found) != 1:
        raise FileNotFoundError(f"Se esperaba un archivo {description}; encontrados: {found}")
    return found[0]


def _find_qrels(root, split):
    files = [p for p in root.rglob("*.tsv") if split in p.stem.lower() and "qrel" in str(p).lower()]
    if len(files) == 1:
        return files[0]
    files = [p for p in root.rglob("*.tsv") if split in p.stem.lower()]
    if len(files) == 1:
        return files[0]
    raise FileNotFoundError(f"No se encontró un TSV de qrels único para {split} en {root}")


def _load_qrels(path):
    qrels = {}
    with path.open(encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f, delimiter="\t"):
            if int(row["score"]) > 0:
                qid, did = str(row["query-id"]), str(row["corpus-id"])
                qrels.setdefault(qid, set()).add(did)
    return qrels


def _normalize_corpus(rows):
    docs = []
    seen = set()
    for row in rows:
        doc_id = str(row.get("_id", row.get("id")))
        if doc_id in seen or doc_id in {"None", ""}:
            raise ValueError(f"ID de documento vacío o duplicado: {doc_id}")
        seen.add(doc_id)
        docs.append({"id": doc_id, "titulo": (row.get("title") or "").strip(),
                     "texto": (row.get("text") or "").strip()})
    return sorted(docs, key=lambda d: int(d["id"]) if d["id"].isdigit() else d["id"])


def prepare(input_dir, allow_nonstandard=False):
    root = Path(input_dir).resolve()
    corpus_path = _find_one(root, lambda p: p.suffix == ".parquet" and "corpus" in str(p).lower(), "corpus Parquet")
    query_path = _find_one(root, lambda p: p.suffix == ".parquet" and "quer" in str(p).lower(), "queries Parquet")
    docs = _normalize_corpus(_read_parquet(corpus_path))
    if len(docs) != EXPECTED_DOCUMENTS and not allow_nonstandard:
        raise ValueError(f"Se requieren los {EXPECTED_DOCUMENTS:,} documentos; encontrados {len(docs):,}. Use --allow-nonstandard-corpus solo para diagnóstico.")
    queries = {}
    for row in _read_parquet(query_path):
        qid = str(row.get("_id", row.get("id")))
        queries[qid] = (row.get("text") or "").strip()

    CARPETA_DATOS.mkdir(parents=True, exist_ok=True)
    ruta_corpus = CARPETA_DATOS / "corpus.jsonl"
    with ruta_corpus.open("w", encoding="utf-8") as f:
        for doc in docs:
            f.write(json.dumps(doc, ensure_ascii=False) + "\n")
    _guardar_gzip(ruta_corpus)

    splits = {}
    for split in SPLITS:
        qrel_path = _find_qrels(root, split)
        qrels = _load_qrels(qrel_path)
        cases = [{"id": qid, "texto": queries[qid], "relevantes": sorted(ids)}
                 for qid, ids in sorted(qrels.items(), key=lambda x: int(x[0])) if qid in queries]
        splits[split] = cases
        (CARPETA_DATOS / f"consultas_{split}.json").write_text(
            json.dumps(cases, ensure_ascii=False, indent=2), encoding="utf-8")
        missing_queries = sorted(set(qrels) - set(queries))
        if missing_queries:
            raise ValueError(f"{len(missing_queries)} consultas con qrels de {split} no están en el Parquet de queries")
    # Keep a compatibility name for the existing UI, with the final held-out split.
    (CARPETA_DATOS / "consultas.json").write_text(
        json.dumps(splits["test"], ensure_ascii=False, indent=2), encoding="utf-8")

    empty = [d["id"] for d in docs if not d["titulo"] and not d["texto"]]
    if len(empty) != 38 and not allow_nonstandard:
        raise ValueError(f"Se esperaban 38 documentos sin título ni texto; encontrados {len(empty)}")
    empty_set = set(empty)
    empty_relevance = {}
    for split, cases in splits.items():
        hits = [(case["id"], sorted(set(case["relevantes"]) & empty_set)) for case in cases]
        affected = [(qid, ids) for qid, ids in hits if ids]
        empty_relevance[split] = {
            "juicios_relevantes_sin_contenido": sum(len(ids) for _qid, ids in affected),
            "consultas_con_al_menos_uno": len(affected),
            "ids_en_qrels": sorted({doc_id for _qid, ids in affected for doc_id in ids}),
        }
    settings = {
        "corpus_id": "BeIR/fiqa", "revision_corpus": REVISION_FIQA,
        "revision_qrels": REVISION_QRELS, "documentos": len(docs),
        "documentos_sin_titulo_ni_texto": len(empty), "ids_sin_contenido": empty,
        "excepcion_embeddings": "Sin contenido: se conservan en corpus y qrels; no se genera vector; cuentan en denominadores de métricas.",
        "qrels_relevantes_sin_contenido_por_split": empty_relevance,
        "fuentes_locales": {"corpus": corpus_path.name, "consultas": query_path.name,
                            "qrels": {s: _find_qrels(root, s).name for s in SPLITS}},
        "modelo_embeddings": MODELO_EMBEDDINGS, "revision_modelo": REVISION_MODELO,
        "fragmento_palabras": TAMANO_FRAGMENTO, "solapamiento": SOLAPAMIENTO,
        "consulta_sin_instruccion": True, "pooling": "CLS", "normalizacion_embeddings": "L2",
        "normalizacion": VERSION_PREPROCESAMIENTO,
        "stopwords_sk_learn_excluidas": sorted(EXCEPCIONES_STOPWORDS),
        "stopwords_efectivas": sorted(PALABRAS_VACIAS), "stemming": "Porter, léxico solamente",
        "bm25": {"k1": K1_BM25, "b": B_BM25}, "tfidf_manual": VERSION_TFIDF,
        "conjuntos_consultas": {k: len(v) for k, v in splits.items()},
    }
    (CARPETA_DATOS / "metadatos.json").write_text(json.dumps(settings, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Corpus: {len(docs):,} documentos, {len(empty)} sin contenido; esos IDs quedan sin embeddings.", flush=True)
    print("Preparando índice invertido y TF-IDF manual...", flush=True)
    preparar_indice_clasico(docs)
    _guardar_gzip(CARPETA_DATOS / "indice_clasico.json")
    print("Preparando embeddings BGE ONNX y FAISS (CUDA si está disponible)...", flush=True)
    preparar_indice_semantico(docs)
    cargar_sistema.cache_clear()
    system = cargar_sistema()
    print("Evaluando train, dev y test completos...", flush=True)
    evaluar_todos(splits, system)
    _guardar_gzip(CARPETA_DATOS / "metricas.json")
    print("Preparación local terminada.", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", default="datos", help="Directorio con Parquet de corpus/queries y TSV qrels")
    parser.add_argument("--allow-nonstandard-corpus", action="store_true", help="Permite otro tamaño para diagnóstico; no usar como resultado final")
    args = parser.parse_args()
    prepare(args.input_dir, args.allow_nonstandard_corpus)


if __name__ == "__main__":
    main()
