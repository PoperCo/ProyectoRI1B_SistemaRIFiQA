import json
import math

from configuracion import CARPETA_DATOS
from motor import NOMBRES_MODELOS, buscar
from texto import texto_documento, tokenizar

CORTES = (10,)


def calcular_metricas(resultados, relevantes, corte=10):
    relevantes = set(map(str, relevantes))
    hits, retrieved_relevant, ap_sum = 0, 0, 0.0
    ganancia = 0.0
    for rank, resultado in enumerate(resultados, 1):
        if str(resultado["id"]) in relevantes:
            hits += 1
            ap_sum += hits / rank
            if rank <= corte:
                retrieved_relevant += 1
                ganancia += 1.0 / math.log2(rank + 1)

    ideal = 0.0
    for posicion in range(1, min(corte, len(relevantes)) + 1):
        ideal += 1.0 / math.log2(posicion + 1)

    return {"precision_10": retrieved_relevant / corte,
            "recall_10": retrieved_relevant / len(relevantes) if relevantes else 0.0,
            "ap": ap_sum / len(relevantes) if relevantes else 0.0,
            "ndcg_10": ganancia / ideal if ideal else 0.0}


def clasificar_consulta(consulta, relevantes, sistema):
    query_terms = set(tokenizar(consulta))
    best = 0.0
    for doc_id in relevantes:
        documento = sistema["por_id"].get(str(doc_id))
        if not documento or not query_terms:
            continue
        doc_terms = set(tokenizar(texto_documento(documento)))
        best = max(best, len(query_terms & doc_terms) / len(query_terms))
    return "Poca coincidencia de términos" if best < 0.5 else "Alta coincidencia de términos"


def evaluar_todos(conjuntos, sistema):
    detalle_por_split, resumen_por_split = {}, {}
    ids_sin_contenido = {str(d["id"]) for d in sistema["documentos"]
                         if not (d.get("titulo") or "").strip() and not (d.get("texto") or "").strip()}
    for split, consultas in conjuntos.items():
        detalle = []
        for numero, caso in enumerate(consultas, 1):
            tipo = clasificar_consulta(caso["texto"], caso["relevantes"], sistema)
            for modelo in NOMBRES_MODELOS:
                # AP usa el ranking completo; las otras métricas tienen corte en diez.
                resultados = buscar(modelo, caso["texto"], len(sistema["documentos"]),
                                    sistema, completo=(modelo == "Embeddings"))
                metricas = calcular_metricas(resultados, caso["relevantes"])
                relevantes_sin_contenido = sorted(set(map(str, caso["relevantes"])) & ids_sin_contenido)
                detalle.append({"id": str(caso["id"]), "consulta": caso["texto"],
                                "tipo": tipo, "modelo": modelo,
                                "relevantes_qrels": len(caso["relevantes"]),
                                "relevantes_sin_contenido": relevantes_sin_contenido,
                                "cantidad_relevantes_sin_contenido": len(relevantes_sin_contenido),
                                **metricas})
            if numero % 50 == 0 or numero == len(consultas):
                print(f"Evaluación {split}: {numero}/{len(consultas)} consultas", flush=True)
        resumen = []
        for modelo in NOMBRES_MODELOS:
            filas = [r for r in detalle if r["modelo"] == modelo]
            if filas:
                resumen.append({"modelo": modelo,
                                "precision_10": sum(r["precision_10"] for r in filas) / len(filas),
                                "recall_10": sum(r["recall_10"] for r in filas) / len(filas),
                                "map": sum(r["ap"] for r in filas) / len(filas),
                                "ndcg_10": sum(r["ndcg_10"] for r in filas) / len(filas)})
        tipos = []
        for tipo in ("Poca coincidencia de términos", "Alta coincidencia de términos"):
            for modelo in NOMBRES_MODELOS:
                rows = [r for r in detalle if r["tipo"] == tipo and r["modelo"] == modelo]
                if rows:
                    tipos.append({"tipo": tipo, "modelo": modelo, "consultas": len(rows),
                                  "map": sum(r["ap"] for r in rows) / len(rows),
                                  "ndcg_10": sum(r["ndcg_10"] for r in rows) / len(rows)})
        by_id = {}
        for fila in detalle:
            by_id.setdefault(fila["id"], {})[fila["modelo"]] = fila["ap"]
        case_text = {str(c["id"]): c["texto"] for c in consultas}
        deltas = [{"id": qid, "consulta": case_text[qid],
                   "diferencia_bm25": scores.get("Embeddings", 0) - scores.get("BM25", 0),
                   "diferencia_tfidf": scores.get("Embeddings", 0) - scores.get("TF-IDF", 0)}
                  for qid, scores in by_id.items()]
        deltas.sort(key=lambda r: r["diferencia_bm25"])
        detalle_por_split[split] = detalle
        resumen_por_split[split] = {"corte": 10, "cantidad_consultas": len(consultas),
                                    "resumen": resumen, "detalle": detalle, "tipos": tipos,
                                    "mejoras": list(reversed(deltas[-3:])),
                                    "empeoramientos": deltas[:3], "consultas": consultas}
    salida = {"split_final": "test", "splits": resumen_por_split,
              "corte": 10, "cantidad_consultas": len(conjuntos.get("test", [])),
              **resumen_por_split.get("test", {})}
    tmp = CARPETA_DATOS / "metricas.json.tmp"
    tmp.write_text(json.dumps(salida, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(CARPETA_DATOS / "metricas.json")
    return salida


def cargar_evaluacion():
    with (CARPETA_DATOS / "metricas.json").open(encoding="utf-8") as archivo:
        return json.load(archivo)
