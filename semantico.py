import json
import os
from pathlib import Path
import shutil
import tempfile
import urllib.request

import faiss
import numpy as np

from configuracion import (CARPETA_DATOS, CARPETA_MODELO, MODELO_EMBEDDINGS,
                           REVISION_FIQA, REVISION_MODELO, SOLAPAMIENTO, TAMANO_FRAGMENTO,
                           TAMANO_LOTE_EMBEDDINGS)
from texto import texto_documento

CARPETA_ONNX = CARPETA_MODELO / f"bge_onnx_{REVISION_MODELO[:12]}"


def archivo_disponible(ruta):
    return ruta.exists() or bool(list(ruta.parent.glob(ruta.name + ".part*")))


def restaurar_archivo(ruta):
    if ruta.exists():
        return ruta
    partes = sorted(ruta.parent.glob(ruta.name + ".part*"))
    if not partes:
        raise FileNotFoundError(f"Falta {ruta.name}")
    destino = Path(tempfile.gettempdir()) / f"fiqa_{os.getpid()}" / ruta.name
    destino.parent.mkdir(parents=True, exist_ok=True)
    tamano = sum(parte.stat().st_size for parte in partes)
    if destino.exists() and destino.stat().st_size == tamano:
        return destino
    temporal = destino.with_suffix(destino.suffix + ".tmp")
    with temporal.open("wb") as salida:
        for parte in partes:
            with parte.open("rb") as entrada:
                shutil.copyfileobj(entrada, salida)
    os.replace(temporal, destino)
    return destino


def dividir_en_fragmentos(documentos):
    fragmentos, identificadores = [], []
    avance = TAMANO_FRAGMENTO - SOLAPAMIENTO
    for documento in documentos:
        palabras = texto_documento(documento).split()
        titulo = (documento.get("titulo") or "").strip()
        for inicio in range(0, len(palabras), avance):
            parte = palabras[inicio:inicio + TAMANO_FRAGMENTO]
            if parte:
                contenido = " ".join(parte)
                fragmentos.append(f"{titulo}. {contenido}" if titulo else contenido)
                identificadores.append(str(documento["id"]))
            if inicio + TAMANO_FRAGMENTO >= len(palabras):
                break
    return fragmentos, identificadores


def _atomic_json(path, value):
    tmp = Path(str(path) + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(value, f, ensure_ascii=False, indent=2)
        f.flush(); os.fsync(f.fileno())
    os.replace(tmp, path)


def _configuracion(cantidad):
    return {"corpus_revision": REVISION_FIQA, "modelo": MODELO_EMBEDDINGS, "revision": REVISION_MODELO,
            "fragmento_palabras": TAMANO_FRAGMENTO, "solapamiento": SOLAPAMIENTO,
            "consulta_sin_instruccion": True, "pooling": "CLS",
            "normalizacion": "L2", "cantidad_fragmentos": cantidad}


def _cargar_runtime(preparar=False):
    os.environ.setdefault("HF_HOME", str(Path(tempfile.gettempdir()) / "fiqa_huggingface"))
    try:
        import onnxruntime as ort
        from transformers import AutoTokenizer
    except ImportError as exc:
        raise RuntimeError("Faltan onnxruntime y transformers para BGE ONNX") from exc
    CARPETA_MODELO.mkdir(parents=True, exist_ok=True)
    providers = ort.get_available_providers()
    preferido = next((p for p in ("CUDAExecutionProvider", "DmlExecutionProvider", "CPUExecutionProvider")
                      if p in providers), "CPUExecutionProvider")
    manifest = CARPETA_ONNX / "bundle.json"
    if not CARPETA_ONNX.exists() and preparar:
        stage = Path(tempfile.mkdtemp(prefix=CARPETA_ONNX.name + ".staging-", dir=CARPETA_MODELO))
        try:
            archivos_modelo = ["onnx/model.onnx", "config.json", "tokenizer.json",
                               "tokenizer_config.json", "vocab.txt", "special_tokens_map.json"]
            hashes = {}
            for archivo_modelo in archivos_modelo:
                origen = f"https://huggingface.co/{MODELO_EMBEDDINGS}/resolve/{REVISION_MODELO}/{archivo_modelo}"
                destino = stage / archivo_modelo
                destino.parent.mkdir(parents=True, exist_ok=True)
                req = urllib.request.Request(origen, headers={"User-Agent": "fiqa-bge-onnx/1.0"})
                import hashlib
                digest, escritos = hashlib.sha256(), 0
                with urllib.request.urlopen(req, timeout=120) as response, destino.open("wb") as output:
                    total = int(response.headers.get("Content-Length", 0))
                    while chunk := response.read(1024 * 1024):
                        output.write(chunk); digest.update(chunk); escritos += len(chunk)
                        if total and (escritos == total or escritos % (20 * 1024 * 1024) < 1024 * 1024):
                            print(f"BGE {Path(archivo_modelo).name}: {escritos:,}/{total:,} bytes", flush=True)
                    output.flush(); os.fsync(output.fileno())
                if total and escritos != total:
                    raise IOError(f"Descarga BGE incompleta: {archivo_modelo} ({escritos}/{total})")
                hashes[archivo_modelo] = {"bytes": escritos, "sha256": digest.hexdigest()}
            source_onnx = stage / "onnx" / "model.onnx"
            os.replace(source_onnx, stage / "model.onnx")
            source_onnx.parent.rmdir()
            (stage / "bundle.json").write_text(json.dumps({"modelo": MODELO_EMBEDDINGS,
                "revision": REVISION_MODELO, "archivos": hashes}, indent=2), encoding="utf-8")
            # Validate the staged graph before publishing the model/tokenizer bundle.
            _crear_sesion(ort, stage / "model.onnx", preferido)
            os.replace(stage, CARPETA_ONNX)
        except Exception:
            shutil.rmtree(stage, ignore_errors=True)
            raise
    if not CARPETA_ONNX.exists() or not archivo_disponible(CARPETA_ONNX / "model.onnx"):
        raise FileNotFoundError("No existe el modelo BGE ONNX preparado")
    if not manifest.exists():
        raise RuntimeError("El bundle BGE ONNX está incompleto; conserve los datos y vuelva a preparar el modelo")
    model_manifest = json.loads(manifest.read_text(encoding="utf-8"))
    if model_manifest.get("modelo") != MODELO_EMBEDDINGS or model_manifest.get("revision") != REVISION_MODELO:
        raise RuntimeError("El modelo ONNX persistido no coincide con la revisión configurada")
    modelo = _crear_sesion(ort, restaurar_archivo(CARPETA_ONNX / "model.onnx"), preferido)
    actual = modelo.get_providers()
    provider = actual[0] if actual else "CPUExecutionProvider"
    tokenizer = AutoTokenizer.from_pretrained(CARPETA_ONNX, local_files_only=True)
    return modelo, tokenizer, provider


def _crear_sesion(ort, model_path, preferred):
    available = ort.get_available_providers()
    candidates = [preferred] + [p for p in ("CUDAExecutionProvider", "DmlExecutionProvider", "CPUExecutionProvider")
                                 if p != preferred and p in available]
    last_error = None
    for provider in candidates:
        options = ort.SessionOptions()
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        options.intra_op_num_threads = 1
        if provider == "DmlExecutionProvider":
            options.enable_mem_pattern = False
            options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        try:
            session = ort.InferenceSession(str(model_path), sess_options=options, providers=[provider])
            actual = session.get_providers()
            if actual and actual[0] == provider:
                print(f"ONNX Runtime activo: {provider}", flush=True)
                return session
        except Exception as exc:
            last_error = exc
            print(f"{provider} no está listo; se prueba el siguiente proveedor: {exc}", flush=True)
    raise RuntimeError(f"No se pudo crear una sesión ONNX (último error: {last_error})")


def _codificar(textos, runtime, batch_size=TAMANO_LOTE_EMBEDDINGS):
    modelo, tokenizer, _provider = runtime
    partes = []
    for inicio in range(0, len(textos), batch_size):
        encoded = tokenizer(textos[inicio:inicio + batch_size], padding=True,
                            truncation=True, return_tensors="np")
        input_names = {item.name for item in modelo.get_inputs()}
        feeds = {name: values for name, values in encoded.items() if name in input_names}
        output_names = [item.name for item in modelo.get_outputs()]
        outputs = modelo.run(output_names, feeds)
        # BGE uses the first token ([CLS]) and L2 normalization.
        hidden = np.asarray(outputs[0])
        if hidden.ndim != 3:
            raise ValueError(f"BGE ONNX debía devolver tokens [batch, seq, dim], no {hidden.shape}")
        vector = hidden[:, 0, :].astype("float32")
        norms = np.linalg.norm(vector, axis=1, keepdims=True)
        partes.append(vector / np.maximum(norms, 1e-12))
    if not partes:
        return np.empty((0, 0), dtype="float32")
    return np.concatenate(partes, axis=0)


def preparar_indice_semantico(documentos):
    CARPETA_DATOS.mkdir(parents=True, exist_ok=True)
    fragmentos, ids = dividir_en_fragmentos(documentos)
    if not fragmentos:
        raise ValueError("El corpus no contiene fragmentos con texto")
    metadata = _configuracion(len(fragmentos))
    meta_path = CARPETA_DATOS / "metadatos_embeddings.json"
    state_path = CARPETA_DATOS / "avance_embeddings.json"
    part_path = CARPETA_DATOS / "embeddings.parcial.npy"
    final_vec_path = CARPETA_DATOS / "embeddings.npy"
    id_path = CARPETA_DATOS / "fragmentos.json"

    # Persist ordering before any vector is written; every vector row maps to this ID list.
    tmp_ids = Path(str(id_path) + ".tmp")
    tmp_ids.write_text(json.dumps(ids), encoding="utf-8")
    os.replace(tmp_ids, id_path)

    # Reuse validated final artifacts before loading or exporting the model.
    if final_vec_path.exists() and meta_path.exists():
        try:
            old_meta = json.loads(meta_path.read_text(encoding="utf-8"))
            old_vectors = np.load(final_vec_path, mmap_mode="r")
            dimension = int(old_meta["dimension"])
            if all(old_meta.get(k) == v for k, v in metadata.items()) and old_vectors.shape == (len(ids), dimension):
                index_path = CARPETA_DATOS / "indice_faiss.index"
                if index_path.exists():
                    old_index = faiss.read_index(str(index_path))
                    if old_index.ntotal == len(ids) and old_index.d == dimension:
                        return
                _construir_faiss(final_vec_path, len(ids), dimension)
                return
        except (ValueError, OSError, KeyError, json.JSONDecodeError):
            pass
    runtime = _cargar_runtime(preparar=True)
    modelo, tokenizer, provider = runtime
    dimension_value = modelo.get_outputs()[0].shape[-1]
    if not isinstance(dimension_value, int):
        dimension_value = json.loads((CARPETA_ONNX / "config.json").read_text(encoding="utf-8"))["hidden_size"]
    dimension = int(dimension_value)
    metadata["dimension"] = dimension
    metadata["proveedor_onnx"] = provider

    if final_vec_path.exists() and meta_path.exists():
        try:
            old_meta = json.loads(meta_path.read_text(encoding="utf-8"))
            old_vec = np.load(final_vec_path, mmap_mode="r")
            if all(old_meta.get(k) == v for k, v in metadata.items()) and old_vec.shape == (len(ids), dimension):
                _construir_faiss(final_vec_path, len(ids), dimension)
                return
        except (ValueError, OSError, json.JSONDecodeError):
            pass

    inicio = 0
    if part_path.exists() and state_path.exists():
        arr = None
        try:
            state = json.loads(state_path.read_text(encoding="utf-8"))
            arr = np.load(part_path, mmap_mode="r+")
            if (state.get("config") == metadata and arr.shape == (len(ids), dimension)
                    and 0 <= int(state.get("confirmados", -1)) <= len(ids)):
                inicio = int(state["confirmados"])
            else:
                inicio = -1
            arr._mmap.close()
            del arr
            if inicio < 0:
                part_path.unlink(missing_ok=True)
                state_path.unlink(missing_ok=True)
                inicio = 0
        except (ValueError, OSError, json.JSONDecodeError, KeyError):
            if arr is not None:
                arr._mmap.close()
            part_path.unlink(missing_ok=True); state_path.unlink(missing_ok=True)
            inicio = 0
    if not part_path.exists():
        mm = np.lib.format.open_memmap(part_path, mode="w+", dtype="float32", shape=(len(ids), dimension))
        mm.flush(); del mm
    mm = np.load(part_path, mmap_mode="r+")
    # Repeat the last unconfirmed batch by checkpointing only after durable writes.
    for start in range(inicio, len(ids), TAMANO_LOTE_EMBEDDINGS):
        end = min(start + TAMANO_LOTE_EMBEDDINGS, len(ids))
        vec = _codificar(fragmentos[start:end], runtime)
        if vec.shape != (end - start, dimension):
            raise ValueError(f"Dimensión de lote inesperada: {vec.shape}")
        mm[start:end] = vec
        mm.flush()
        with open(part_path, "r+b") as f:
            os.fsync(f.fileno())
        _atomic_json(state_path, {"config": metadata, "confirmados": end})
        print(f"Embeddings: {end}/{len(ids)}; ONNX provider={provider}", flush=True)
    del mm
    # Validate fully written vector matrix before publishing it.
    vectors = np.load(part_path, mmap_mode="r")
    if vectors.shape != (len(ids), dimension) or not np.isfinite(vectors).all():
        raise ValueError("La matriz de embeddings está incompleta o contiene valores no finitos")
    # Windows refuses to rename a file while its NumPy memmap handle is open.
    vectors._mmap.close()
    del vectors
    os.replace(part_path, final_vec_path)
    _atomic_json(meta_path, metadata)
    _construir_faiss(final_vec_path, len(ids), dimension)
    state_path.unlink(missing_ok=True)


def _construir_faiss(ruta_embeddings, n, dimension):
    vectors = np.load(ruta_embeddings, mmap_mode="r")
    ids = json.loads((CARPETA_DATOS / "fragmentos.json").read_text(encoding="utf-8"))
    if vectors.shape != (n, dimension) or len(ids) != n:
        raise ValueError("No coincide el conteo de vectores, dimensión y fragmentos")
    idx = faiss.IndexFlatIP(dimension)
    idx.add(np.asarray(vectors, dtype="float32"))
    if idx.ntotal != n:
        raise ValueError("FAISS no contiene todos los vectores")
    tmp = CARPETA_DATOS / "indice_faiss.index.tmp"
    faiss.write_index(idx, str(tmp))
    checked = faiss.read_index(str(tmp))
    if checked.ntotal != n or checked.d != dimension:
        tmp.unlink(missing_ok=True)
        raise ValueError("Validación del índice FAISS falló")
    os.replace(tmp, CARPETA_DATOS / "indice_faiss.index")


def cargar_indice_semantico(documentos=None):
    metadata = json.loads((CARPETA_DATOS / "metadatos_embeddings.json").read_text(encoding="utf-8"))
    expected = _configuracion(metadata.get("cantidad_fragmentos", -1))
    if any(metadata.get(k) != v for k, v in expected.items()):
        raise RuntimeError("Los embeddings guardados no coinciden con el modelo/configuración BGE actual")
    modelo, tokenizer, provider = _cargar_runtime()
    indice = faiss.read_index(str(restaurar_archivo(CARPETA_DATOS / "indice_faiss.index")))
    ids = json.loads((CARPETA_DATOS / "fragmentos.json").read_text(encoding="utf-8"))
    if (indice.ntotal != len(ids) or len(ids) != metadata["cantidad_fragmentos"]
            or indice.d != metadata["dimension"]):
        raise ValueError("Los artefactos semánticos no coinciden; ejecute preparar_datos.py")
    return {"modelo": modelo, "tokenizer": tokenizer, "provider": provider,
            "indice": indice, "identificadores": ids}


def buscar_semantico(consulta, indice_semantico, limite, completo=False):
    vector = _codificar([consulta], (indice_semantico["modelo"], indice_semantico["tokenizer"], indice_semantico["provider"]))
    if completo:
        puntajes, posiciones = indice_semantico["indice"].search(vector, indice_semantico["indice"].ntotal)
        mejores = {}
        for puntaje, posicion in zip(puntajes[0], posiciones[0]):
            if posicion < 0:
                continue
            identificador = indice_semantico["identificadores"][posicion]
            if identificador not in mejores:
                mejores[identificador] = float(puntaje)
        return ordenar_puntajes(mejores, limite)

    # Expand the exact FAISS candidate window until enough distinct documents
    # are present. This avoids sorting every fragment for normal web searches.
    total = indice_semantico["indice"].ntotal
    cantidad = min(total, max(limite * 4, 50))
    resultados, vistos = [], set()
    while True:
        puntajes, posiciones = indice_semantico["indice"].search(vector, cantidad)
        resultados, vistos = [], set()
        for score, posicion in zip(puntajes[0], posiciones[0]):
            if posicion < 0:
                continue
            identificador = indice_semantico["identificadores"][posicion]
            if identificador not in vistos:
                vistos.add(identificador)
                resultados.append((identificador, float(score)))
                if len(resultados) >= limite:
                    return resultados
        if cantidad == total:
            return resultados
        cantidad = min(total, cantidad * 2)


def ordenar_puntajes(puntajes, limite):
    return sorted(puntajes.items(), key=lambda fila: (-fila[1], fila[0]))[:limite]
