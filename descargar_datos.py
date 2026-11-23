"""Fetch the pinned FiQA Parquet and qrels source files from Hugging Face."""
import hashlib
import json
import os
import urllib.request
from pathlib import Path

from configuracion import CARPETA_DATOS, REVISION_FIQA, REVISION_QRELS

CARPETA_FUENTE = CARPETA_DATOS / "fuente"
URL_FIQA = "https://huggingface.co/datasets/BeIR/fiqa/resolve/"
URL_QRELS = "https://huggingface.co/datasets/BeIR/fiqa-qrels/resolve/"
ARCHIVOS = {
    "corpus/corpus-00000-of-00001.parquet": URL_FIQA + REVISION_FIQA + "/corpus/corpus-00000-of-00001.parquet",
    "queries/queries-00000-of-00001.parquet": URL_FIQA + REVISION_FIQA + "/queries/queries-00000-of-00001.parquet",
    **{f"qrels/{split}.tsv": URL_QRELS + REVISION_QRELS + f"/{split}.tsv"
       for split in ("train", "dev", "test")},
}


def descargar(url, destino):
    destino.parent.mkdir(parents=True, exist_ok=True)
    temporal = Path(str(destino) + ".part")
    req = urllib.request.Request(url, headers={"User-Agent": "fiqa-full-corpus-preparer/1.0"})
    digest, recibidos = hashlib.sha256(), 0
    try:
        with urllib.request.urlopen(req, timeout=120) as respuesta, temporal.open("wb") as archivo:
            total = int(respuesta.headers.get("Content-Length", 0))
            while bloque := respuesta.read(1024 * 1024):
                archivo.write(bloque)
                digest.update(bloque)
                recibidos += len(bloque)
                if total and recibidos == total or recibidos % (20 * 1024 * 1024) < 1024 * 1024:
                    print(f"  {destino.name}: {recibidos:,}/{total:,} bytes", flush=True)
            archivo.flush()
            os.fsync(archivo.fileno())
        if total and recibidos != total:
            raise IOError(f"Descarga incompleta: {recibidos} de {total} bytes")
        os.replace(temporal, destino)
        return {"url": url, "bytes": recibidos, "sha256": digest.hexdigest()}
    except Exception:
        temporal.unlink(missing_ok=True)
        raise


def main():
    manifest = {"corpus_revision": REVISION_FIQA, "qrels_revision": REVISION_QRELS,
                "archivos": {}}
    for relpath, url in ARCHIVOS.items():
        path = CARPETA_FUENTE / relpath
        print(f"Descargando {relpath}...", flush=True)
        info = descargar(url, path)
        manifest["archivos"][relpath] = info
    (CARPETA_FUENTE / "origen.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print("Descarga completa; fuentes guardadas sin modificar en datos/fuente/.", flush=True)


if __name__ == "__main__":
    main()
