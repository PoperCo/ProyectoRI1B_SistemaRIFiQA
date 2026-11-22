import json
import gzip
import re
from collections import Counter
from pathlib import Path

from nltk.stem import PorterStemmer


EXCEPCIONES_STOPWORDS = {"no", "not", "never", "neither", "nor", "bill", "interest"}
RUTA_METADATOS = Path(__file__).resolve().parent / "datos" / "metadatos.json"
with RUTA_METADATOS.open(encoding="utf-8") as archivo:
    PALABRAS_VACIAS = set(json.load(archivo)["stopwords_efectivas"])
STEMMER = PorterStemmer()
CONTRACCIONES = {
    "can't": "can not", "cannot": "can not", "won't": "will not",
    "shan't": "shall not", "ain't": "is not", "let's": "let us",
    "don't": "do not", "doesn't": "does not", "didn't": "did not",
    "isn't": "is not", "aren't": "are not", "wasn't": "was not",
    "weren't": "were not", "hasn't": "has not", "haven't": "have not",
    "hadn't": "had not", "wouldn't": "would not", "shouldn't": "should not",
    "couldn't": "could not", "mustn't": "must not",
}
PATRON_CONTRACCION = re.compile(r"\b([a-z]+)'(re|ve|ll|d|m|s|t)\b")
EXPANSIONES = {"re": " are", "ve": " have", "ll": " will", "d": " would", "m": " am", "s": " is", "t": " not"}


def normalizar_texto(texto):
    texto = (texto or "").lower().replace("’", "'").replace("‘", "'").replace("ʼ", "'")
    for contraccion, expansion in CONTRACCIONES.items():
        texto = re.sub(r"\b" + re.escape(contraccion) + r"\b", expansion, texto)
    return PATRON_CONTRACCION.sub(lambda m: m.group(1) + EXPANSIONES[m.group(2)], texto)


def tokenizar(texto, stemming=True):
    palabras = re.findall(r"[a-z0-9]+", normalizar_texto(texto))
    salida = []
    for palabra in palabras:
        if palabra not in PALABRAS_VACIAS:
            salida.append(STEMMER.stem(palabra) if stemming else palabra)
    return salida


def leer_documentos(ruta):
    with gzip.open(ruta, "rt", encoding="utf-8") as archivo:
        return [json.loads(linea) for linea in archivo if linea.strip()]


def texto_documento(documento):
    titulo = (documento.get("titulo") or "").strip()
    contenido = (documento.get("texto") or "").strip()
    return f"{titulo}. {contenido}" if titulo and contenido else titulo or contenido


def contar_terminos(documento):
    return Counter(tokenizar(texto_documento(documento)))
