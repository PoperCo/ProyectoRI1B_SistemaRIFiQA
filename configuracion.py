from pathlib import Path


CARPETA_PROYECTO = Path(__file__).resolve().parent
CARPETA_DATOS = CARPETA_PROYECTO / "datos"
CARPETA_MODELO = CARPETA_PROYECTO / "modelo"
MODELO_EMBEDDINGS = "BAAI/bge-small-en-v1.5"
REVISION_MODELO = "5c38ec7c405ec4b44b94cc5a9bb96e735b38267a"
TAMANO_FRAGMENTO = 150
SOLAPAMIENTO = 30
K1_BM25 = 1.2
B_BM25 = 0.7
VERSION_TFIDF = "manual-unigram-tf-raw-idf-smoothed-l2-v1"
VERSION_PREPROCESAMIENTO = "lower-apostrophe-expand-alnum-stopwords-porter-v1"
TAMANO_LOTE_EMBEDDINGS = 64

# Revisions pinned in the source project. Corpus files can be supplied locally.
REVISION_FIQA = "979c07a7cb5ccc6ca009792241fa1250b98055dd"
REVISION_QRELS = "252958f2d646e22cab6d0c72dd3f0d5de6d0655a"
