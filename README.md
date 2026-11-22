# Sistema de recuperación de información sobre FiQA

Proyecto de primer bimestre de Recuperación de Información.

**Integrantes:** David Cuasquer, Presly Morales y Eddy Sangucho.

**Aplicación pública:** [fiqa-ri-proyecto.vercel.app](https://fiqa-ri-proyecto.vercel.app/)

## Descripción

Esta aplicación busca respuestas en los 57.638 documentos del corpus financiero FiQA de BeIR. Permite consultar cuatro modelos de recuperación —Jaccard, TF-IDF, BM25 y embeddings—, comparar sus rankings y examinar métricas calculadas con los juicios de relevancia (qrels) de `train`, `dev` y `test`.

El repositorio de esta carpeta corresponde a la **versión de ejecución**. Incluye los índices y el modelo ya preparados; por ello, la página se inicia sin volver a procesar el corpus ni a generar embeddings. Los archivos Parquet y TSV originales y los programas de preparación se conservan en la carpeta fuente del proyecto, separada de esta versión de despliegue.

## Funcionalidades

- **Buscar:** acepta una consulta libre o una de las sugerencias del conjunto FiQA, permite elegir un modelo y muestra hasta 5, 10 o 20 documentos ordenados.
- **Comparar modelos:** presenta los resultados de dos o más modelos para una misma consulta e identifica documentos que aparecen en varios rankings.
- **Evaluación:** muestra Precision@10, Recall@10, MAP y NDCG@10 para `train`, `dev`, `test` o el promedio de los tres conjuntos. También agrupa las consultas según su coincidencia de términos con documentos relevantes.
- **Métricas de una consulta:** si la consulta coincide con una pregunta identificada en los qrels, muestra Precision@10, Recall@10 y NDCG@10 para esa pregunta, junto con el MAP de su conjunto. Para una consulta libre sin qrels, muestra el ranking e indica que no hay métricas de relevancia disponibles.

Las sugerencias iniciales son seis ejemplos; al escribir dos o más caracteres, el buscador consulta las 6.648 preguntas disponibles en `train`, `dev` y `test`. Se puede escribir cualquier otra consulta aunque no figure entre ellas.

## Datos y modelos

Se utilizaron los 57.638 documentos de [BeIR/fiqa](https://huggingface.co/datasets/BeIR/fiqa) y los juicios de [BeIR/fiqa-qrels](https://huggingface.co/datasets/BeIR/fiqa-qrels). Los qrels comprenden 5.500 consultas de `train`, 500 de `dev` y 648 de `test`. Se conservaron los 38 documentos sin título ni cuerpo para mantener los identificadores y los juicios originales. Esos documentos no aportan términos ni vectores y, por tanto, no pueden recuperarse por su contenido.

| Modelo | Representación y ordenación |
|---|---|
| Jaccard | Compara los conjuntos de términos de la consulta y cada documento candidato. |
| TF-IDF | Calcula pesos TF-IDF manualmente y ordena mediante similitud del coseno. Utiliza el índice invertido; no emplea `TfidfVectorizer` ni una matriz dispersa de scikit-learn. |
| BM25 | Utiliza frecuencia de términos, frecuencia documental y longitud del documento; sus parámetros son `k1 = 1,2` y `b = 0,7`. |
| Embeddings | Codifica consulta y fragmentos con `BAAI/bge-small-en-v1.5`, y busca vectores normalizados con `FAISS IndexFlatIP`. |

En los tres modelos léxicos, documentos y consultas reciben el mismo preprocesamiento: minúsculas, unificación de apóstrofes, expansión de contracciones frecuentes, tokenización alfanumérica, eliminación de stopwords inglesas y stemming Porter. Se conservan las palabras `no`, `not`, `never`, `neither`, `nor`, `bill` e `interest` para no perder negaciones ni dos términos útiles en finanzas. El modelo BGE usa su propio tokenizador y no recibe stemming ni un prefijo adicional en las consultas.

Los documentos con contenido se dividen en fragmentos de 150 palabras con 30 palabras de solapamiento. El índice semántico contiene 84.238 vectores de 384 dimensiones. Para el ranking de documentos se utiliza la mejor puntuación de sus fragmentos. Los vectores se normalizan con L2 y `IndexFlatIP` realiza una búsqueda exacta por producto interno, equivalente a similitud del coseno sobre esos vectores.

## Resultados de evaluación

Se considera relevante un par consulta-documento cuando su qrel tiene una puntuación mayor que cero. La tabla resume las 648 consultas de `test` sobre el corpus completo:

| Modelo | Precision@10 | Recall@10 | MAP | NDCG@10 |
|---|---:|---:|---:|---:|
| Jaccard | 0,0156 | 0,0651 | 0,0380 | 0,0480 |
| TF-IDF | 0,0526 | 0,2235 | 0,1411 | 0,1726 |
| BM25 | 0,0685 | 0,3111 | 0,2069 | 0,2501 |
| Embeddings | **0,1052** | **0,4357** | **0,3273** | **0,3801** |

En este conjunto, embeddings obtuvo los promedios más altos en las cuatro medidas. BM25 superó a TF-IDF y Jaccard; TF-IDF superó a Jaccard. Estas cifras describen esta implementación y este corpus, no garantizan el mismo orden para cada consulta individual.

Precision@10, Recall@10 y NDCG@10 consideran los diez primeros resultados, independientemente de que la página muestre 5, 10 o 20. La precisión media (AP) de una consulta se calcula sobre su ranking completo; MAP es el promedio de esas AP en un conjunto. Por eso, el MAP que aparece junto a una consulta pertenece a `train`, `dev` o `test`, no es una medida individual de esa pregunta. Los puntajes utilizados para ordenar documentos tienen escalas propias de cada modelo y no equivalen a Precision, Recall ni MAP.

## Estructura de la versión publicada

| Archivo o carpeta | Función |
|---|---|
| `app.py` | Servidor FastAPI, vistas web y rutas `/salud` y `/sugerencias`. |
| `templates/inicio.html` | Interfaz de búsqueda, comparación y evaluación. |
| `motor.py` | Carga los artefactos y conecta cada modelo con la interfaz. |
| `clasicos.py` | Búsquedas Jaccard, TF-IDF y BM25. |
| `semantico.py` | Carga BGE y FAISS y ejecuta la búsqueda semántica. |
| `texto.py` | Normalización, tokenización y lectura del corpus preparado. |
| `configuracion.py` | Parámetros y revisiones usados por los índices. |
| `datos/` | Corpus preparado, índices, relación fragmento-documento, metadatos y métricas. |
| `modelo/` | Modelo ONNX y tokenizador de BGE. |
| `requirements.txt`, `.python-version` | Dependencias de ejecución y versión de Python. |

Algunos artefactos se guardaron comprimidos o divididos en partes para el despliegue. `semantico.py` reúne las partes del índice FAISS y del modelo ONNX al iniciar el servicio; no vuelve a calcular sus vectores. Debe conservarse la estructura de carpetas y todas las partes de cada archivo.

## Ejecución local en Windows

Se requiere Python 3.12 y espacio libre para instalar dependencias y reunir temporalmente los archivos divididos. Desde PowerShell, dentro de esta carpeta:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pip install "uvicorn>=0.30,<1"
.\.venv\Scripts\python.exe -m uvicorn app:app --host 127.0.0.1 --port 8000
```

Después, abrir <http://127.0.0.1:8000/>. El servicio carga el modelo y los índices al iniciarse; la primera apertura puede tardar más que las consultas posteriores. Para detenerlo, pulsar `Ctrl+C`. La ejecución de consultas utiliza CPU y no requiere la GPU empleada durante la preparación.

Rutas disponibles:

| Ruta | Uso |
|---|---|
| `/` | Interfaz de búsqueda, comparación y evaluación. |
| `/docs` | Documentación interactiva de FastAPI. |
| `/salud` | Respuesta básica de estado: `{"estado":"ok"}`. |
| `/sugerencias?q=texto` | Preguntas de FiQA que coinciden con el texto escrito. |

Por ejemplo, se puede abrir `http://127.0.0.1:8000/?vista=buscar&q=retirement&modelo=BM25&n=10` para realizar una búsqueda. La ruta `/salud` confirma que el servicio responde; no mide por sí sola la calidad de los rankings.

## Alcance y reproducibilidad

Esta carpeta permite **consultar y evaluar los artefactos ya construidos**. Para reconstruirlos desde los Parquet y TSV se necesita la carpeta fuente del proyecto, que contiene `preparar_datos.py`, `evaluacion.py`, los requisitos de preparación y los archivos de origen. La configuración del modelo y del preprocesamiento se registra en los metadatos para evitar combinar índices incompatibles.

La aplicación solo puede calcular métricas de relevancia para consultas con qrels. Los 38 documentos sin contenido permanecen en el total de documentos y en los juicios de relevancia, aunque no pueden recuperarse por texto o embeddings. Las métricas presentadas describen el sistema completo; no aíslan experimentalmente el efecto de cada decisión de preprocesamiento.

## Repositorio y actualizaciones

Esta versión ejecutable se publicó en [GitHub](https://github.com/PoperCo/ProyectoRI1B_SistemaRIFiQA) con `datos/`, `modelo/` y `templates/` en la raíz. No se incluyen `.vercel/`, `__pycache__/`, `.venv/` ni archivos `.pyc`. Los fragmentos mayores pesan cerca de 64 MiB y se subieron mediante Git; no es necesario subir el ZIP del proyecto.

Para enviar cambios posteriores desde la carpeta del repositorio, se utilizan:

```powershell
git add .
git commit -m "Actualizar sistema FiQA"
git push
```

El despliegue actual de Vercel se hizo desde la carpeta local. Subir el repositorio a GitHub no lo vincula automáticamente al proyecto de Vercel; esa conexión se realiza aparte en la configuración de Vercel si se desean despliegues automáticos al actualizar GitHub.

## Fuentes

- [Corpus FiQA de BeIR](https://huggingface.co/datasets/BeIR/fiqa).
- [Juicios de relevancia FiQA de BeIR](https://huggingface.co/datasets/BeIR/fiqa-qrels).
- [Modelo `BAAI/bge-small-en-v1.5`](https://huggingface.co/BAAI/bge-small-en-v1.5).
- [Límites de archivos de GitHub](https://docs.github.com/en/repositories/working-with-files/managing-files/adding-a-file-to-a-repository).
- [Integración de repositorios Git con Vercel](https://vercel.com/docs/git).
