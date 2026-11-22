import json
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Query, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

from motor import NOMBRES_MODELOS, buscar, cargar_evaluacion, cargar_sistema


CARPETA_PROYECTO = Path(__file__).resolve().parent
templates = Jinja2Templates(directory=str(CARPETA_PROYECTO / "templates"))


@asynccontextmanager
async def lifespan(_app):
    cargar_sistema()
    cargar_evaluacion()
    yield


app = FastAPI(
    title="FiQA Search",
    description="Búsqueda léxica y semántica sobre el corpus completo FiQA",
    lifespan=lifespan,
)


def leer_limite(valor):
    if valor in (5, 10, 20):
        return valor
    return 10


def elegir_consulta(conjuntos, split, identificador, texto):
    consulta = texto.strip()
    caso_elegido = None

    if identificador:
        for caso in conjuntos[split]["consultas"]:
            if caso["id"] == identificador:
                caso_elegido = caso
                consulta = caso["texto"]
                break

    if caso_elegido is None and consulta:
        coincidencias = []
        for nombre_split, datos in conjuntos.items():
            for caso in datos["consultas"]:
                if caso["texto"].strip() == consulta:
                    coincidencias.append((nombre_split, caso))
        if len(coincidencias) == 1:
            split, caso_elegido = coincidencias[0]

    return consulta, caso_elegido, split


def elegir_sugerencias(conjuntos):
    ejemplos = (("train", "682"),
                ("dev", "451"),
                ("test", "2968"),
                ("train", "661"),
                ("dev", "1325"),
                ("test", "2076"))
    sugerencias = []
    for split, identificador in ejemplos:
        for caso in conjuntos[split]["consultas"]:
            if caso["id"] == identificador:
                sugerencias.append({"id": identificador, "split": split,
                                    "texto": caso["texto"]})
                break
    return sugerencias


def resumir_todos(conjuntos):
    total = 0
    acumulados = {}
    tipos = {}
    for datos in conjuntos.values():
        cantidad = datos["cantidad_consultas"]
        total += cantidad
        for fila in datos["resumen"]:
            modelo = fila["modelo"]
            if modelo not in acumulados:
                acumulados[modelo] = {"modelo": modelo, "precision_10": 0.0,
                                     "recall_10": 0.0, "map": 0.0, "ndcg_10": 0.0}
            for medida in ("precision_10", "recall_10", "map", "ndcg_10"):
                acumulados[modelo][medida] += fila[medida] * cantidad
        for fila in datos["tipos"]:
            clave = (fila["tipo"], fila["modelo"])
            if clave not in tipos:
                tipos[clave] = {"tipo": fila["tipo"], "modelo": fila["modelo"],
                                "consultas": 0, "map": 0.0}
            tipos[clave]["consultas"] += fila["consultas"]
            tipos[clave]["map"] += fila["map"] * fila["consultas"]

    resumen = []
    for modelo in NOMBRES_MODELOS:
        fila = acumulados[modelo]
        for medida in ("precision_10", "recall_10", "map", "ndcg_10"):
            fila[medida] /= total
        resumen.append(fila)

    filas_tipos = []
    for fila in tipos.values():
        fila["map"] /= fila["consultas"]
        filas_tipos.append(fila)
    return {"cantidad_consultas": total, "resumen": resumen, "tipos": filas_tipos}


def elegir_modelos(nombres, seleccion_aplicada):
    if not nombres:
        if seleccion_aplicada:
            return []
        return list(NOMBRES_MODELOS)

    seleccionados = []
    for modelo in NOMBRES_MODELOS:
        if modelo in nombres:
            seleccionados.append(modelo)
    return seleccionados


def organizar_tipos(filas):
    tabla = {}
    for fila in filas:
        tipo = fila["tipo"]
        if tipo not in tabla:
            tabla[tipo] = {"tipo": tipo, "consultas": fila["consultas"], "modelos": {}}
        tabla[tipo]["modelos"][fila["modelo"]] = fila["map"]
    return list(tabla.values())


@app.get("/", response_class=HTMLResponse, name="inicio")
def inicio(
    request: Request,
    q: str = "",
    modelo: str = "BM25",
    modelos: list[str] | None = Query(default=None),
    n: int = Query(default=10),
    caso: str = "",
    split: str = "",
    vista: str = "buscar",
    seleccion_aplicada: bool = False,
):
    evaluacion_total = cargar_evaluacion()
    if vista not in ("buscar", "comparar", "evaluacion"):
        vista = "buscar"
    if vista == "evaluacion":
        if split not in (*evaluacion_total["splits"], "todos"):
            split = "todos"
    elif split not in evaluacion_total["splits"]:
        split = "test"
    if modelo not in NOMBRES_MODELOS:
        modelo = "BM25"

    caso_elegido = None
    consulta = ""
    if vista != "evaluacion":
        consulta, caso_elegido, split = elegir_consulta(evaluacion_total["splits"], split, caso, q)
    if split == "todos":
        evaluacion = resumir_todos(evaluacion_total["splits"])
    else:
        evaluacion = evaluacion_total["splits"][split]
    limite = leer_limite(n)
    seleccionados = elegir_modelos(modelos, seleccion_aplicada)
    error_modelos = vista == "comparar" and len(seleccionados) < 2

    metricas_caso = {}
    map_conjunto = {}
    for fila in evaluacion["resumen"]:
        map_conjunto[fila["modelo"]] = fila["map"]
    relevantes = set()
    if caso_elegido is not None:
        relevantes = set(caso_elegido["relevantes"])
        for fila in evaluacion["detalle"]:
            if fila["id"] == caso_elegido["id"]:
                metricas_caso[fila["modelo"]] = fila

    resultados = []
    comparacion = {}
    coincidencias = {}
    if consulta and vista != "evaluacion" and not error_modelos:
        sistema = cargar_sistema()
        if vista == "buscar":
            resultados = buscar(modelo, consulta, limite, sistema)
        else:
            for nombre in seleccionados:
                comparacion[nombre] = buscar(nombre, consulta, limite, sistema)
                for resultado in comparacion[nombre]:
                    identificador = resultado["id"]
                    coincidencias[identificador] = coincidencias.get(identificador, 0) + 1

    ruta_metadatos = CARPETA_PROYECTO / "datos" / "metadatos.json"
    metadatos = json.loads(ruta_metadatos.read_text(encoding="utf-8"))

    return templates.TemplateResponse(
        request=request,
        name="inicio.html",
        context={
            "vista": vista,
            "consulta": consulta,
            "caso_id": caso_elegido["id"] if caso_elegido else "",
            "sugerencias": elegir_sugerencias(evaluacion_total["splits"]),
            "nombre_modelo": modelo,
            "nombres_modelos": NOMBRES_MODELOS,
            "modelos_seleccionados": seleccionados,
            "error_modelos": error_modelos,
            "limite": limite,
            "resultados": resultados,
            "comparacion": comparacion,
            "coincidencias": coincidencias,
            "metricas_caso": metricas_caso,
            "map_conjunto": map_conjunto,
            "relevantes": relevantes,
            "evaluacion": evaluacion,
            "tipos_tabla": organizar_tipos(evaluacion["tipos"]),
            "split": split,
            "metadatos": metadatos,
        },
    )


@app.get("/salud")
def salud():
    return {"estado": "ok"}


@app.get("/sugerencias")
def sugerencias(q: str = ""):
    termino = q.strip().casefold()
    if len(termino) < 2:
        return elegir_sugerencias(cargar_evaluacion()["splits"])

    conjuntos = cargar_evaluacion()["splits"]
    comienzan = []
    contienen = []
    for split, datos in conjuntos.items():
        for caso in datos["consultas"]:
            texto = caso["texto"].strip()
            posicion = texto.casefold().find(termino)
            if posicion == 0:
                comienzan.append({"id": caso["id"], "split": split, "texto": texto})
            elif posicion > 0:
                contienen.append({"id": caso["id"], "split": split, "texto": texto})
    return (comienzan + contienen)[:8]
