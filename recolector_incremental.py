# Recolector incremental de noticias RSS.
# Conserva noticias existentes, elimina duplicados por URL y agrega solo novedades.

import csv
import os
import shutil
import tempfile
import uuid
from datetime import datetime
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit

import feedparser
import pandas as pd

FUENTES_RSS = [
    {"url": "https://www.jornada.com.mx/rss/cultura.xml?v=1", "source": "La Jornada", "category": "Cultura"},
    {"url": "https://www.jornada.com.mx/rss/ciencias.xml?v=1", "source": "La Jornada", "category": "Ciencias"},
    {"url": "https://www.jornada.com.mx/rss/economia.xml?v=1", "source": "La Jornada", "category": "Economía"},
    {"url": "https://www.reforma.com/rss/ciencia.xml", "source": "Reforma", "category": "Ciencia"},
    {"url": "https://www.reforma.com/rss/cultura.xml", "source": "Reforma", "category": "Cultura"},
    {"url": "https://www.reforma.com/rss/negocios.xml", "source": "Reforma", "category": "Negocios"},
    {"url": "https://www.excelsior.com.mx/rss/cultura.xml", "source": "Excélsior", "category": "Cultura"},
    {"url": "https://www.elfinanciero.com.mx/rss/ciencia.xml", "source": "El Financiero", "category": "Ciencia"},
    {"url": "https://www.elfinanciero.com.mx/rss/cultura.xml", "source": "El Financiero", "category": "Cultura"},
    {"url": "https://www.elfinanciero.com.mx/rss/economia.xml", "source": "El Financiero", "category": "Economía"},
    {"url": "https://www.elfinanciero.com.mx/rss/negocios.xml", "source": "El Financiero", "category": "Negocios"},
    {"url": "https://www.mural.com.mx/rss/ciencia.xml", "source": "Mural", "category": "Ciencia"},
    {"url": "https://www.mural.com.mx/rss/cultura.xml", "source": "Mural", "category": "Cultura"},
    {"url": "https://www.mural.com.mx/rss/economia.xml", "source": "Mural", "category": "Economía"},
    {"url": "https://www.mural.com.mx/rss/negocios.xml", "source": "Mural", "category": "Negocios"},
]

ARCHIVO = Path("news_raw_corpus.tsv")
COLUMNAS = ["id", "title", "description", "source", "category", "date", "url", "retrieved_at"]


def limpiar_campo(valor):
    """Evita que saltos de línea o tabuladores rompan la estructura TSV."""
    if valor is None:
        return ""
    return " ".join(str(valor).replace("\t", " ").replace("\r", " ").replace("\n", " ").split())


def clave_url(url):
    """Normaliza una URL para comparar duplicados sin modificar la URL guardada."""
    url = (url or "").strip()
    if not url or url.lower() == "no url":
        return ""
    try:
        partes = urlsplit(url)
        esquema = (partes.scheme or "https").lower()
        host = (partes.netloc or "").lower()
        ruta = partes.path.rstrip("/") or "/"
        # Se elimina el fragmento (#...), pero se conserva la query string.
        return urlunsplit((esquema, host, ruta, partes.query, ""))
    except ValueError:
        return url


def leer_corpus_existente():
    """Lee el corpus, descarta cabeceras repetidas y conserva la primera URL."""
    if not ARCHIVO.exists():
        return []

    df = pd.read_csv(
        ARCHIVO,
        sep="\t",
        dtype=str,
        keep_default_na=False,
        encoding="utf-8",
        on_bad_lines="warn"
    )

    # Garantiza todas las columnas esperadas aunque falte alguna en el archivo.
    for columna in COLUMNAS:
        if columna not in df.columns:
            df[columna] = ""

    df = df[COLUMNAS].copy()

    # El recolector anterior podía insertar la cabecera como si fuera una noticia.
    df = df[df["id"].str.strip().str.lower() != "id"].copy()

    filas = []
    urls_vistas = set()
    ids_vistos = set()
    duplicados_url = 0

    for registro in df.to_dict("records"):
        registro = {c: limpiar_campo(registro.get(c, "")) for c in COLUMNAS}
        url_id = clave_url(registro["url"])

        # Para filas con URL válida, conservar solo la primera aparición.
        if url_id and url_id in urls_vistas:
            duplicados_url += 1
            continue
        if url_id:
            urls_vistas.add(url_id)

        # Si falta el ID o está repetido, generar uno nuevo para evitar colisiones.
        if not registro["id"] or registro["id"] in ids_vistos:
            registro["id"] = str(uuid.uuid4())
        ids_vistos.add(registro["id"])
        filas.append(registro)

    print(f"Noticias existentes leídas: {len(df)}")
    print(f"Duplicados por URL eliminados del conjunto en memoria: {duplicados_url}")
    return filas


def recolectar_novedades(urls_existentes):
    nuevas = []
    urls_nuevas = set()
    errores = 0

    for fuente in FUENTES_RSS:
        print(f"Consultando: {fuente['source']} - {fuente['category']}...")
        feed = feedparser.parse(fuente["url"])

        if getattr(feed, "bozo", False):
            print(f"  Aviso: el feed presentó un problema de lectura: {fuente['url']}")
            errores += 1

        if not feed.entries:
            print("  Sin entradas disponibles en esta consulta.")
            continue

        for entry in feed.entries:
            url = limpiar_campo(entry.get("link", ""))
            url_id = clave_url(url)

            # Sin URL no podemos identificar con seguridad una noticia ya recolectada.
            if not url_id:
                print("  Entrada omitida porque no tiene URL.")
                continue

            if url_id in urls_existentes or url_id in urls_nuevas:
                continue

            titulo = limpiar_campo(entry.get("title", "Sin título"))
            descripcion = limpiar_campo(
                entry.get("description", entry.get("summary", "Sin contenido"))
            )
            fecha = limpiar_campo(
                entry.get("published", entry.get("updated", "Sin fecha"))
            )

            nuevas.append({
                "id": str(uuid.uuid4()),
                "title": titulo,
                "description": descripcion,
                "source": fuente["source"],
                "category": fuente["category"],
                "date": fecha,
                "url": url,
                "retrieved_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            })
            urls_nuevas.add(url_id)

    print(f"Feeds con aviso de lectura: {errores}")
    return nuevas


def guardar_atomico(filas):
    """Escribe un TSV temporal y luego lo sustituye para evitar archivos a medias."""
    carpeta = str(ARCHIVO.resolve().parent)
    fd, temporal = tempfile.mkstemp(prefix="corpus_tmp_", suffix=".tsv", dir=carpeta)
    os.close(fd)
    try:
        with open(temporal, "w", newline="", encoding="utf-8") as archivo:
            escritor = csv.DictWriter(
                archivo,
                fieldnames=COLUMNAS,
                delimiter="\t",
                extrasaction="ignore",
                lineterminator="\n"
            )
            escritor.writeheader()
            escritor.writerows(filas)
        os.replace(temporal, ARCHIVO)
    finally:
        if os.path.exists(temporal):
            os.remove(temporal)


def main():
    # Conserva una copia del corpus original antes de cualquier modificación.
    if ARCHIVO.exists():
        marca = datetime.now().strftime("%Y%m%d_%H%M%S")
        respaldo = ARCHIVO.with_name(f"{ARCHIVO.stem}_backup_{marca}{ARCHIVO.suffix}")
        shutil.copy2(ARCHIVO, respaldo)
        print(f"Copia de seguridad creada: {respaldo}")

    existentes = leer_corpus_existente()
    urls_existentes = {
        clave_url(fila["url"]) for fila in existentes if clave_url(fila["url"])
    }

    nuevas = recolectar_novedades(urls_existentes)
    corpus_final = existentes + nuevas
    guardar_atomico(corpus_final)

    print("\nRecolección finalizada.")
    print(f"Noticias existentes conservadas tras deduplicar: {len(existentes)}")
    print(f"Noticias nuevas agregadas: {len(nuevas)}")
    print(f"Total final de noticias: {len(corpus_final)}")
    print(f"Archivo actualizado: {ARCHIVO.resolve()}")


if __name__ == "__main__":
    main()
