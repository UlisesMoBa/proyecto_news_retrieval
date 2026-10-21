# Script para extraer noticias de múltiples fuentes RSS y guardarlas en un archivo TSV.
import feedparser
import csv
import uuid
from datetime import datetime

fuentes_rss = [
    {
        "url": "https://www.jornada.com.mx/rss/cultura.xml?v=1",
        "source": "La Jornada",
        "category": "Cultura"
    },
    {
        "url": "https://www.jornada.com.mx/rss/ciencias.xml?v=1",
        "source": "La Jornada",
        "category": "Ciencias"
    },
    {
        "url": "https://www.jornada.com.mx/rss/economia.xml?v=1",
        "source": "La Jornada",
        "category": "Economía"
    },
    {
        "url": "https://www.reforma.com/rss/ciencia.xml",
        "source": "Reforma",
        "category": "Ciencia"
    },
    {
        "url": "https://www.reforma.com/rss/cultura.xml",
        "source": "Reforma",
        "category": "Cultura"
    },
    {
        "url": "https://www.reforma.com/rss/negocios.xml",
        "source": "Reforma",
        "category": "Negocios"
    },
    {
        "url": "https://www.excelsior.com.mx/rss/cultura.xml",
        "source": "Excélsior",
        "category": "Cultura"
    },
    {
        "url": "https://www.elfinanciero.com.mx/rss/ciencia.xml",
        "source": "El Financiero",
        "category": "Ciencia"
    },
    {
        "url": "https://www.elfinanciero.com.mx/rss/cultura.xml",
        "source": "El Financiero",
        "category": "Cultura"
    },
    {
        "url": "https://www.elfinanciero.com.mx/rss/economia.xml",
        "source": "El Financiero",
        "category": "Economía"
    },
    {
        "url": "https://www.elfinanciero.com.mx/rss/negocios.xml",
        "source": "El Financiero",
        "category": "Negocios"
    },
    {
        "url": "https://www.mural.com.mx/rss/ciencia.xml",
        "source": "Mural",
        "category": "Ciencia"
    },
    {
        "url": "https://www.mural.com.mx/rss/cultura.xml",
        "source": "Mural",
        "category": "Cultura"
    },
    {
        "url": "https://www.mural.com.mx/rss/economia.xml",
        "source": "Mural",
        "category": "Economía"
    },
    {
        "url": "https://www.mural.com.mx/rss/negocios.xml",
        "source": "Mural",
        "category": "Negocios"
    }
]

nombre_archivo = "news_raw_corpus.tsv"

# 2. Abrir el archivo TSV una sola vez al principio
with open(nombre_archivo, mode="a", newline="", encoding="utf-8") as archivo_tsv:
    escritor_tsv = csv.writer(archivo_tsv, delimiter="\t")
    escritor_tsv.writerow(["id", "title", "description", "source", "category", "date", "url", "retrieved_at"])

    # 3. Ciclo exterior: Recorrer cada enlace de nuestra lista
    for fuente in fuentes_rss:
        print(f"Consultando: {fuente['source']} - {fuente['category']}...")
        
        # Parsear el XML de la URL actual
        feed = feedparser.parse(fuente['url'])
        
        # 4. Ciclo interior: Extraer las noticias de ese feed
        for entry in feed.entries:
            doc_id = str(uuid.uuid4())
            
            title = entry.get("title", "No title").replace("\n", " ").replace("\t", " ")
            content = entry.get("description", entry.get("summary", "No content")).replace("\n", " ").replace("\t", " ")
            
            date = entry.get("published", entry.get("updated", "No date"))
            url = entry.get("link", "No url")
            retrieved_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            
            # Escribir la fila usando los datos dinámicos del diccionario actual
            escritor_tsv.writerow([doc_id, title, content, fuente['source'], fuente['category'], date, url, retrieved_at])

print(f"\nExtracción finalizada. Los datos crudos se guardaron en: {nombre_archivo}")