# Corpus crudo ampliable

`news_raw_corpus.tsv` es la colección de adquisición de Fase I. Conserva una
fila por noticia y las columnas:

`id`, `title`, `description`, `source`, `category`, `date`, `url`,
`retrieved_at`.

Las categorías válidas son `Ciencia`, `Cultura` y `Negocios`. Las fuentes que
publican economía se asignan a `Negocios`, según la decisión del proyecto.

## Agregar noticias futuras

Desde la raíz del proyecto:

```powershell
python src\collect_corpus.py --target-total 750 --target-per-category 250
python src\prepare_corpus.py --max-per-category 250
python src\18_stage2_complete.py
```

El recolector conserva los registros existentes, consulta las fuentes RSS
configuradas, prioriza categorías que todavía no alcanzan su cuota, evita
duplicados por URL y título, y utiliza un UUID5 estable derivado de la URL.
`--target-per-category` limita la captura de Ciencia para que las consultas
posteriores obtengan principalmente Cultura y Negocios. Cambiar
`--target-total` solo solicita más registros; no borra noticias existentes.
`prepare_corpus.py` conserva como máximo la misma cantidad por categoría,
crea la versión normalizada, el texto concatenado título-descripción y las
particiones estratificadas.

El corpus crudo actual conserva 746 registros: Ciencia=324, Cultura=250 y
Negocios=172. El corpus procesado actual contiene 420 noticias balanceadas:
140 por categoría. Se redujo Ciencia al nivel disponible de Negocios para que
las tres clases tengan el mismo peso en clasificación, LDA y búsqueda.
El corpus crudo no se elimina ni se reduce, por lo que puede seguir creciendo.
