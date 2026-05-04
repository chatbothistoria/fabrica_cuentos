# Mi Fábrica de Cuentos

**Mi Fábrica de Cuentos** es una app educativa en Streamlit para que alumnado de **3 a 12 años** cree cuentos personalizados con una estructura adaptada a su nivel educativo.

## Niveles incluidos

1. **Infantil · 3-5 años**  
   Personaje + lugar + problema + aventura + solución + final.  
   Enfoque: oralidad, imágenes, secuencias y cuentos muy breves.

2. **6-8 años**  
   Personaje + lugar + deseo + problema + ayudante + objeto especial + aventura + solución + final.  
   Enfoque: frases guiadas, conectores y revisión básica.

3. **8-10 años**  
   Género + protagonista + rasgo + lugar + situación inicial + deseo + problema + causa + ayudante + objeto/pista + aventuras + diálogo + solución + cambio + final.  
   Enfoque: párrafos, diálogos, emociones y descripciones.

4. **10-12 años**  
   Género + tema + protagonista + objetivo + conflicto externo + conflicto interno + antagonista + escenarios + detonante + obstáculos + decisión + giro + clímax + solución + aprendizaje + final.  
   Enfoque: tramas completas, conflictos, giros narrativos y revisión de estilo.

## Qué incluye esta primera versión

- App web en Streamlit.
- Cuatro niveles educativos diferenciados.
- Datos de cada nivel en archivos JSON.
- Motor narrativo en Python sin IA.
- Escritura guiada por secciones.
- Sincronización inmediata del nivel educativo entre el selector central y el panel lateral.
- Campos de escritura libre en todas las opciones narrativas para que el alumnado pueda crear sus propias elecciones.
- Banco de palabras y retos de mejora.
- Guía docente básica.
- Descarga del cuento en TXT, PDF y JSON.
- Sin base de datos: no guarda cuentos online.

## Estructura del proyecto

```text
mi-fabrica-de-cuentos/
├── app.py
├── requirements.txt
├── README.md
├── data/
│   ├── infantil.json
│   ├── nivel_6_8.json
│   ├── nivel_8_10.json
│   └── nivel_10_12.json
├── core/
│   ├── __init__.py
│   ├── story_engine.py
│   └── pdf_exporter.py
├── assets/
│   └── README.md
└── .streamlit/
    └── config.toml
```

## Cómo ejecutarla en local

1. Instala Python 3.10 o superior.
2. Crea un entorno virtual si lo deseas.
3. Instala dependencias:

```bash
pip install -r requirements.txt
```

4. Ejecuta la app:

```bash
streamlit run app.py
```

## Cómo subirla a Streamlit Community Cloud

1. Crea un repositorio en GitHub.
2. Sube todos los archivos de esta carpeta.
3. En Streamlit Community Cloud, crea una nueva app.
4. Selecciona el repositorio.
5. Indica como archivo principal:

```text
app.py
```

6. Despliega la app.

## Privacidad

Esta versión no crea usuarios ni guarda datos en una base de datos. Los cuentos solo existen durante la sesión del navegador y pueden descargarse. Para uso real con menores, evita introducir apellidos, fotos personales u otros datos identificativos.

## Próximos pasos posibles

- Añadir biblioteca de cuentos con base de datos.
- Añadir panel docente con clases y retos.
- Añadir modo familia.
- Añadir ilustraciones y portadas.
- Añadir grabación de voz para Infantil.
- Añadir IA como ayudante pedagógico para títulos, vocabulario, preguntas y revisión.
- Exportar libros colectivos de aula.

## Comprobación de prioridad del campo libre

En cada pieza narrativa, el alumno puede seleccionar una opción del desplegable y también escribir una opción propia. La regla es:

- Si el campo libre tiene texto, se usa el campo libre.
- Si el campo libre está vacío, se usa la opción del desplegable.

Para comprobarlo de forma automática:

```bash
python scripts/check_selection_priority.py
```

El script valida todos los niveles y todos los campos narrativos. Comprueba 404 casos con campo libre que sustituye al desplegable, 404 casos con campo libre vacío que usan el desplegable y una prueba de exportación a PDF por cada campo narrativo.
