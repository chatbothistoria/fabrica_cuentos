import streamlit as st
from supabase import create_client
from sentence_transformers import SentenceTransformer
import google.generativeai as genai
import csv, os, json, textwrap, time, requests
import numpy as np
from fpdf import FPDF

# =============================================================================
# CONFIGURACIÓN CENTRAL
# =============================================================================
GEMINI_MODEL_PRINCIPAL = "gemini-2.0-flash"
GEMINI_MODEL_RAPIDO    = "gemini-2.0-flash"
MAX_TOKENS_RESPUESTA  = 2500
MAX_TOKENS_RAPIDO     = 380
MAX_CHARS_PREGUNTA    = 500
MATCH_THRESHOLD_ALTO  = 0.40
MATCH_THRESHOLD_BAJO  = 0.25
MATCH_COUNT           = 15
HISTORIAL_TURNOS      = 3
COLLECTION_NAME       = "normativa"

# =============================================================================
# CONFIGURACIÓN DE PÁGINA
# =============================================================================
st.set_page_config(page_title="Normativa Educativa CyL", page_icon="📚", layout="centered")

# =============================================================================
# SESSION STATE
# =============================================================================
_DEFAULTS = {
    "historial_completo": [], "ultima_pregunta": None,
    "ultima_respuesta": None, "ultimas_fuentes": [],
    "confirmar_borrar": False,
    "feedback_pendiente": False, "feedback_pregunta": None,
    "feedback_respuesta": None, "pregunta_actual": "",
}
for _k, _v in _DEFAULTS.items():
    if _k not in st.session_state:
        st.session_state[_k] = _v

# =============================================================================
# PDF
# =============================================================================
_UNICODE_FIX = {
    "\u2018":"'", "\u2019":"'", "\u201C":'"', "\u201D":'"',
    "\u2013":"-", "\u2014":"-", "\u2022":"-", "\u00B7":"-",
    "\u2026":"...", "\u00A0":" ", "\u00AD":"-",
}

def _limpiar(texto):
    for orig, repl in _UNICODE_FIX.items():
        texto = texto.replace(orig, repl)
    return texto.encode("latin-1", "replace").decode("latin-1")

def generar_pdf(lista_interacciones, titulo="Normativa Educativa"):
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 16)
    pdf.cell(0, 10, _limpiar(titulo), ln=True, align="C")
    pdf.ln(8)
    for item in lista_interacciones:
        pdf.set_font("Helvetica", "B", 12)
        for linea in textwrap.wrap(f"PREGUNTA: {_limpiar(item['pregunta'])}", 80):
            pdf.cell(0, 6, linea, ln=True)
        corr = item.get("pregunta_corregida", "")
        if corr and corr.strip().lower() != item["pregunta"].strip().lower():
            pdf.set_font("Helvetica", "I", 10)
            pdf.cell(0, 5, _limpiar(f"(Corregida a: {corr})"), ln=True)
        pdf.ln(2)
        pdf.set_font("Helvetica", size=11)
        for linea in textwrap.wrap(_limpiar(item["respuesta"]), 90):
            pdf.cell(0, 6, linea, ln=True)
        pdf.ln(4)
        pdf.set_font("Helvetica", "I", 10)
        pdf.cell(0, 5, "FUENTES CONSULTADAS:", ln=True)
        for fuente in item.get("fuentes", []):
            for linea in textwrap.wrap(f"- {_limpiar(fuente)}", 90):
                pdf.cell(0, 5, linea, ln=True)
        pdf.ln(8)
    return bytes(pdf.output())

# =============================================================================
# CLAVES Y SERVICIOS
# =============================================================================
SUPABASE_URL  = st.secrets["SUPABASE_URL"]
SUPABASE_KEY  = st.secrets["SUPABASE_KEY"]
GOOGLE_API_KEY = st.secrets["GOOGLE_API_KEY"]
QDRANT_URL    = st.secrets["QDRANT_URL"]
QDRANT_API_KEY = st.secrets["QDRANT_API_KEY"]

@st.cache_resource
def init_supabase():
    return create_client(SUPABASE_URL, SUPABASE_KEY)

@st.cache_resource
def load_model():
    # ⚠️ NO cambiar sin re-vectorizar los documentos en Qdrant.
    return SentenceTransformer("intfloat/multilingual-e5-base")

@st.cache_data
def cargar_enlaces():
    """Carga el diccionario nombre_archivo → URL desde enlaces.csv.
    Usa utf-8-sig para manejar el BOM que tiene el fichero.
    Busca el fichero en varias ubicaciones posibles.
    """
    enlaces = {}
    rutas_posibles = ["enlaces.csv", "normativa_educativa/enlaces.csv", "/app/enlaces.csv"]
    ruta_encontrada = None
    for ruta in rutas_posibles:
        if os.path.exists(ruta):
            ruta_encontrada = ruta
            break
    if ruta_encontrada is None:
        return enlaces
    try:
        with open(ruta_encontrada, encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            for fila in reader:
                nombre = (fila.get("nombre_archivo") or "").strip()
                url    = (fila.get("url_oficial_verificada") or "").strip()
                if nombre and url:
                    enlaces[nombre] = url
    except Exception:
        # Fallback: lectura posicional
        try:
            with open(ruta_encontrada, encoding="utf-8-sig") as f:
                for i, fila in enumerate(csv.reader(f)):
                    if i == 0:
                        continue
                    if len(fila) >= 2 and fila[0].strip() and fila[1].strip():
                        enlaces[fila[0].strip()] = fila[1].strip()
        except Exception:
            pass
    return enlaces

supabase     = init_supabase()
model        = load_model()
genai.configure(api_key=GOOGLE_API_KEY)
enlaces      = cargar_enlaces()
if not enlaces:
    st.sidebar.warning("⚠️ enlaces.csv no encontrado — las fuentes no tendrán enlace.")

# =============================================================================
# FUNCIONES AUXILIARES
# =============================================================================

def _parse_json(text, default):
    text = text.strip()
    if "```" in text:
        partes = text.split("```")
        text = partes[1] if len(partes) > 1 else partes[0]
        if text.startswith("json"):
            text = text[4:]
    try:
        return json.loads(text.strip())
    except Exception:
        return default

def validar_input(pregunta):
    if not pregunta or not pregunta.strip():
        return False, "La pregunta no puede estar vacía."
    if len(pregunta) > MAX_CHARS_PREGUNTA:
        return False, f"Pregunta demasiado larga (máximo {MAX_CHARS_PREGUNTA} caracteres)."
    patrones = ["ignore previous", "ignora las instrucciones", "system:", "</s>", "[inst]", "###"]
    if any(p in pregunta.lower() for p in patrones):
        return False, "La pregunta contiene contenido no válido."
    return True, ""

def expandir_y_corregir(pregunta):
    try:
        _m = genai.GenerativeModel(GEMINI_MODEL_RAPIDO)
        resp = _m.generate_content(
            "Eres un experto en normativa educativa y derecho administrativo español.\n"
            "Dado el siguiente texto de un docente o familiar:\n"
            "  1. Corrige errores ortográficos\n"
            "  2. Genera 3 reformulaciones MUY DISTINTAS para mejorar la búsqueda en un RAG jurídico:\n"
            "     - opcion1: reformulación con terminología jurídica exacta del BOE/BOCYL\n"
            "       (usa: Artículo, párrafo, apartado, días hábiles, consanguinidad, etc.)\n"
            "     - opcion2: reformulación desde el punto de vista del funcionario docente\n"
            "       (usa vocabulario administrativo: solicitar, conceder, autorizar, derecho)\n"
            "     - opcion3: reformulación que mencione el tipo de norma relevante\n"
            "       (EBEP, LOE, LOMLOE, Decreto, Orden EDU, Resolución, etc.)\n\n"
            "Responde ÚNICAMENTE con JSON válido:\n"
            '{"corregida": "texto corregido", "reformulaciones": ["boe_bocyl", "funcionario", "norma"]}\n\n'
            f"Texto: {pregunta}",
            generation_config=genai.GenerationConfig(temperature=0.2, max_output_tokens=MAX_TOKENS_RAPIDO)
        )
        data = _parse_json(resp.text,
                           {"corregida": pregunta, "reformulaciones": []})
        corregida = data.get("corregida") or pregunta
        reformulaciones = [r for r in (data.get("reformulaciones") or []) if r]
        return corregida, reformulaciones
    except Exception:
        return pregunta, []


# Stopwords españolas para extracción de términos clave
_STOPWORDS = {
    "qué","cuál","cuáles","cómo","cuándo","cuánto","cuántos","cuántas",
    "dónde","quién","quiénes","por","para","con","sin","sobre","entre",
    "desde","hasta","hacia","ante","bajo","según","durante","mediante",
    "un","una","unos","unas","el","la","los","las","del","al",
    "es","son","está","están","ser","tener","tiene","tienen","haber",
    "hay","puede","pueden","debe","deben","se","me","te","le","nos",
    "de","en","a","y","o","e","u","que","si","no","más","pero",
    "yo","tú","él","ella","usted","nosotros","ellos","su","sus",
    "mi","mis","tu","tus","un","una","lo","le","les",
    "docente","docentes","alumno","alumna","alumnos","alumnas",
    "derecho","derechos","tiene","tendrá","podrá","podrán",
    "favor","hacer","realizar","solicitar","pedir",
}

def extraer_terminos_clave(pregunta: str) -> list[str]:
    """Extrae 4-6 términos clave de la pregunta eliminando stopwords.
    Devuelve también bigramas de términos legales relevantes.
    """
    import re
    # Normalizar
    texto = pregunta.lower().strip("¿?.,;:")
    texto = re.sub(r"[¿?.,;:()\.\[\]{}!]", " ", texto)
    palabras = [p for p in texto.split() if len(p) > 2 and p not in _STOPWORDS]

    terminos = []
    # Añadir palabras individuales relevantes
    for p in palabras:
        if p not in terminos:
            terminos.append(p)

    # Añadir bigramas de palabras consecutivas
    for i in range(len(palabras) - 1):
        bigrama = f"{palabras[i]} {palabras[i+1]}"
        if bigrama not in terminos:
            terminos.append(bigrama)

    return terminos[:8]  # máx 8 términos/bigramas

def _qdrant_search_rest(embedding, bloque, threshold=None):
    """Búsqueda semántica via REST API directa.
    Si bloque=None o bloque="general" busca en toda la colección sin filtro.
    """
    url = f"{QDRANT_URL}/collections/{COLLECTION_NAME}/points/search"
    headers = {"api-key": QDRANT_API_KEY, "Content-Type": "application/json"}
    if hasattr(embedding, 'tolist'):
        embedding = embedding.tolist()
    embedding = [float(x) for x in embedding]
    payload = {
        "vector": embedding,
        "limit": MATCH_COUNT,
        "with_payload": True,
    }
    # bloque="general" o None → sin filtro (busca en toda la colección)
    if bloque is not None and bloque != "general":
        payload["filter"] = {"must": [{"key": "bloque", "match": {"value": bloque}}]}
    if threshold is not None:
        payload["score_threshold"] = threshold
    try:
        r = requests.post(url, json=payload, headers=headers, timeout=30)
        r.raise_for_status()
        return r.json().get("result", [])
    except Exception:
        return []

def _qdrant_text_search_rest(pregunta_texto, bloque, terminos=None):
    """Búsqueda textual via REST API directa.
    Busca por términos clave extraídos de la pregunta.
    Si bloque=None o bloque="general" busca en toda la colección.
    """
    try:
        url = f"{QDRANT_URL}/collections/{COLLECTION_NAME}/points/scroll"
        headers = {"api-key": QDRANT_API_KEY, "Content-Type": "application/json"}
        # Usar términos clave si están disponibles, si no usar pregunta completa
        lista_terminos = terminos if terminos else [pregunta_texto]
        todos_resultados = []
        ids_vistos = set()

        for termino in lista_terminos[:5]:  # máx 5 búsquedas
            condiciones = [{"key": "contenido", "match": {"text": termino}}]
            if bloque is not None and bloque != "general":
                condiciones.append({"key": "bloque", "match": {"value": bloque}})
            payload = {
                "filter": {"must": condiciones},
                "limit": 4,
                "with_payload": True,
                "with_vector": False,
            }
            r = requests.post(url, json=payload, headers=headers, timeout=30)
            r.raise_for_status()
            puntos = r.json().get("result", {}).get("points", [])
            for p in puntos:
                pid = str(p.get("id", ""))
                if pid not in ids_vistos:
                    ids_vistos.add(pid)
                    todos_resultados.append(p)

        return todos_resultados[:8]
    except Exception:
        return []

def buscar_normativa_hibrida(embedding, pregunta_texto, bloque):
    """Búsqueda híbrida: semántica + textual via REST API de Qdrant.

    Estrategia de búsqueda:
    - bloque="general": busca en toda la colección sin filtro (EBEP, permisos, bajas...)
    - otros bloques: busca con filtro + fallback sin filtro si los resultados son pobres
    """
    # Extraer términos clave para búsqueda keyword
    terminos_clave = extraer_terminos_clave(pregunta_texto)

    # Para nivel GENERAL: buscar en toda la colección sin filtro
    if bloque == "general":
        resultados_v = []
        for threshold in [MATCH_THRESHOLD_ALTO, MATCH_THRESHOLD_BAJO, None]:
            hits = _qdrant_search_rest(embedding, None, threshold)
            resultados_v = hits
            if resultados_v:
                break
        resultados_t = _qdrant_text_search_rest(pregunta_texto, None, terminos_clave)

    else:
        # Niveles 1-3: con filtro de bloque
        resultados_v = []
        for threshold in [MATCH_THRESHOLD_ALTO, MATCH_THRESHOLD_BAJO, None]:
            hits = _qdrant_search_rest(embedding, bloque, threshold)
            resultados_v = hits
            if resultados_v:
                break

        # Nivel 4: sin filtro si los resultados son pocos o con score bajo
        score_max = max((h.get("score", 0) for h in resultados_v), default=0)
        if len(resultados_v) < 3 or score_max < 0.45:
            hits_sin_filtro = _qdrant_search_rest(embedding, None, None)
            ids_vistos_pre = {str(h.get("id")) for h in resultados_v}
            for h in hits_sin_filtro:
                if str(h.get("id")) not in ids_vistos_pre:
                    resultados_v.append(h)
            resultados_v = sorted(resultados_v, key=lambda x: x.get("score", 0), reverse=True)

        # Búsqueda keyword con términos extraídos
        resultados_t = _qdrant_text_search_rest(pregunta_texto, bloque, terminos_clave)

    # ── Boost: identificar fragmentos que aparecen en AMBAS búsquedas ────────
    # Estos son casi con certeza los artículos correctos — subirlos al top
    ids_semanticos = {str(h.get("id", "")) for h in resultados_v}
    ids_keyword    = {str(r.get("id", "")) for r in resultados_t}
    ids_en_ambas   = ids_semanticos & ids_keyword

    # Fusionar y deduplicar con boost para resultados cruzados
    vistos_contenido = set()
    ids_vistos = set()
    boost = []      # encontrados en ambas búsquedas → van primero
    resto = []      # solo en una búsqueda

    # Primero procesar semánticos
    for hit in resultados_v:
        rid = str(hit.get("id", ""))
        payload = hit.get("payload", {})
        contenido = payload.get("contenido", "")
        clave = contenido[:120].strip()
        if rid not in ids_vistos and clave not in vistos_contenido:
            ids_vistos.add(rid)
            if clave: vistos_contenido.add(clave)
            item = {
                "id":             rid,
                "contenido":      contenido,
                "nombre_archivo": payload.get("nombre_archivo", ""),
                "pagina_num":     payload.get("pagina_num", 0),
                "bloque":         payload.get("bloque", ""),
                "similarity":     hit.get("score", 0.0),
            }
            if rid in ids_en_ambas:
                boost.append(item)
            else:
                resto.append(item)

    # Luego añadir los keyword-only
    for record in resultados_t:
        rid = str(record.get("id", ""))
        payload = record.get("payload", {})
        contenido = payload.get("contenido", "")
        clave = contenido[:120].strip()
        if rid not in ids_vistos and clave not in vistos_contenido:
            ids_vistos.add(rid)
            if clave: vistos_contenido.add(clave)
            item = {
                "id":             rid,
                "contenido":      contenido,
                "nombre_archivo": payload.get("nombre_archivo", ""),
                "pagina_num":     payload.get("pagina_num", 0),
                "bloque":         payload.get("bloque", ""),
                "similarity":     0.95,  # score alto para keyword matches
            }
            # keyword-only siempre van al boost (contienen las palabras exactas)
            boost.append(item)

    # boost primero, luego resto semántico
    combinados = boost + resto
    return combinados[:MATCH_COUNT]

def reranquear(pregunta, fragmentos):
    if len(fragmentos) <= 2:
        return fragmentos
    try:
        lista_txt = "\n".join([
            f"[{i+1}] {f.get('contenido','')[:250]}"
            for i, f in enumerate(fragmentos)
        ])
        _m = genai.GenerativeModel(GEMINI_MODEL_RAPIDO)
        resp = _m.generate_content(
            f'Pregunta: "{pregunta}"\n\n'
            "Puntúa del 1 al 5 la relevancia de cada fragmento.\n"
            f'Responde SOLO con JSON: {{"puntuaciones": [n, n, ...]}}\n\n'
            f"Fragmentos:\n{lista_txt}",
            generation_config=genai.GenerationConfig(temperature=0, max_output_tokens=80)
        )
        data = _parse_json(resp.text, {"puntuaciones": []})
        punts = data.get("puntuaciones", [])
        if len(punts) == len(fragmentos):
            pares = sorted(zip(fragmentos, punts), key=lambda x: x[1], reverse=True)
            return [f for f, _ in pares]
    except Exception:
        pass
    return fragmentos

def construir_contexto_xml(fragmentos, enlaces_dict):
    contexto_xml = ""
    links_screen = []
    fuentes_pdf  = []
    for i, res in enumerate(fragmentos, 1):
        nombre   = res.get("nombre_archivo", "")
        pagina   = res.get("pagina_num", "")
        score    = res.get("similarity", "")
        nombre_l = nombre.replace(".pdf", "").replace("_", " ")
        score_s  = f"{score:.2f}" if isinstance(score, float) else ""
        contexto_xml += (
            f'<fragmento id="{i}" documento="{nombre_l}" '
            f'pagina="{pagina}" relevancia="{score_s}">\n'
            f'{res.get("contenido", "")}\n</fragmento>\n\n'
        )
        url = enlaces_dict.get(nombre)
        if url:
            # #page=N abre el PDF directamente en la página indicada en la mayoría de navegadores
            link = f"{url}#page={pagina}"
            links_screen.append(f"[{nombre_l} — pág. {pagina}]({link})")
            fuentes_pdf.append(f"{nombre_l} (Pág. {pagina}) — {url}")
        else:
            links_screen.append(f"**{nombre_l}** — pág. {pagina} *(enlace no disponible)*")
            fuentes_pdf.append(f"{nombre_l} (Pág. {pagina})")
    return contexto_xml, links_screen, fuentes_pdf

def construir_mensajes(pregunta, contexto_xml):
    PROMPT_SISTEMA = """\
Eres un asesor jurídico experto en normativa educativa española \
(legislación estatal y de Castilla y León).

FUENTES DE INFORMACIÓN:
Dispones de dos fuentes:
1. FRAGMENTOS NORMATIVOS: los <fragmento> proporcionados con el contexto.
2. CONOCIMIENTO JURÍDICO PROPIO: tu formación en derecho educativo español.

REGLAS:
- Usa SIEMPRE los fragmentos como fuente principal.
- Si los fragmentos contienen la respuesta, cítala con documento y página exactos.
- Si los fragmentos son parciales o insuficientes, COMPLETA con tu conocimiento jurídico general pero indícalo claramente con: *(información general — verifica en la normativa oficial)*
- NUNCA inventes artículos concretos ni números específicos que no estén en los fragmentos.
- Cita el documento y la página de cada afirmación que extraigas de los fragmentos.

REGLAS DE FORMATO OBLIGATORIAS:
- Usa ## y ### para estructurar secciones.
- Cuando haya varios casos (distintos días según parentesco, distintos plazos...) usa SIEMPRE una tabla Markdown.
- Para listas de requisitos o pasos usa viñetas con guion (-).
- Lenguaje claro y accesible para docentes, sin jerga innecesaria.
- Respuestas completas y detalladas — nunca cortes por brevedad.

ESTRUCTURA OBLIGATORIA:

## Respuesta
[respuesta directa, clara y completa — mínimo 4-5 frases con todo el detalle relevante]

## Normativa aplicable
[tabla o lista con artículos, documentos y páginas — todos los casos relevantes]

## Qué debes hacer
[pasos concretos y prácticos para el docente, familia o equipo directivo]

---
EJEMPLO:

Pregunta: ¿Cuántos días de permiso tiene un docente por fallecimiento de familiar?

## Respuesta
Los docentes funcionarios tienen derecho a permiso retribuido por fallecimiento,
accidente o enfermedad grave de un familiar. La duración varía según el grado
de parentesco y si se requiere desplazamiento fuera de la localidad.
Este derecho está reconocido tanto en la normativa estatal (EBEP) como en los
acuerdos de función pública de Castilla y León.

## Normativa aplicable

| Parentesco | Sin desplazamiento | Con desplazamiento |
|---|---|---|
| 1er grado: cónyuge, hijos, padres | 3 días hábiles | 5 días hábiles |
| 2º grado: hermanos, abuelos, nietos, suegros | 2 días hábiles | 4 días hábiles |

Fuente: EBEP, RD Legislativo 5/2015, artículo 48.a) — pág. 14

## Qué debes hacer
- Comunica el permiso a la dirección del centro lo antes posible.
- Aporta el certificado de defunción o el parte médico al reincorporarte.
- Los días cuentan como **hábiles**: no se incluyen fines de semana ni festivos.
- Si hay desplazamiento, guarda los justificantes de viaje por si se requieren."""

    mensajes = [{"role": "system", "content": PROMPT_SISTEMA}]
    ultimos = st.session_state.historial_completo[-HISTORIAL_TURNOS:]
    for turno in ultimos:
        resp_prev = turno["respuesta"]
        if len(resp_prev) > 1200:
            resp_prev = resp_prev[:1200] + "..."
        mensajes.append({"role": "user",      "content": turno["pregunta"]})
        mensajes.append({"role": "assistant", "content": resp_prev})
    mensajes.append({
        "role": "user",
        "content": f"CONTEXTO NORMATIVO:\n{contexto_xml}\n\nPREGUNTA: {pregunta}",
    })
    return mensajes

def guardar_log(bloque, preg_orig, preg_corr, num_res, tiempo_ms, tiene_resp):
    try:
        supabase.table("consultas_log").insert({
            "bloque": bloque, "pregunta_original": preg_orig[:500],
            "pregunta_corregida": preg_corr[:500], "num_resultados": num_res,
            "tiempo_ms": int(tiempo_ms), "tiene_respuesta": tiene_resp,
        }).execute()
    except Exception:
        pass

def guardar_feedback(pregunta, respuesta, util):
    try:
        supabase.table("feedback").insert({
            "pregunta": pregunta[:500], "respuesta": respuesta[:2000], "util": util,
        }).execute()
    except Exception:
        pass

# =============================================================================
# INTERFAZ — BARRA LATERAL
# =============================================================================
with st.sidebar:
    st.markdown("### 📊 Sesión actual")
    if st.session_state.historial_completo:
        st.caption(f"Consultas realizadas: {len(st.session_state.historial_completo)}")

# =============================================================================
# INTERFAZ — CUERPO PRINCIPAL
# =============================================================================
st.title("📚 Buscador Inteligente de Normativa Educativa")

bloque_elegido = st.selectbox(
    "Nivel educativo:",
    ["ninguno", "general", "infantil_primaria", "secundaria_bachillerato", "fp"],
    format_func=lambda x: {
        "ninguno":                 "— Selecciona un nivel educativo —",
        "general":                 "📋 General (permisos, bajas, vacaciones, EBEP...)",
        "infantil_primaria":       "🧒 Infantil y Primaria",
        "secundaria_bachillerato": "🎓 Secundaria y Bachillerato",
        "fp":                      "🔧 Formación Profesional",
    }[x],
)


with st.form(key="form_busqueda"):
    pregunta_input = st.text_area(
        "Haz tu pregunta sobre la normativa:",
        value=st.session_state.get("pregunta_actual", ""),
        height=100, max_chars=MAX_CHARS_PREGUNTA,
        placeholder="Escribe tu consulta sobre normativa educativa...",
    )
    submit = st.form_submit_button("🔍 Buscar", use_container_width=True)

# =============================================================================
# PROCESAMIENTO
# =============================================================================
if submit and pregunta_input:

    if bloque_elegido == "ninguno":
        st.warning("⚠️ Selecciona un nivel educativo antes de buscar.")

    else:
        valido, msg_error = validar_input(pregunta_input)
        if not valido:
            st.warning(f"⚠️ {msg_error}")
        else:
            try:
                t0 = time.time()

                with st.spinner("✏️ Analizando la consulta..."):
                    pregunta_corregida, reformulaciones = expandir_y_corregir(pregunta_input)

                if pregunta_corregida.strip().lower() != pregunta_input.strip().lower():
                    st.info(f"✏️ He corregido tu consulta a: **{pregunta_corregida}**")

                with st.spinner("🔎 Buscando en la normativa..."):
                    # e5 requiere prefijo "query: " en las consultas
                    todas = [pregunta_corregida] + reformulaciones[:2]
                    embedding_avg = np.mean(
                        [model.encode('query: ' + q, normalize_embeddings=True)
                         for q in todas], axis=0
                    ).tolist()
                    resultados = buscar_normativa_hibrida(
                        embedding_avg, pregunta_corregida, bloque_elegido
                    )

                if not resultados:
                    st.warning("No encontré normativa relacionada. Prueba a reformular la pregunta.")
                    guardar_log(bloque_elegido, pregunta_input, pregunta_corregida,
                                0, (time.time()-t0)*1000, False)
                else:
                    with st.spinner("📊 Ordenando por relevancia..."):
                        resultados = reranquear(pregunta_corregida, resultados)
                        resultados = resultados[:MATCH_COUNT]

                    contexto_xml, links_screen, fuentes_pdf = construir_contexto_xml(
                        resultados, enlaces
                    )
                    mensajes = construir_mensajes(pregunta_corregida, contexto_xml)

                    st.write("---")
                    st.markdown("### 📝 Respuesta:")

                    # Gemini no usa el formato messages de OpenAI
                    # Convertimos mensajes a texto plano para Gemini
                    prompt_completo = "\n\n".join([
                        m["content"] for m in mensajes if m.get("content")
                    ])
                    _m = genai.GenerativeModel(
                        GEMINI_MODEL_PRINCIPAL,
                        system_instruction=mensajes[0]["content"] if mensajes and mensajes[0]["role"] == "system" else None
                    )
                    _historial = [
                        {"role": "user" if m["role"] == "user" else "model",
                         "parts": [m["content"]]}
                        for m in mensajes
                        if m.get("content") and m["role"] in ("user", "assistant")
                    ]
                    if not _historial:
                        _historial = [{"role": "user", "parts": [prompt_completo]}]

                    stream = _m.generate_content(
                        _historial,
                        generation_config=genai.GenerationConfig(
                            temperature=0.1,
                            max_output_tokens=MAX_TOKENS_RESPUESTA,
                        ),
                        stream=True,
                    )

                    def _gen():
                        for chunk in stream:
                            if chunk.text:
                                yield chunk.text

                    texto_final = st.write_stream(_gen())

                    fuentes_u  = list(dict.fromkeys(links_screen))
                    fuentes_up = list(dict.fromkeys(fuentes_pdf))
                    st.markdown("### 📚 Fuentes consultadas:")
                    for f in fuentes_u:
                        st.markdown(f"- 📄 {f}", unsafe_allow_html=False)

                    st.session_state.ultima_pregunta   = pregunta_input
                    st.session_state.pregunta_actual   = pregunta_input
                    st.session_state.ultima_respuesta  = texto_final
                    st.session_state.ultimas_fuentes   = fuentes_u
                    st.session_state.historial_completo.append({
                        "pregunta":           pregunta_input,
                        "pregunta_corregida": pregunta_corregida,
                        "respuesta":          texto_final,
                        "fuentes":            fuentes_up,
                    })
                    if len(st.session_state.historial_completo) > 20:
                        st.session_state.historial_completo = \
                            st.session_state.historial_completo[-20:]

                    st.session_state.feedback_pendiente = True
                    st.session_state.feedback_pregunta  = pregunta_input
                    st.session_state.feedback_respuesta = texto_final

                    guardar_log(bloque_elegido, pregunta_input, pregunta_corregida,
                                len(resultados), (time.time()-t0)*1000, True)

            except Exception as e:
                err = str(e).lower()
                if "429" in err or "rate_limit" in err:
                    st.error("⏳ Límite de la API de Google alcanzado. Inténtalo en unos minutos.")
                else:
                    st.error(f"Error técnico: {e}")

elif st.session_state.ultima_respuesta:
    st.write("---")
    st.markdown(st.session_state.ultima_respuesta)
    st.markdown("### 📚 Fuentes consultadas:")
    for f in st.session_state.ultimas_fuentes:
        st.markdown(f"- 📄 {f}", unsafe_allow_html=False)

# =============================================================================
# FEEDBACK
# =============================================================================
if st.session_state.feedback_pendiente:
    st.markdown("---")
    st.markdown("**¿Te ha resultado útil esta respuesta?**")
    c1, c2, c3 = st.columns([1, 1, 5])
    with c1:
        if st.button("👍 Sí"):
            guardar_feedback(st.session_state.feedback_pregunta,
                             st.session_state.feedback_respuesta, True)
            st.session_state.feedback_pendiente = False
            st.success("¡Gracias!")
            st.rerun()
    with c2:
        if st.button("👎 No"):
            guardar_feedback(st.session_state.feedback_pregunta,
                             st.session_state.feedback_respuesta, False)
            st.session_state.feedback_pendiente = False
            st.info("Lo tendremos en cuenta.")
            st.rerun()

# =============================================================================
# HISTORIAL EN PANTALLA
# =============================================================================
historial = st.session_state.historial_completo
if len(historial) > 1:
    st.write("---")
    with st.expander(f"📋 Historial ({len(historial)} consultas)", expanded=False):
        for item in reversed(historial[:-1]):
            st.markdown(f"**Pregunta:** {item['pregunta']}")
            prev = item["respuesta"]
            st.markdown(prev[:400] + "..." if len(prev) > 400 else prev)
            st.divider()

# =============================================================================
# BOTONES DE ACCIÓN
# =============================================================================
if historial:
    st.write("---")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.download_button("📄 Descargar esta consulta",
            data=generar_pdf([historial[-1]], "Consulta de Normativa Educativa"),
            file_name="consulta_normativa.pdf", mime="application/pdf",
            use_container_width=True)
    with c2:
        st.download_button("📚 Descargar historial",
            data=generar_pdf(historial, "Historial Completo"),
            file_name="historial_normativa.pdf", mime="application/pdf",
            use_container_width=True)
    with c3:
        if not st.session_state.confirmar_borrar:
            if st.button("🔄 Reiniciar chat", use_container_width=True):
                st.session_state.confirmar_borrar = True
                st.rerun()
        else:
            st.warning("⚠️ ¿Seguro? Se borrará todo el historial.")
            ca, cb = st.columns(2)
            with ca:
                if st.button("✅ Sí, borrar", use_container_width=True):
                    for k, v in _DEFAULTS.items():
                        st.session_state[k] = ([] if isinstance(v, list) else v)
                    st.rerun()
            with cb:
                if st.button("❌ Cancelar", use_container_width=True):
                    st.session_state.confirmar_borrar = False
                    st.rerun()
