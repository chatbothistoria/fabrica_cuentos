import streamlit as st
from supabase import create_client
from sentence_transformers import SentenceTransformer
from groq import Groq
import csv, os, json, textwrap, time, requests
import numpy as np
from fpdf import FPDF

# =============================================================================
# CONFIGURACIÓN CENTRAL
# =============================================================================
GROQ_MODEL_PRINCIPAL  = "llama-3.3-70b-versatile"
GROQ_MODEL_RAPIDO     = "llama-3.1-8b-instant"
MAX_TOKENS_RESPUESTA  = 1200
MAX_TOKENS_RAPIDO     = 380
MAX_CHARS_PREGUNTA    = 500
MATCH_THRESHOLD_ALTO  = 0.40
MATCH_THRESHOLD_BAJO  = 0.25
MATCH_COUNT           = 8
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
GROQ_API_KEY  = st.secrets["GROQ_API_KEY"]
QDRANT_URL    = st.secrets["QDRANT_URL"]
QDRANT_API_KEY = st.secrets["QDRANT_API_KEY"]

@st.cache_resource
def init_supabase():
    return create_client(SUPABASE_URL, SUPABASE_KEY)

@st.cache_resource
def load_model():
    # ⚠️ NO cambiar sin re-vectorizar los documentos en Qdrant.
    return SentenceTransformer("paraphrase-multilingual-mpnet-base-v2")

@st.cache_data
def cargar_enlaces():
    enlaces = {}
    if os.path.exists("enlaces.csv"):
        with open("enlaces.csv", encoding="utf-8") as f:
            for i, fila in enumerate(csv.reader(f)):
                if i == 0:
                    continue
                if len(fila) >= 2:
                    enlaces[fila[0].strip()] = fila[1].strip()
    return enlaces

supabase     = init_supabase()
model        = load_model()
groq_client  = Groq(api_key=GROQ_API_KEY)
enlaces      = cargar_enlaces()

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
        resp = groq_client.chat.completions.create(
            model=GROQ_MODEL_RAPIDO,
            messages=[{"role": "user", "content": (
                "Eres un asistente especializado en normativa educativa española.\n"
                "Dado el siguiente texto:\n"
                "  1. Corrige todos los errores ortográficos y tipográficos\n"
                "  2. Genera 3 reformulaciones usando terminología jurídica y educativa\n\n"
                "Responde ÚNICAMENTE con JSON válido, sin texto adicional:\n"
                '{"corregida": "texto corregido", "reformulaciones": ["opcion1", "opcion2", "opcion3"]}\n\n'
                f"Texto: {pregunta}"
            )}],
            temperature=0.2,
            max_tokens=MAX_TOKENS_RAPIDO,
        )
        data = _parse_json(resp.choices[0].message.content,
                           {"corregida": pregunta, "reformulaciones": []})
        corregida = data.get("corregida") or pregunta
        reformulaciones = [r for r in (data.get("reformulaciones") or []) if r]
        return corregida, reformulaciones
    except Exception:
        return pregunta, []

def _qdrant_search_rest(embedding, bloque, threshold=None):
    """Búsqueda semántica via REST API directa.
    Si bloque=None busca en toda la colección sin filtro (normativa general).
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
    # Solo añadir filtro si se especifica un bloque
    if bloque is not None:
        payload["filter"] = {"must": [{"key": "bloque", "match": {"value": bloque}}]}
    if threshold is not None:
        payload["score_threshold"] = threshold
    try:
        r = requests.post(url, json=payload, headers=headers, timeout=30)
        r.raise_for_status()
        return r.json().get("result", [])
    except Exception:
        return []

def _qdrant_text_search_rest(pregunta_texto, bloque):
    """Búsqueda textual via REST API directa.
    Si bloque=None busca en toda la colección.
    """
    try:
        url = f"{QDRANT_URL}/collections/{COLLECTION_NAME}/points/scroll"
        headers = {"api-key": QDRANT_API_KEY, "Content-Type": "application/json"}
        condiciones = [{"key": "contenido", "match": {"text": pregunta_texto}}]
        if bloque is not None:
            condiciones.append({"key": "bloque", "match": {"value": bloque}})
        payload = {
            "filter": {"must": condiciones},
            "limit": 3,
            "with_payload": True,
            "with_vector": False,
        }
        r = requests.post(url, json=payload, headers=headers, timeout=30)
        r.raise_for_status()
        return r.json().get("result", {}).get("points", [])
    except Exception:
        return []

def buscar_normativa_hibrida(embedding, pregunta_texto, bloque):
    """Búsqueda híbrida: semántica + textual via REST API de Qdrant.

    Estrategia de búsqueda:
    - bloque="general": busca en toda la colección sin filtro (EBEP, permisos, bajas...)
    - otros bloques: busca con filtro + fallback sin filtro si los resultados son pobres
    """
    # Para nivel GENERAL: buscar en toda la colección sin filtro
    if bloque == "general":
        resultados_v = []
        for threshold in [MATCH_THRESHOLD_ALTO, MATCH_THRESHOLD_BAJO, None]:
            hits = _qdrant_search_rest(embedding, None, threshold)
            resultados_v = hits
            if resultados_v:
                break
        resultados_t = _qdrant_text_search_rest(pregunta_texto, None)

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

        # Búsqueda textual
        resultados_t = _qdrant_text_search_rest(pregunta_texto, bloque)

    # Fusionar resultados (ambas fuentes devuelven dicts via REST)
    ids_vistos = set()
    combinados = []

    for hit in resultados_v:
        rid = str(hit.get("id", ""))
        if rid not in ids_vistos:
            ids_vistos.add(rid)
            payload = hit.get("payload", {})
            combinados.append({
                "id":             rid,
                "contenido":      payload.get("contenido", ""),
                "nombre_archivo": payload.get("nombre_archivo", ""),
                "pagina_num":     payload.get("pagina_num", 0),
                "bloque":         payload.get("bloque", ""),
                "similarity":     hit.get("score", 0.0),
            })

    for record in resultados_t:
        rid = str(record.get("id", ""))
        if rid not in ids_vistos:
            ids_vistos.add(rid)
            payload = record.get("payload", {})
            combinados.append({
                "id":             rid,
                "contenido":      payload.get("contenido", ""),
                "nombre_archivo": payload.get("nombre_archivo", ""),
                "pagina_num":     payload.get("pagina_num", 0),
                "bloque":         payload.get("bloque", ""),
                "similarity":     1.0,
            })

    return combinados[:MATCH_COUNT + 2]

def reranquear(pregunta, fragmentos):
    if len(fragmentos) <= 2:
        return fragmentos
    try:
        lista_txt = "\n".join([
            f"[{i+1}] {f.get('contenido','')[:250]}"
            for i, f in enumerate(fragmentos)
        ])
        resp = groq_client.chat.completions.create(
            model=GROQ_MODEL_RAPIDO,
            messages=[{"role": "user", "content": (
                f'Pregunta: "{pregunta}"\n\n'
                "Puntúa del 1 al 5 la relevancia de cada fragmento.\n"
                f'Responde SOLO con JSON: {{"puntuaciones": [n, n, ...]}}\n\n'
                f"Fragmentos:\n{lista_txt}"
            )}],
            temperature=0,
            max_tokens=80,
        )
        data = _parse_json(resp.choices[0].message.content, {"puntuaciones": []})
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

REGLAS ESTRICTAS:
- Responde SOLO con información de los <fragmento> proporcionados.
- NUNCA inventes ni cites normativas que no aparezcan en el contexto.
- Si la información es insuficiente, indica qué tipo de normativa regula esa materia para orientar al usuario, sin inventar artículos concretos.
- Cita siempre el documento y la página exacta.

REGLAS DE FORMATO OBLIGATORIAS:
- Usa ## y ### para estructurar secciones.
- Cuando haya varios casos o variantes (distintos días según parentesco, distintos plazos...) usa SIEMPRE una tabla Markdown con columnas claras.
- Para listas de requisitos o pasos usa viñetas con guion (-).
- Separa secciones con línea en blanco.
- Nunca escribas bloques de texto denso sin estructura.
- Lenguaje claro y accesible para docentes, sin jerga jurídica innecesaria.

ESTRUCTURA OBLIGATORIA:

## Respuesta
[respuesta directa y clara en 2-3 frases]

## Normativa aplicable
[tabla o lista estructurada con artículos, documentos y páginas]

## Qué debes hacer
[pasos concretos y prácticos]

---
EJEMPLO:

Pregunta: ¿Cuántos días de permiso tiene un docente por fallecimiento de familiar?

## Respuesta
Los docentes tienen derecho a permiso retribuido por fallecimiento de familiar.
La duración depende del grado de parentesco y de si hay desplazamiento.

## Normativa aplicable

| Parentesco | Sin desplazamiento | Con desplazamiento |
|---|---|---|
| 1er grado (cónyuge, hijos, padres) | 3 días hábiles | 5 días hábiles |
| 2º grado (hermanos, abuelos, nietos) | 2 días hábiles | 4 días hábiles |

Fuente: EBEP, RD Legislativo 5/2015, artículo 48.a) — pág. 14

## Qué debes hacer
- Comunica el permiso a dirección lo antes posible.
- Aporta el certificado de defunción al reincorporarte.
- Los días son **hábiles**: no cuentan fines de semana ni festivos."""

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
                    todas = [pregunta_corregida] + reformulaciones[:2]
                    embedding_avg = np.mean(
                        [model.encode(q) for q in todas], axis=0
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

                    stream = groq_client.chat.completions.create(
                        model=GROQ_MODEL_PRINCIPAL,
                        messages=mensajes,
                        temperature=0.1,
                        max_tokens=MAX_TOKENS_RESPUESTA,
                        stream=True,
                    )

                    def _gen():
                        for chunk in stream:
                            delta = chunk.choices[0].delta.content
                            if delta:
                                yield delta

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
                    st.error("⏳ Límite diario de Groq alcanzado. Inténtalo mañana.")
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
