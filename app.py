import streamlit as st
from supabase import create_client
import numpy as np
import os
import re
import io
from sentence_transformers import SentenceTransformer
from groq import Groq
from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet

# ==============================================================
# 1. CONFIGURACIÓN DE PÁGINA Y PARÁMETROS
# ==============================================================
st.set_page_config(page_title="Asistente Normativa Educativa CyL", page_icon="📚", layout="centered")

# Parámetros de búsqueda
MAX_CHUNKS_TO_LLM = 8       
MODEL_NAME = "paraphrase-multilingual-mpnet-base-v2" 

# Configuración de APIs (Sustituye con tus claves o usa st.secrets)
SUPABASE_URL = st.secrets["SUPABASE_URL"]
SUPABASE_KEY = st.secrets["SUPABASE_KEY"]
GROQ_API_KEY = st.secrets["GROQ_API_KEY"]

# Inicialización de clientes
supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
client_groq = Groq(api_key=GROQ_API_KEY)

# Prompt del sistema para la IA
SYSTEM_PROMPT = """Eres un experto legal en normativa educativa de Castilla y León.
Tu objetivo es responder a las dudas de los usuarios basándote ÚNICAMENTE en el contexto proporcionado.

REGLAS:
1. Lee detenidamente el contexto proporcionado.
2. Si el contexto contiene la respuesta, redacta una respuesta clara, profesional y empática.
3. Menciona el nombre del documento y la página si están disponibles.
4. Si la información no está en el contexto, di amablemente que no dispones de esa información específica.
5. NO inventes normativas ni artículos.
"""

# ==============================================================
# 2. CARGA DE MODELO Y FUNCIONES DE APOYO
# ==============================================================
@st.cache_resource
def load_model():
    return SentenceTransformer(MODEL_NAME)

model = load_model()

def buscar_normativa(query_text, bloque_seleccionado):
    """Realiza la búsqueda vectorial en Supabase"""
    # 1. Convertir pregunta a vector
    query_vector = model.encode(query_text).tolist()
    
    # 2. Llamar a la función RPC de Supabase
    rpc_params = {
        'query_embedding': query_vector,
        'match_threshold': 0.4,
        'match_count': MAX_CHUNKS_TO_LLM,
        'filter_bloque': bloque_seleccionado
    }
    
    try:
        response = supabase.rpc('match_normativa_educativa', rpc_params).execute()
        return response.data
    except Exception as e:
        st.error(f"Error en la base de datos: {e}")
        return []

def generar_pdf(texto_respuesta):
    """Crea un PDF descargable con la respuesta"""
    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4)
    styles = getSampleStyleSheet()
    flowables = [
        Paragraph("Consulta de Normativa Educativa CyL", styles['Title']),
        Spacer(1, 12),
        Paragraph(texto_respuesta.replace("\n", "<br/>"), styles['Normal'])
    ]
    doc.build(flowables)
    buf.seek(0)
    return buf

# ==============================================================
# 3. INTERFAZ DE USUARIO (STREAMLIT)
# ==============================================================
st.title("📚 Consultor de Normativa CyL")
st.markdown("Pregunta sobre decretos, órdenes o reglamentos educativos de Castilla y León.")

# Barra lateral para selección de bloque
with st.sidebar:
    st.header("Configuración")
    bloque = st.selectbox(
        "Selecciona el ámbito educativo:",
        options=["infantil_primaria", "secundaria_bachillerato", "fp"],
        format_func=lambda x: x.replace("_", " ").title()
    )
    if st.button("Limpiar Chat"):
        st.session_state.messages = []
        st.rerun()

# Inicializar historial de chat
if "messages" not in st.session_state:
    st.session_state.messages = []

# Mostrar mensajes anteriores
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Entrada del usuario
if prompt := st.chat_input("¿En qué puedo ayudarte?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Lógica de respuesta
    with st.chat_message("assistant"):
        with st.status("Buscando en la normativa...", expanded=False) as status:
            # 1. Búsqueda en Supabase
            resultados = buscar_normativa(prompt, bloque)
            
            # 2. Construir contexto y citas
            contexto_str = ""
            citas = []
            if resultados:
                for res in resultados:
                    contexto_str += f"Doc: {res['nombre_archivo']} (Pág. {res['pagina_num']})\nContenido: {res['contenido']}\n\n"
                    citas.append(f"📄 {res['nombre_archivo']} (Pág. {res['pagina_num']})")
                status.update(label="Normativa encontrada. Generando respuesta...", state="complete")
            else:
                status.update(label="No se encontró normativa específica.", state="complete")

        # 3. Llamada a Groq
        full_prompt = f"CONTEXTO:\n{contexto_str}\n\nPREGUNTA: {prompt}"
        
        messages_api = [{"role": "system", "content": SYSTEM_PROMPT}]
        # Añadimos historial reciente (últimos 4 mensajes)
        for m in st.session_state.messages[-4:]:
            messages_api.append(m)
        messages_api[-1] = {"role": "user", "content": full_prompt}

        respuesta_placeholder = st.empty()
        full_response = ""

        try:
            stream = client_groq.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=messages_api,
                temperature=0.2,
                stream=True
            )
            for chunk in stream:
                if chunk.choices[0].delta.content:
                    full_response += chunk.choices[0].delta.content
                    respuesta_placeholder.markdown(full_response + "▌")
            
            # Añadir fuentes al final si existen
            if citas and "no encuentro" not in full_response.lower():
                fuentes_md = "\n\n---\n**Fuentes consultadas:**\n" + "\n".join(list(set(citas))[:4])
                full_response += fuentes_md
            
            respuesta_placeholder.markdown(full_response)
            st.session_state.messages.append({"role": "assistant", "content": full_response})

            # Botón para descargar PDF
            pdf_file = generar_pdf(full_response)
            st.download_button(
                label="📥 Descargar esta respuesta en PDF",
                data=pdf_file,
                file_name="consulta_normativa_cyl.pdf",
                mime="application/pdf"
            )

        except Exception as e:
            st.error(f"Error al conectar con la IA: {e}")