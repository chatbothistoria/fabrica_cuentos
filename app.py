import streamlit as st
from supabase import create_client
from sentence_transformers import SentenceTransformer
from groq import Groq
import csv
import os
from fpdf import FPDF

# --- 1. CONFIGURACIÓN DE LA PÁGINA ---
st.set_page_config(page_title="Buscador de Normativa Educativa", page_icon="📚")

# --- MEMORIA SEPARADA (Coste 0€) ---
# 1. Memoria corta para la IA (Solo recuerda el último turno)
if 'ultima_pregunta' not in st.session_state:
    st.session_state.ultima_pregunta = None
if 'ultima_respuesta' not in st.session_state:
    st.session_state.ultima_respuesta = None

# 2. Memoria larga para los PDFs (Guarda todo el historial de la sesión actual)
if 'historial_completo' not in st.session_state:
    st.session_state.historial_completo = []

# --- FUNCIONES PARA CREAR PDFS GRATIS ---
def generar_pdf(lista_interacciones, titulo_documento="Normativa Educativa"):
    # Usamos FPDF para generar un PDF al vuelo
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    
    # Título
    pdf.set_font("Helvetica", style="B", size=16)
    pdf.cell(0, 10, titulo_documento, ln=True, align="C")
    pdf.ln(10)
    
    # Contenido
    for item in lista_interacciones:
        # Pregunta
        pdf.set_font("Helvetica", style="B", size=12)
        pdf.multi_cell(0, 8, txt=f"PREGUNTA: {item['pregunta']}")
        pdf.ln(2)
        
        # Respuesta (Cambiamos caracteres raros para que FPDF no falle con acentos)
        pdf.set_font("Helvetica", size=11)
        respuesta_limpia = item['respuesta'].encode('latin-1', 'replace').decode('latin-1')
        pdf.multi_cell(0, 6, txt=respuesta_limpia)
        pdf.ln(5)
        
        # Fuentes
        pdf.set_font("Helvetica", style="I", size=10)
        pdf.multi_cell(0, 6, txt="FUENTES CONSULTADAS:")
        for fuente in item['fuentes']:
            fuente_limpia = fuente.encode('latin-1', 'replace').decode('latin-1')
            pdf.multi_cell(0, 6, txt=f"- {fuente_limpia}")
        
        pdf.ln(10) # Espacio entre consultas
        
    return pdf.output() # Devuelve los bytes del PDF

# --- 2. CLAVES DE ACCESO ---
SUPABASE_URL = st.secrets["SUPABASE_URL"]
SUPABASE_KEY = st.secrets["SUPABASE_KEY"]
GROQ_API_KEY = st.secrets["GROQ_API_KEY"]

# --- 3. INICIALIZAR HERRAMIENTAS ---
@st.cache_resource
def init_supabase():
    return create_client(SUPABASE_URL, SUPABASE_KEY)

@st.cache_resource
def load_model():
    return SentenceTransformer("paraphrase-multilingual-mpnet-base-v2")

@st.cache_data
def cargar_diccionario_enlaces():
    enlaces = {}
    archivo_csv = "enlaces.csv"
    if os.path.exists(archivo_csv):
        with open(archivo_csv, mode='r', encoding='utf-8') as f:
            lector = csv.reader(f)
            for i, fila in enumerate(lector):
                if i == 0: continue
                if len(fila) >= 2:
                    nombre = fila[0].strip()
                    url = fila[1].strip()
                    enlaces[nombre] = url
    return enlaces

supabase = init_supabase()
model = load_model()
groq_client = Groq(api_key=GROQ_API_KEY)
diccionario_enlaces = cargar_diccionario_enlaces()

# --- 4. INTERFAZ WEB ---
st.title("📚 Buscador Inteligente de Normativa Educativa")

bloque_elegido = st.selectbox(
    "Nivel educativo:",
    ["ninguno", "infantil_primaria", "secundaria_bachillerato", "fp"],
    format_func=lambda x: {
        "ninguno": "Por favor, elige un nivel educativo",
        "infantil_primaria": "Infantil y Primaria",
        "secundaria_bachillerato": "Secundaria y Bachillerato",
        "fp": "Formación Profesional"
    }[x]
)

pregunta = st.text_input("Haz tu pregunta sobre la normativa:")

if st.button("Buscar") and pregunta:
    if bloque_elegido == "ninguno":
        st.warning("⚠️ Por favor, selecciona un nivel educativo en el menú desplegable antes de buscar.")
    else:
        with st.spinner("Buscando en las leyes y redactando la respuesta..."):
            try:
                # 1. Búsqueda
                embedding_pregunta = model.encode(pregunta).tolist()
                respuesta_bd = supabase.rpc(
                    "buscar_normativa", 
                    {
                        "query_embedding": embedding_pregunta, 
                        "filtro_bloque": bloque_elegido,
                        "match_threshold": 0.3, 
                        "match_count": 6 
                    }
                ).execute()

                resultados = respuesta_bd.data

                if resultados:
                    contexto_para_ia = ""
                    enlaces_fuentes = []
                    textos_fuentes_pdf = [] # Guardamos esto limpio para el PDF
                    
                    for res in resultados:
                        nombre_archivo = res['nombre_archivo']
                        pagina = res['pagina_num']
                        nombre_limpio = nombre_archivo.replace(".pdf", "").replace("_", " ")
                        
                        contexto_para_ia += f"DOCUMENTO: {nombre_limpio} | PÁGINA: {pagina}\nCONTENIDO: {res['contenido']}\n\n"
                        
                        url_base = diccionario_enlaces.get(nombre_archivo, None)
                        if url_base:
                            url_directa = f"{url_base}#page={pagina}"
                            texto_fuente = f"[{nombre_limpio} (Pág. {pagina})]({url_directa})"
                            textos_fuentes_pdf.append(f"{nombre_limpio} (Pág. {pagina}) - Enlace: {url_base}")
                        else:
                            texto_fuente = f"**{nombre_limpio}** (Pág. {pagina}) *(Enlace web no disponible)*"
                            textos_fuentes_pdf.append(f"{nombre_limpio} (Pág. {pagina})")
                            
                        enlaces_fuentes.append(texto_fuente)
                    
                    # 2. IA Redactora (Protegida contra inventos)
                    prompt_sistema = (
                        "Eres un experto asesor jurista especializado en normativa educativa. "
                        "REGLA DE ORO: Responde ÚNICAMENTE utilizando la información de los fragmentos proporcionados en el contexto. "
                        "Si el contexto no menciona la respuesta a lo que te preguntan, DEBES responder exactamente esto: "
                        "'No he encontrado información sobre esta cuestión específica en la normativa consultada.' "
                        "Bajo NINGÚN concepto asumas, inventes o deduzcas leyes. "
                        "Indica siempre el nombre del documento y la página en tu texto."
                    )
                    
                    # Memoria corta (solo la consulta anterior para Groq)
                    historial_texto = ""
                    if st.session_state.ultima_pregunta:
                        historial_texto = (
                            f"--- CONTEXTO ANTERIOR ---\n"
                            f"El usuario preguntó: {st.session_state.ultima_pregunta}\n"
                            f"Tú respondiste: {st.session_state.ultima_respuesta}\n"
                            f"-------------------------\n\n"
                        )

                    prompt_usuario = f"{historial_texto}CONTEXTO ACTUAL:\n{contexto_para_ia}\n\nPREGUNTA ACTUAL: {pregunta}"

                    respuesta_ia = groq_client.chat.completions.create(
                        model="llama-3.1-8b-instant", 
                        messages=[
                            {"role": "system", "content": prompt_sistema},
                            {"role": "user", "content": prompt_usuario}
                        ],
                        temperature=0.1 
                    )

                    texto_final = respuesta_ia.choices[0].message.content
                    fuentes_unicas = list(dict.fromkeys(enlaces_fuentes))
                    fuentes_unicas_pdf = list(dict.fromkeys(textos_fuentes_pdf))

                    # 3. GUARDAMOS EN LAS DOS MEMORIAS
                    # Memoria Corta (Groq)
                    st.session_state.ultima_pregunta = pregunta
                    st.session_state.ultima_respuesta = texto_final
                    
                    # Memoria Larga (Para los PDFs)
                    st.session_state.historial_completo.append({
                        "pregunta": pregunta,
                        "respuesta": texto_final,
                        "fuentes": fuentes_unicas_pdf
                    })

                    # 4. Mostrar en pantalla
                    st.write("---")
                    st.markdown(texto_final)
                    
                    st.markdown("### 📚 Fuentes consultadas:")
                    for fuente in fuentes_unicas:
                        st.markdown(f"- 📄 {fuente}")
                    
                    st.write("---")
                    
                    # --- BOTONES DE EXPORTACIÓN A PDF ---
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        # Genera PDF solo con esta consulta (el último elemento de la lista)
                        pdf_actual = generar_pdf([st.session_state.historial_completo[-1]], "Consulta de Normativa Educativa")
                        st.download_button(
                            label="📄 Descargar esta consulta (PDF)",
                            data=pdf_actual,
                            file_name="consulta_normativa.pdf",
                            mime="application/pdf"
                        )
                        
                    with col2:
                        # Genera PDF con todo el historial de la sesión
                        pdf_historial = generar_pdf(st.session_state.historial_completo, "Historial Completo de Consultas")
                        st.download_button(
                            label="📚 Descargar historial de chat (PDF)",
                            data=pdf_historial,
                            file_name="historial_normativa.pdf",
                            mime="application/pdf"
                        )
                
                else:
                    st.warning("No he encontrado nada específico en este bloque con esas palabras.")

            except Exception as e:
                st.error(f"Error técnico al buscar: {e}")
