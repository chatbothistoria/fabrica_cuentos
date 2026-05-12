import streamlit as st
from supabase import create_client
from sentence_transformers import SentenceTransformer
from groq import Groq
import csv
import os

# --- 1. CONFIGURACIÓN DE LA PÁGINA ---
st.set_page_config(page_title="Buscador de Normativa Educativa", page_icon="📚")

# --- 2. CLAVES DE ACCESO ---
SUPABASE_URL = st.secrets["SUPABASE_URL"]
SUPABASE_KEY = st.secrets["SUPABASE_KEY"]
GROQ_API_KEY = st.secrets["GROQ_API_KEY"]

# --- 3. INICIALIZAR HERRAMIENTAS Y CARGAR ENLACES ---
@st.cache_resource
def init_supabase():
    return create_client(SUPABASE_URL, SUPABASE_KEY)

@st.cache_resource
def load_model():
    return SentenceTransformer("paraphrase-multilingual-mpnet-base-v2")

@st.cache_data
def cargar_diccionario_enlaces():
    """Lee el archivo enlaces.csv y crea un diccionario {nombre_archivo: url}"""
    enlaces = {}
    archivo_csv = "enlaces.csv"
    
    if os.path.exists(archivo_csv):
        # Leemos el archivo asegurando que entiende los acentos y caracteres especiales (utf-8)
        with open(archivo_csv, mode='r', encoding='utf-8') as f:
            lector = csv.reader(f)
            for i, fila in enumerate(lector):
                # Saltamos la primera fila porque son los títulos (nombre_archivo, url_oficial_verificada)
                if i == 0:
                    continue
                if len(fila) >= 2:
                    nombre = fila[0].strip()
                    url = fila[1].strip()
                    enlaces[nombre] = url
    else:
        st.warning("⚠️ No se ha encontrado el archivo 'enlaces.csv' en la misma carpeta que app.py.")
        
    return enlaces

supabase = init_supabase()
model = load_model()
groq_client = Groq(api_key=GROQ_API_KEY)

# Cargamos los enlaces en memoria al arrancar la app
diccionario_enlaces = cargar_diccionario_enlaces()

# --- 4. INTERFAZ WEB ---
st.title("📚 Buscador Inteligente de Normativa Educativa")

# Menú con el orden solicitado
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
                # 1. Búsqueda en Supabase
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
                    
                    for res in resultados:
                        nombre_archivo = res['nombre_archivo']
                        pagina = res['pagina_num']
                        nombre_limpio = nombre_archivo.replace(".pdf", "").replace("_", " ")
                        
                        # Acumulamos el texto para que la IA lo lea
                        contexto_para_ia += f"DOCUMENTO: {nombre_limpio} | PÁGINA: {pagina}\nCONTENIDO: {res['contenido']}\n\n"
                        
                        # Buscamos el enlace exacto en tu archivo CSV
                        url_base = diccionario_enlaces.get(nombre_archivo, None)
                        
                        if url_base:
                            # Le añadimos el número de página al final de la URL oficial
                            url_directa = f"{url_base}#page={pagina}"
                            texto_fuente = f"[{nombre_limpio} (Pág. {pagina})]({url_directa})"
                        else:
                            texto_fuente = f"**{nombre_limpio}** (Pág. {pagina}) *(Enlace web no disponible)*"
                            
                        enlaces_fuentes.append(texto_fuente)
                    
                    # 2. Redacción con Groq
                    prompt_sistema = (
                        "Eres un experto asesor en normativa educativa. "
                        "Tu tarea es responder a la pregunta del usuario utilizando ÚNICAMENTE la información proporcionada en el contexto. "
                        "Si mencionas algo específico, indica en tu redacción el nombre del documento y la página tal y como aparece en el contexto. "
                        "Escribe de forma clara, estructurada, resumiendo los puntos clave de forma natural."
                    )
                    prompt_usuario = f"CONTEXTO:\n{contexto_para_ia}\n\nPREGUNTA: {pregunta}"

                    respuesta_ia = groq_client.chat.completions.create(
                        model="llama-3.1-8b-instant", 
                        messages=[
                            {"role": "system", "content": prompt_sistema},
                            {"role": "user", "content": prompt_usuario}
                        ],
                        temperature=0.2 
                    )

                    # 3. Mostrar la respuesta redactada
                    st.write("---")
                    st.markdown(respuesta_ia.choices[0].message.content)
                    
                    # 4. Sección de Fuentes Consultadas
                    st.markdown("### 📚 Fuentes consultadas:")
                    
                    # Eliminamos fuentes repetidas (por si un archivo sale dos veces en la misma página)
                    fuentes_unicas = list(dict.fromkeys(enlaces_fuentes))
                    
                    for fuente in fuentes_unicas:
                        st.markdown(f"- 📄 {fuente}")
                
                else:
                    st.warning("No he encontrado nada específico en este bloque con esas palabras. Prueba a reformular la pregunta.")

            except Exception as e:
                st.error(f"Error técnico al buscar: {e}")
