import streamlit as st
from supabase import create_client
from sentence_transformers import SentenceTransformer
from groq import Groq

# --- 1. CONFIGURACIÓN DE LA PÁGINA ---
st.set_page_config(page_title="Buscador de Normativa Educativa", page_icon="📚")

# --- 2. CLAVES DE ACCESO (DESDE LA CAJA FUERTE DE STREAMLIT) ---
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

supabase = init_supabase()
model = load_model()
groq_client = Groq(api_key=GROQ_API_KEY)

# --- 4. INTERFAZ WEB ---
st.title("📚 Buscador Inteligente de Normativa Educativa")
st.write("Selecciona el nivel educativo y haz tu pregunta para encontrar la ley exacta.")

# Menú desplegable con el ORDEN SOLICITADO
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
    # Verificamos si el usuario ha elegido un nivel antes de buscar
    if bloque_elegido == "ninguno":
        st.warning("⚠️ Por favor, selecciona un nivel educativo en el menú desplegable antes de buscar.")
    else:
        with st.spinner("Buscando en las leyes y redactando la respuesta..."):
            try:
                # PASO 1: El bibliotecario (Supabase) busca los fragmentos
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
                    # PASO 2: Preparamos los textos crudos
                    contexto_para_ia = ""
                    fuentes_lista = []
                    
                    for res in resultados:
                        nombre_limpio = res['nombre_archivo'].replace(".pdf", "").replace("_", " ")
                        contexto_para_ia += f"DOCUMENTO: {nombre_limpio} | PÁGINA: {res['pagina_num']}\nCONTENIDO: {res['contenido']}\n\n"
                        fuentes_lista.append(f"{nombre_limpio} (Pág. {res['pagina_num']})")
                    
                    fuentes_unicas = list(dict.fromkeys(fuentes_lista))

                    # PASO 3: El escritor (Groq) lee el contexto y redacta 
                    prompt_sistema = (
                        "Eres un experto asesor en normativa educativa. "
                        "Tu tarea es responder a la pregunta del usuario utilizando ÚNICAMENTE la información proporcionada en el contexto. "
                        "Si mencionas algo específico, indica en tu redacción el nombre del documento y la página tal y como aparece en el contexto. "
                        "Escribe de forma clara, estructurada, resumiendo los puntos clave de forma natural."
                    )
                    
                    prompt_usuario = f"CONTEXTO DE LAS LEYES:\n{contexto_para_ia}\n\nPREGUNTA DEL USUARIO: {pregunta}"

                    respuesta_ia = groq_client.chat.completions.create(
                        model="llama-3.1-8b-instant", 
                        messages=[
                            {"role": "system", "content": prompt_sistema},
                            {"role": "user", "content": prompt_usuario}
                        ],
                        temperature=0.2 
                    )

                    texto_final = respuesta_ia.choices[0].message.content

                    # PASO 4: Mostrar el resultado exacto
                    st.write("---")
                    st.markdown(texto_final)
                    
                    st.markdown("### 📚 Fuentes consultadas:")
                    for fuente in fuentes_unicas:
                        st.markdown(f"- 📄 **{fuente}**")
                    
                else:
                    st.warning("No he encontrado nada específico en este bloque con esas palabras. Prueba a reformular la pregunta.")

            except Exception as e:
                st.error(f"Error técnico al buscar: {e}")
