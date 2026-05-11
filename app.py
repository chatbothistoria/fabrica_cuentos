import streamlit as st
from supabase import create_client
from sentence_transformers import SentenceTransformer

# --- 1. CONFIGURACIÓN DE LA PÁGINA ---
st.set_page_config(page_title="Buscador de Normativa Educativa", page_icon="📚")

# --- 2. CONEXIÓN A SUPABASE ---
# Tus claves de conexión
SUPABASE_URL = "https://dwclbwmdybdlxehvkhtz.supabase.co"
SUPABASE_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImR3Y2xid21keWJkbHhlaHZraHR6Iiwicm9sZSI6ImFub24iLCJpYXQiOjE3Nzg0Mzc4MzUsImV4cCI6MjA5NDAxMzgzNX0.37vb-v1ByPiyLB94GVUaJwxu0wfropa4Xpx3lQJ-oFY"

@st.cache_resource
def init_supabase():
    return create_client(SUPABASE_URL, SUPABASE_KEY)

supabase = init_supabase()

# --- 3. CARGAR EL MOTOR DE IA (EMBEDDINGS) ---
@st.cache_resource
def load_model():
    return SentenceTransformer("paraphrase-multilingual-mpnet-base-v2")

model = load_model()

# --- 4. INTERFAZ WEB ---
st.title("📚 Buscador Inteligente de Normativa Educativa")
st.write("Selecciona el nivel educativo y haz tu pregunta para encontrar la ley exacta.")

# El nuevo menú desplegable para elegir el bloque
bloque_elegido = st.selectbox(
    "Nivel educativo:",
    ["fp", "infantil_primaria", "secundaria_bachillerato"],
    format_func=lambda x: {
        "fp": "Formación Profesional",
        "infantil_primaria": "Infantil y Primaria",
        "secundaria_bachillerato": "Secundaria y Bachillerato"
    }[x]
)

pregunta = st.text_input("Haz tu pregunta sobre la normativa:")

if st.button("Buscar") and pregunta:
    with st.spinner("Buscando en las leyes de este bloque..."):
        try:
            # 1. Convertimos la pregunta del usuario a números
            embedding_pregunta = model.encode(pregunta).tolist()

            # 2. Buscamos en Supabase pasando el bloque elegido como filtro
            respuesta = supabase.rpc(
                "buscar_normativa", 
                {
                    "query_embedding": embedding_pregunta, 
                    "filtro_bloque": bloque_elegido, # <-- AQUÍ APLICAMOS EL FILTRO
                    "match_threshold": 0.3, # Nivel de coincidencia (0.3 es flexible)
                    "match_count": 5 # Número de resultados a mostrar
                }
            ).execute()

            resultados = respuesta.data

            # 3. Mostrar los resultados
            if resultados:
                st.success("¡He encontrado estos fragmentos en la normativa!")
                for i, res in enumerate(resultados):
                    st.markdown(f"#### 📄 Documento: `{res['nombre_archivo']}` (Página {res['pagina_num']})")
                    st.info(res['contenido'])
                    st.write("---")
            else:
                st.warning("No he encontrado nada específico en este bloque con esas palabras. Prueba a reformular la pregunta.")

        except Exception as e:
            st.error(f"Error técnico al buscar: {e}")