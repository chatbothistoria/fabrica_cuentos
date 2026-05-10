import streamlit as st
from supabase import create_client
from sentence_transformers import SentenceTransformer

# 1. CONFIGURACIÓN DE PÁGINA
st.set_page_config(page_title="Buscador de Normativa Educativa", page_icon="📚")

st.title("📚 Buscador de Normativa Inteligente")
st.markdown("Consulta artículos y leyes usando IA y búsqueda por palabras clave.")

# 2. CONEXIÓN A SUPABASE (Usando Secrets)
# Asegúrate de haber configurado estos nombres en los Secrets de Streamlit
url = st.secrets["SUPABASE_URL"]
key = st.secrets["SUPABASE_KEY"]
supabase = create_client(url, key)

# 3. CARGAR MODELO DE IA (Caché para que sea rápido)
@st.cache_resource
def load_model():
    # Usamos el mismo modelo que usamos en Colab para que los vectores coincidan
    return SentenceTransformer('paraphrase-multilingual-mpnet-base-v2')

model = load_model()

# 4. BARRA LATERAL (Filtros)
st.sidebar.header("Opciones de búsqueda")
bloque = st.sidebar.selectbox(
    "Selecciona el bloque:",
    ["general", "infantil_primaria", "secundaria_bachillerato", "fp"]
)
umbral = st.sidebar.slider("Umbral de similitud (IA)", 0.0, 1.0, 0.4)
cantidad = st.sidebar.slider("Número de resultados", 1, 20, 10)

# 5. INTERFAZ DE BÚSQUEDA
query = st.text_input("Introduce tu consulta (ej: 'Evaluación en primaria' o 'Artículo 14'):")

if query:
    with st.spinner("Buscando en la normativa..."):
        # A. Convertimos la pregunta del usuario en un vector
        query_embedding = model.encode(query).tolist()

        # B. Llamamos a nuestra función de "Embudo" en Supabase
        try:
            res = supabase.rpc(
                "match_normativa_educativa",
                {
                    "query_embedding": query_embedding,
                    "query_text": query,  # Enviamos el texto original para la búsqueda híbrida
                    "match_threshold": umbral,
                    "match_count": cantidad,
                    "filter_bloque": bloque,
                }
            ).execute()

            # 6. MOSTRAR RESULTADOS
            if res.data:
                st.success(f"He encontrado {len(res.data)} fragmentos relevantes:")
                
                for item in res.data:
                    with st.expander(f"📄 {item['nombre_archivo']} - Página {item['pagina_num']} (Similitud: {round(item['similarity'] * 100, 2)}%)"):
                        st.write(item['contenido'])
                        st.info(f"Archivo: {item['nombre_archivo']} | Página: {item['pagina_num']}")
            else:
                st.warning("No se han encontrado resultados exactos. Prueba a bajar el umbral de similitud.")

        except Exception as e:
            st.error(f"Error en la búsqueda: {e}")

# Pie de página
st.markdown("---")
st.caption("Sistema de búsqueda híbrida (Semántica + Keywords) optimizado.")