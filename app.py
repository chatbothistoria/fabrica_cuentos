import streamlit as st
from supabase import create_client
from sentence_transformers import SentenceTransformer
from groq import Groq

# --- 1. CONFIGURACIÓN DE PÁGINA ---
st.set_page_config(page_title="Asistente de Normativa Educativa", page_icon="🧠")
st.title("🧠 Asistente de Normativa Educativa")
st.markdown("Haz una pregunta y la IA redactará la respuesta basándose **exclusivamente** en tus documentos oficiales.")

# --- 2. CREDENCIALES (Secrets) ---
url = st.secrets["SUPABASE_URL"]
key = st.secrets["SUPABASE_KEY"]
groq_api_key = st.secrets["GROQ_API_KEY"] 

# Inicializar clientes
supabase = create_client(url, key)
groq_client = Groq(api_key=groq_api_key)

# --- 3. CARGAR MODELO DE EMBEDDINGS (El "Buscador") ---
@st.cache_resource
def load_model():
    return SentenceTransformer('paraphrase-multilingual-mpnet-base-v2')

model = load_model()

# --- 4. BARRA LATERAL (Filtros) ---
st.sidebar.header("Filtros")
bloque = st.sidebar.selectbox("Bloque normativo:", ["general", "infantil_primaria", "secundaria_bachillerato", "fp"])

# --- 5. INTERFAZ DE USUARIO ---
query = st.text_input("Haz tu pregunta (ej: '¿Cuáles son los criterios de evaluación en primaria?'):")

if query:
    with st.spinner("1️⃣ Buscando en los PDFs (Búsqueda Híbrida)..."):
        # Convertir pregunta a vector
        query_embedding = model.encode(query).tolist()

        try:
            # Buscar los mejores fragmentos en Supabase
            res = supabase.rpc(
                "match_normativa_educativa",
                {
                    "query_embedding": query_embedding,
                    "query_text": query,
                    "match_threshold": 0.3, # Umbral bajo para asegurar que pillamos contexto
                    "match_count": 5,       # Pasamos los 5 mejores trozos a Groq
                    "filter_bloque": bloque,
                }
            ).execute()

            if not res.data:
                st.warning("No he encontrado información relevante en los documentos para esta consulta.")
            else:
                st.success("2️⃣ Información encontrada. Redactando respuesta...")
                
                # Unir los fragmentos encontrados para dárselos a Groq
                contexto_crudo = "\n\n".join([item['contenido'] for item in res.data])
                # Sacar las fuentes únicas para mostrarlas después
                fuentes = set([f"{item['nombre_archivo']} (Pág {item['pagina_num']})" for item in res.data])

                # --- 6. LLAMADA A GROQ (El "Redactor") ---
                prompt_sistema = """Eres un asistente experto en normativa educativa. 
                Tu tarea es responder a la pregunta del usuario utilizando ÚNICAMENTE la información contenida en el CONTEXTO proporcionado.
                REGLAS ESTRICTAS:
                1. Escribe la respuesta en UN SOLO PÁRRAFO continuo.
                2. Usa un lenguaje natural, directo y fácil de entender.
                3. Si la respuesta a la pregunta no está en el CONTEXTO, responde EXACTAMENTE: 'No dispongo de información en la normativa referenciada para responder a esta pregunta.'
                4. NUNCA inventes información que no esté en el contexto."""

                prompt_usuario = f"CONTEXTO:\n{contexto_crudo}\n\nPREGUNTA DEL USUARIO: {query}"

                chat_completion = groq_client.chat.completions.create(
                    messages=[
                        {"role": "system", "content": prompt_sistema},
                        {"role": "user", "content": prompt_usuario}
                    ],
                    # AQUÍ ESTÁ LA MAGIA: Modelo actualizado y súper rápido
                    model="llama-3.1-8b-instant", 
                    temperature=0.1, # Temperatura baja (0.1) para que no sea creativo, sino estricto con los datos
                )

                respuesta_final = chat_completion.choices[0].message.content

                # --- 7. MOSTRAR RESULTADO ---
                st.markdown("### Respuesta")
                st.info(respuesta_final)
                
                # Mostrar de dónde ha sacado la info (oculto por defecto para que quede limpio)
                with st.expander("📄 Ver fuentes utilizadas"):
                    st.markdown("La IA ha construido esta respuesta basándose en estos documentos exactos:")
                    for fuente in fuentes:
                        st.write(f"- {fuente}")

        except Exception as e:
            st.error(f"Se ha producido un error: {e}")

# --- PIE DE PÁGINA ---
st.markdown("---")
st.caption("Sistema RAG: Búsqueda Híbrida (Supabase) + Generación de Respuestas (Groq / Llama 3.1)")
