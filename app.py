import streamlit as st
from supabase import create_client
from sentence_transformers import SentenceTransformer
from groq import Groq

# --- 1. CONFIGURACIÓN ---
st.set_page_config(page_title="Asistente de Normativa", page_icon="⚖️")
st.title("⚖️ Asistente de Normativa Educativa")

# --- 2. CONEXIÓN ---
url = st.secrets["SUPABASE_URL"]
key = st.secrets["SUPABASE_KEY"]
groq_api_key = st.secrets["GROQ_API_KEY"]

supabase = create_client(url, key)
groq_client = Groq(api_key=groq_api_key)

# --- 3. MODELO DE BÚSQUEDA ---
@st.cache_resource
def load_model():
    return SentenceTransformer('paraphrase-multilingual-mpnet-base-v2')

model = load_model()

# --- 4. INTERFAZ ---
bloque = st.sidebar.selectbox("Selecciona el bloque:", ["general", "infantil_primaria", "secundaria_bachillerato", "fp"])
query = st.text_input("Escribe tu duda aquí:")

if query:
    with st.spinner("Buscando en los documentos oficiales..."):
        # Usamos la frase entera para el vector (entiende mejor el contexto)
        query_embedding = model.encode(query).tolist()

        try:
            # BUSCAMOS EN SUPABASE
            res = supabase.rpc(
                "match_normativa_educativa",
                {
                    "query_embedding": query_embedding,
                    "query_text": query, 
                    "match_threshold": 0.15, # Filtro suave para que atrape la información
                    "match_count": 15,       # Subimos a 15 (límite seguro para Groq)
                    "filter_bloque": bloque,
                }
            ).execute()

            if res.data and len(res.data) > 0:
                contexto = "\n\n".join([item['contenido'] for item in res.data])
                
                # Cortafuegos de seguridad
                if len(contexto) > 15000:
                    contexto = contexto[:15000]

                # --- 5. SOLICITUD A GROQ ---
                # Hemos relajado un pelín la orden para que sepa interpretar sinónimos ("Corresponde a la tutoría" = "Funciones del tutor")
                prompt_sistema = """Eres un consultor legal experto en normativa educativa.
                Tu labor es responder a la pregunta del usuario basándote EXCLUSIVAMENTE en el CONTEXTO proporcionado.
                
                REGLAS:
                1. Redacta la respuesta en formato de párrafo fluido, sintetizando la información.
                2. Si la información está en el contexto pero expresada con otras palabras (ej. "tareas", "corresponde a..."), dedúcelo y úsalo.
                3. NUNCA inventes datos externos. Si tras leer bien el contexto NO hay mención al tema, responde: "La normativa referenciada no contiene información explícita sobre este tema."
                4. Responde directamente sin decir "Según el contexto". """

                prompt_usuario = f"CONTEXTO OBLIGATORIO:\n{contexto}\n\nPREGUNTA: {query}"

                respuesta_ia = groq_client.chat.completions.create(
                    messages=[
                        {"role": "system", "content": prompt_sistema},
                        {"role": "user", "content": prompt_usuario}
                    ],
                    model="llama-3.1-8b-instant",
                    temperature=0.0, 
                )

                # --- 6. RESULTADO FINAL ---
                st.markdown("### Respuesta oficial:")
                st.write(respuesta_ia.choices[0].message.content)
                
                # --- HERRAMIENTAS DE DIAGNÓSTICO ---
                st.divider()
                st.caption("Herramientas de verificación")
                
                with st.expander("🔍 Verificar páginas encontradas"):
                    fuentes = set([f"{i['nombre_archivo']} (Pág {i['pagina_num']})" for i in res.data])
                    for f in fuentes:
                        st.write(f"- {f}")
                
                with st.expander("🛠️ Modo Diagnóstico: Ver texto crudo (Lo que lee la IA)"):
                    st.warning("Esto es exactamente lo que la base de datos le ha pasado a la IA para que lea:")
                    st.write(contexto)

            else:
                st.warning(f"La búsqueda en la base de datos ha devuelto 0 resultados para el bloque '{bloque}'.")

        except Exception as e:
            st.error(f"Error técnico: {e}")
