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
bloque = st.sidebar.selectbox("Bloque:", ["general", "infantil_primaria", "secundaria_bachillerato", "fp"])
query = st.text_input("Escribe tu duda aquí:")

if query:
    with st.spinner("Analizando la normativa al pie de la letra..."):
        query_embedding = model.encode(query).tolist()

        try:
            # BUSCAMOS EN SUPABASE
            res = supabase.rpc(
                "match_normativa_educativa",
                {
                    "query_embedding": query_embedding,
                    "query_text": query,
                    "match_threshold": 0.25, # PUNTO DE EQUILIBRIO: Ni muy estricto ni muy laxo
                    "match_count": 20,       # Le damos 20 fragmentos para que tenga lectura de sobra
                    "filter_bloque": bloque,
                }
            ).execute()

            if res.data:
                contexto = "\n".join([item['contenido'] for item in res.data])
                
                # --- 5. SOLICITUD A GROQ (EL REDACTOR ESTRICTO) ---
                prompt_sistema = """Eres un estricto consultor legal en normativa educativa.
                Tu ÚNICA labor es responder a la pregunta del usuario basándote EXCLUSIVAMENTE en el CONTEXTO proporcionado.
                
                REGLAS INQUEBRANTABLES:
                1. Redacta la respuesta en formato de párrafo (o varios si es necesario), de forma fluida. No uses listas numeradas.
                2. NUNCA inventes, asumas o deduzcas información. Tu conocimiento externo está apagado. Todo dato debe provenir del CONTEXTO.
                3. Si el CONTEXTO proporcionado no contiene la información explícita para responder a la pregunta, DEBES detenerte y responder EXACTAMENTE esto: "La normativa referenciada no contiene información explícita para responder a esta pregunta."
                4. No uses frases de relleno como "Según el contexto proporcionado". Responde directamente."""

                prompt_usuario = f"CONTEXTO OBLIGATORIO:\n{contexto}\n\nPREGUNTA: {query}"

                respuesta_ia = groq_client.chat.completions.create(
                    messages=[
                        {"role": "system", "content": prompt_sistema},
                        {"role": "user", "content": prompt_usuario}
                    ],
                    model="llama-3.1-8b-instant",
                    temperature=0.0, # TEMPERATURA A CERO: Cero creatividad, máxima fidelidad a los textos.
                )

                # --- 6. RESULTADO FINAL ---
                st.markdown(respuesta_ia.choices[0].message.content)
                
                with st.expander("Ver fuentes oficiales consultadas"):
                    fuentes = set([f"{i['nombre_archivo']} (Pág {i['pagina_num']})" for i in res.data])
                    for f in fuentes:
                        st.write(f"- {f}")
            else:
                st.warning("La búsqueda inicial no ha encontrado ningún artículo o fragmento relacionado con esas palabras.")

        except Exception as e:
            st.error(f"Error técnico: {e}")
