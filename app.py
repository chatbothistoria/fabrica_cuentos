from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import streamlit as st

from core.pdf_exporter import story_to_pdf_bytes
from core.story_engine import (
    LEVEL_LABELS,
    combine_sections,
    empty_selection,
    generate_draft,
    guided_sections,
    load_level,
    make_title,
    make_txt,
    planning_summary,
    random_selection,
)

APP_TITLE = "Mi Fábrica de Cuentos"

st.set_page_config(
    page_title=APP_TITLE,
    page_icon="🏭",
    layout="wide",
    initial_sidebar_state="expanded",
)


CUSTOM_CSS = """
<style>
:root {
    --card-bg: #fffdf7;
    --soft-border: #f0dfc8;
}
.main .block-container {
    padding-top: 2rem;
    padding-bottom: 3rem;
}
.story-card {
    padding: 1.1rem 1.25rem;
    border-radius: 1.2rem;
    border: 1px solid var(--soft-border);
    background: var(--card-bg);
    box-shadow: 0 2px 12px rgba(70, 50, 20, 0.08);
    margin-bottom: 1rem;
}
.big-title {
    font-size: 2.4rem;
    line-height: 1.1;
    font-weight: 800;
    margin-bottom: 0.2rem;
}
.subtitle {
    font-size: 1.08rem;
    color: #5f5a50;
}
.small-note {
    font-size: 0.92rem;
    color: #6f675e;
}
.badge {
    display: inline-block;
    padding: 0.25rem 0.55rem;
    margin: 0.15rem 0.2rem 0.15rem 0;
    border-radius: 999px;
    background: #f6efe4;
    border: 1px solid #ead8bd;
    font-size: 0.85rem;
}
</style>
"""

st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


@st.cache_data(show_spinner=False)
def cached_level(level_key: str) -> dict:
    return load_level(level_key)


def init_state() -> None:
    defaults = {
        "level_key": "infantil",
        "selection": {},
        "story_title": "",
        "author": "",
        "draft": "",
        "section_texts": {},
        "final_story": "",
        "generated_at": "",
    }
    for key, value in defaults.items():
        st.session_state.setdefault(key, value)


init_state()


def set_level(level_key: str) -> None:
    if st.session_state.level_key != level_key:
        st.session_state.level_key = level_key
        level_data = cached_level(level_key)
        st.session_state.selection = empty_selection(level_data)
        st.session_state.story_title = ""
        st.session_state.draft = ""
        st.session_state.section_texts = {}
        st.session_state.final_story = ""


def current_level_data() -> dict:
    return cached_level(st.session_state.level_key)


def render_header() -> None:
    st.markdown(
        f"""
        <div class="story-card">
            <div class="big-title">🏭 {APP_TITLE}</div>
            <div class="subtitle">Una app para crear cuentos personalizados de 3 a 12 años, con apoyo pedagógico por nivel.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_level_summary(level_data: dict) -> None:
    st.markdown(
        f"""
        <div class="story-card">
            <h3>{level_data['level_name']} · {level_data['age_range']}</h3>
            <p>{level_data['subtitle']}</p>
            <p><strong>Modo de trabajo:</strong> {level_data['writing_mode']}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.expander("Ver enfoque pedagógico de este nivel"):
        for item in level_data["pedagogical_focus"]:
            st.write(f"• {item}")


def ensure_selection(level_data: dict) -> None:
    expected_keys = [step["key"] for step in level_data["steps"]]
    if not st.session_state.selection or set(st.session_state.selection.keys()) != set(expected_keys):
        st.session_state.selection = empty_selection(level_data)


def generate_story(level_key: str, level_data: dict) -> None:
    title = st.session_state.story_title.strip()
    if not title:
        title = make_title(level_key, st.session_state.selection)
    st.session_state.story_title = title
    st.session_state.draft = generate_draft(level_key, st.session_state.selection)
    sections = guided_sections(level_key, st.session_state.draft)
    st.session_state.section_texts = {section["title"]: section["default"] for section in sections}
    st.session_state.final_story = combine_sections(st.session_state.section_texts)
    st.session_state.generated_at = datetime.now().strftime("%d/%m/%Y %H:%M")


def render_inicio() -> None:
    render_header()
    col1, col2 = st.columns([1.2, 1])
    with col1:
        st.markdown("### ¿Qué hace esta app?")
        st.write(
            "El alumnado construye un cuento eligiendo elementos narrativos adaptados a su edad: "
            "personajes, lugares, problemas, aventuras, emociones, conflictos, giros y finales."
        )
        st.write(
            "La app genera un borrador inicial que puede leerse, editarse, revisarse y descargar en PDF o TXT. "
            "No sustituye la creatividad del niño o la niña: la guía."
        )
        st.markdown("### Progresión educativa")
        st.write("**Infantil:** imágenes, oralidad y secuencias sencillas.")
        st.write("**6-8 años:** frases guiadas, conectores y cuentos breves.")
        st.write("**8-10 años:** párrafos, diálogos, emociones y descripciones.")
        st.write("**10-12 años:** conflictos internos, giros, clímax y revisión de estilo.")
    with col2:
        st.markdown("### Flujo de creación")
        st.markdown(
            """
            1. Elige el nivel educativo.  
            2. Selecciona los elementos del cuento.  
            3. Genera un borrador.  
            4. Escribe y mejora.  
            5. Descarga el cuento final.  
            """
        )
        st.info("Primera versión sin base de datos: los cuentos se descargan, pero no se guardan online.")


def render_crear_cuento() -> None:
    st.header("🪄 Crear cuento")
    level_keys = list(LEVEL_LABELS.keys())
    selected_label = st.selectbox(
        "Elige nivel educativo",
        options=list(LEVEL_LABELS.values()),
        index=level_keys.index(st.session_state.level_key),
    )
    selected_key = level_keys[list(LEVEL_LABELS.values()).index(selected_label)]
    set_level(selected_key)
    level_data = current_level_data()
    ensure_selection(level_data)

    render_level_summary(level_data)

    author = st.text_input("Nombre del autor o autora", value=st.session_state.author, placeholder="Ej.: Luna, 2ºB")
    st.session_state.author = author

    title = st.text_input(
        "Título del cuento",
        value=st.session_state.story_title,
        placeholder="Puedes dejarlo vacío y la app propondrá uno.",
    )
    st.session_state.story_title = title

    col_random, col_reset = st.columns([1, 1])
    with col_random:
        if st.button("🎲 Sorpréndeme", use_container_width=True):
            st.session_state.selection = random_selection(level_data)
            st.session_state.draft = ""
            st.session_state.section_texts = {}
            st.session_state.final_story = ""
            st.rerun()
    with col_reset:
        if st.button("↩️ Reiniciar elecciones", use_container_width=True):
            st.session_state.selection = empty_selection(level_data)
            st.session_state.draft = ""
            st.session_state.section_texts = {}
            st.session_state.final_story = ""
            st.rerun()

    st.subheader("1. Elige las piezas de tu cuento")
    st.caption("Las opciones cambian según la edad y la complejidad narrativa del nivel.")

    columns = st.columns(2)
    for idx, step in enumerate(level_data["steps"]):
        key = step["key"]
        current = st.session_state.selection.get(key, step["options"][0])
        default_index = step["options"].index(current) if current in step["options"] else 0
        with columns[idx % 2]:
            st.session_state.selection[key] = st.selectbox(
                f"{step['label']} — {step['prompt']}",
                options=step["options"],
                index=default_index,
                key=f"select_{st.session_state.level_key}_{key}",
            )

    st.subheader("2. Genera el borrador")
    if st.button("✨ Generar borrador de cuento", type="primary", use_container_width=True):
        generate_story(st.session_state.level_key, level_data)
        st.success("Borrador generado. Ve a “Escribir y mejorar” para personalizarlo.")

    if st.session_state.draft:
        st.markdown("### Vista rápida del borrador")
        st.text_area("Borrador generado", st.session_state.draft, height=260, disabled=True)


def render_taller() -> None:
    st.header("📚 Escribir y mejorar")
    level_data = current_level_data()

    if not st.session_state.draft:
        st.warning("Primero genera un borrador en la sección “Crear cuento”.")
        return

    st.markdown(f"### {st.session_state.story_title or 'Mi cuento'}")
    st.caption(f"Nivel: {LEVEL_LABELS[st.session_state.level_key]}")

    left, right = st.columns([1.15, 0.85])
    with left:
        st.subheader("Plan del cuento")
        for line in planning_summary(level_data, st.session_state.selection):
            st.markdown(f"<span class='badge'>{line}</span>", unsafe_allow_html=True)

    with right:
        st.subheader("Preguntas guía")
        for question in level_data["guided_questions"]:
            st.write(f"• {question}")

    st.divider()
    st.subheader("Edita por partes")
    st.caption("Puedes dejar el borrador como está o cambiarlo completamente. Lo importante es que el cuento sea tuyo.")

    sections = guided_sections(st.session_state.level_key, st.session_state.draft)
    new_sections = {}
    for section in sections:
        title = section["title"]
        default_text = st.session_state.section_texts.get(title, section["default"])
        with st.expander(f"✏️ {title}", expanded=True):
            st.caption(section["help"])
            new_sections[title] = st.text_area(
                label=f"Texto de {title}",
                value=default_text,
                height=180 if st.session_state.level_key in ["nivel_8_10", "nivel_10_12"] else 140,
                key=f"area_{st.session_state.level_key}_{title}",
            )

    st.session_state.section_texts = new_sections
    st.session_state.final_story = combine_sections(new_sections)

    st.divider()
    col_bank, col_tasks = st.columns(2)
    with col_bank:
        st.subheader("Banco de palabras")
        for group, words in level_data["word_bank"].items():
            st.markdown(f"**{group}**")
            st.write(", ".join(words))
    with col_tasks:
        st.subheader("Retos de mejora")
        for task in level_data["improvement_tasks"]:
            st.checkbox(task, key=f"task_{st.session_state.level_key}_{task}")

    st.success("Los cambios se guardan mientras esta sesión del navegador siga abierta.")


def render_cuento_final() -> None:
    st.header("🎨 Cuento final")
    level_data = current_level_data()

    if not st.session_state.final_story:
        st.warning("Todavía no hay cuento final. Genera un borrador y edítalo en el taller.")
        return

    title = st.session_state.story_title or make_title(st.session_state.level_key, st.session_state.selection)
    author = st.session_state.author or "Autor/a sin indicar"
    level_label = LEVEL_LABELS[st.session_state.level_key]
    selection_lines = planning_summary(level_data, st.session_state.selection)

    st.markdown(
        f"""
        <div class="story-card">
            <h2>{title}</h2>
            <p><strong>Autor/a:</strong> {author}</p>
            <p><strong>Nivel:</strong> {level_label}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.subheader("Texto del cuento")
    st.write(st.session_state.final_story)

    txt = make_txt(title, author, level_label, st.session_state.final_story, selection_lines)
    pdf_bytes = story_to_pdf_bytes(title, author, level_label, st.session_state.final_story, selection_lines)

    safe_filename = "mi_fabrica_de_cuentos"
    col1, col2, col3 = st.columns(3)
    with col1:
        st.download_button(
            "⬇️ Descargar TXT",
            data=txt.encode("utf-8"),
            file_name=f"{safe_filename}.txt",
            mime="text/plain",
            use_container_width=True,
        )
    with col2:
        st.download_button(
            "⬇️ Descargar PDF",
            data=pdf_bytes,
            file_name=f"{safe_filename}.pdf",
            mime="application/pdf",
            use_container_width=True,
        )
    with col3:
        story_data = {
            "app": APP_TITLE,
            "title": title,
            "author": author,
            "level": level_label,
            "selection": st.session_state.selection,
            "story": st.session_state.final_story,
            "generated_at": st.session_state.generated_at,
        }
        st.download_button(
            "⬇️ Descargar JSON",
            data=json.dumps(story_data, ensure_ascii=False, indent=2).encode("utf-8"),
            file_name=f"{safe_filename}.json",
            mime="application/json",
            use_container_width=True,
        )

    with st.expander("Ver elementos narrativos usados"):
        for line in selection_lines:
            st.write(f"• {line}")


def render_guia_docente() -> None:
    st.header("✅ Guía docente")
    st.write(
        "Esta sección resume cómo usar la app en el aula. La herramienta está pensada para acompañar "
        "la creación, no para sustituir la escritura ni la conversación literaria."
    )

    level_data = current_level_data()
    render_level_summary(level_data)

    st.subheader("Sugerencias para este nivel")
    for suggestion in level_data["teacher_suggestions"]:
        st.write(f"• {suggestion}")

    st.subheader("Ideas de evaluación sencilla")
    st.write("• ¿El cuento tiene una secuencia comprensible?")
    st.write("• ¿El alumno o alumna ha tomado decisiones creativas?")
    st.write("• ¿Ha revisado o mejorado algo del borrador?")
    st.write("• ¿El nivel de ayuda ha sido adecuado para su edad?")
    st.write("• ¿Puede explicar oralmente de qué trata su cuento?")

    st.subheader("Privacidad y uso responsable")
    st.write(
        "Esta primera versión no guarda datos en servidor ni crea perfiles. Para uso real con alumnado, "
        "conviene evitar introducir apellidos, fotos personales u otros datos identificativos."
    )


def render_configuracion() -> None:
    st.header("⚙️ Configuración y próximos pasos")
    st.markdown("### Qué incluye esta versión")
    st.write("• Cuatro niveles educativos diferenciados.")
    st.write("• Datos pedagógicos separados en JSON.")
    st.write("• Motor narrativo sin IA.")
    st.write("• Escritura guiada por secciones.")
    st.write("• Descarga en TXT, PDF y JSON.")
    st.write("• Guía docente básica.")

    st.markdown("### Qué se puede añadir después")
    st.write("• Biblioteca con usuarios y cuentos guardados.")
    st.write("• Panel docente con clases y retos.")
    st.write("• Ilustraciones o portadas personalizadas.")
    st.write("• Grabación de voz para Infantil.")
    st.write("• IA de apoyo para títulos, preguntas, vocabulario y revisión.")
    st.write("• Exportación de libros colectivos de aula.")

    st.info("Para guardar cuentos online hará falta conectar una base de datos como Supabase, Firebase o Google Sheets.")


with st.sidebar:
    st.title("🏭 Mi Fábrica")
    page = st.radio(
        "Navegación",
        [
            "Inicio",
            "Crear cuento",
            "Escribir y mejorar",
            "Cuento final",
            "Guía docente",
            "Configuración",
        ],
    )
    st.divider()
    st.caption("Nivel activo")
    active_level = LEVEL_LABELS.get(st.session_state.level_key, "Infantil")
    st.write(f"**{active_level}**")
    if st.session_state.story_title:
        st.caption("Cuento activo")
        st.write(st.session_state.story_title)

if page == "Inicio":
    render_inicio()
elif page == "Crear cuento":
    render_crear_cuento()
elif page == "Escribir y mejorar":
    render_taller()
elif page == "Cuento final":
    render_cuento_final()
elif page == "Guía docente":
    render_guia_docente()
else:
    render_configuracion()
