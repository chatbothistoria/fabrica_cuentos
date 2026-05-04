"""Motor narrativo de Mi Fábrica de Cuentos.

Este módulo no usa IA: genera borradores pedagógicos a partir de las elecciones
del alumnado y permite que el texto sea editado, revisado y mejorado.
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any, Dict, List

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

LEVEL_FILES = {
    "infantil": "infantil.json",
    "nivel_6_8": "nivel_6_8.json",
    "nivel_8_10": "nivel_8_10.json",
    "nivel_10_12": "nivel_10_12.json",
}

LEVEL_LABELS = {
    "infantil": "Infantil · 3-5 años",
    "nivel_6_8": "6-8 años",
    "nivel_8_10": "8-10 años",
    "nivel_10_12": "10-12 años",
}


def load_level(level_key: str) -> Dict[str, Any]:
    """Carga los datos pedagógicos de un nivel."""
    if level_key not in LEVEL_FILES:
        raise ValueError(f"Nivel no reconocido: {level_key}")
    path = DATA_DIR / LEVEL_FILES[level_key]
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def empty_selection(level_data: Dict[str, Any]) -> Dict[str, str]:
    """Devuelve una selección inicial vacía para que el alumnado elija conscientemente."""
    return {step["key"]: "" for step in level_data["steps"]}


def random_selection(level_data: Dict[str, Any]) -> Dict[str, str]:
    """Devuelve una selección aleatoria adaptada al nivel."""
    return {step["key"]: random.choice(step["options"]) for step in level_data["steps"]}


def step_labels(level_data: Dict[str, Any]) -> Dict[str, str]:
    """Mapa key -> label para mostrar las elecciones de forma clara."""
    return {step["key"]: step["label"] for step in level_data["steps"]}


def lower_first(text: str) -> str:
    """Convierte en minúscula solo la primera letra, si procede."""
    if not text:
        return text
    return text[0].lower() + text[1:]


def sentence(text: str) -> str:
    """Normaliza una frase sencilla terminada en punto."""
    cleaned = text.strip()
    if not cleaned:
        return ""
    if cleaned[-1] not in ".!?":
        cleaned += "."
    return cleaned


def make_title(level_key: str, s: Dict[str, str]) -> str:
    """Propone un título según el nivel y las elecciones."""
    if level_key == "infantil":
        return f"{s['personaje']} en {s['lugar']}"
    if level_key == "nivel_6_8":
        return f"{s['personaje']} y la {s['objeto_especial']}"
    if level_key == "nivel_8_10":
        return f"{s['protagonista']} y el secreto de la {s['objeto_pista']}"
    if level_key == "nivel_10_12":
        return f"Una historia sobre {lower_first(s['tema'])}"
    return "Mi cuento"


def planning_summary(level_data: Dict[str, Any], selection: Dict[str, str]) -> List[str]:
    """Crea una lista legible con los elementos narrativos elegidos."""
    labels = step_labels(level_data)
    summary = []
    for key in level_data["structure"]:
        value = selection.get(key, "")
        if value:
            summary.append(f"{labels.get(key, key.replace('_', ' ').title())}: {value}")
    return summary


def generate_draft(level_key: str, selection: Dict[str, str]) -> str:
    """Genera un borrador inicial adaptado a la edad.

    Este borrador debe entenderse como punto de partida. La app invita a que
    el alumnado lo complete, lo lea, lo revise y lo personalice.
    """
    s = selection

    if level_key == "infantil":
        return (
            f"Había una vez {lower_first(s['personaje'])} que estaba en {lower_first(s['lugar'])}.\n\n"
            f"Un día, {lower_first(s['problema'])}. Entonces decidió vivir una aventura: {lower_first(s['aventura'])}.\n\n"
            f"Para solucionarlo, {lower_first(s['solucion'])}.\n\n"
            f"Al final, {lower_first(s['final'])}."
        )

    if level_key == "nivel_6_8":
        return (
            f"Había una vez {lower_first(s['personaje'])} que vivía una aventura en {lower_first(s['lugar'])}. "
            f"Su gran deseo era {lower_first(s['deseo'])}.\n\n"
            f"Pero un día ocurrió un problema: {lower_first(s['problema'])}. "
            f"Por suerte, recibió la ayuda de {lower_first(s['ayudante'])} y encontró una {lower_first(s['objeto_especial'])}.\n\n"
            f"Después, {lower_first(s['aventura'])}. "
            f"Para resolverlo, {lower_first(s['solucion'])}.\n\n"
            f"Finalmente, {lower_first(s['final'])}."
        )

    if level_key == "nivel_8_10":
        return (
            f"Este es un cuento de {lower_first(s['genero'])}. Su protagonista es {s['protagonista']}, "
            f"un personaje {lower_first(s['rasgo'])}. La historia comienza en una {lower_first(s['lugar'])}, "
            f"donde {lower_first(s['situacion_inicial'])}.\n\n"
            f"El protagonista quería {lower_first(s['deseo'])}, pero apareció un problema: {lower_first(s['problema'])}. "
            f"Todo ocurrió por {lower_first(s['causa'])}.\n\n"
            f"Con la ayuda de {lower_first(s['ayudante'])} y gracias a una {lower_first(s['objeto_pista'])}, "
            f"comenzó la aventura. Primero, {lower_first(s['aventura_1'])}. Después, {lower_first(s['aventura_2'])}.\n\n"
            f"En el momento más importante, alguien dijo: {s['dialogo']}\n\n"
            f"La solución llegó cuando {lower_first(s['solucion'])}. Desde entonces, el protagonista aprendió "
            f"{lower_first(s['cambio_personaje'])}.\n\n"
            f"Al final, {lower_first(s['final'])}."
        )

    if level_key == "nivel_10_12":
        return (
            f"Este cuento de {lower_first(s['genero'])} trata sobre {lower_first(s['tema'])}. "
            f"El personaje principal es {s['protagonista']}. Su objetivo es {lower_first(s['objetivo'])}.\n\n"
            f"El conflicto externo aparece cuando {lower_first(s['conflicto_externo'])}. "
            f"Al mismo tiempo, vive un conflicto interno: {lower_first(s['conflicto_interno'])}. "
            f"La fuerza que se opone a su avance es {lower_first(s['antagonista'])}.\n\n"
            f"La historia se desarrolla principalmente en {lower_first(s['lugar_principal'])}, "
            f"aunque también aparece {lower_first(s['lugar_secundario'])}. Todo comienza cuando {lower_first(s['detonante'])}.\n\n"
            f"Primero, {lower_first(s['obstaculo_1'])}. Más tarde, la situación se complica porque {lower_first(s['obstaculo_2'])}. "
            f"El protagonista debe tomar una decisión importante: {lower_first(s['decision'])}.\n\n"
            f"Entonces llega el giro narrativo: {lower_first(s['giro_narrativo'])}. En el clímax, {lower_first(s['climax'])}.\n\n"
            f"El conflicto se resuelve cuando {lower_first(s['solucion'])}. El protagonista aprende "
            f"{lower_first(s['aprendizaje'])}.\n\n"
            f"El cuento termina con un final {lower_first(s['tipo_final'])}."
        )

    return ""


def guided_sections(level_key: str, draft: str) -> List[Dict[str, str]]:
    """Propone secciones de escritura según la edad."""
    if level_key == "infantil":
        return [
            {"title": "Cuento narrado", "help": "El adulto puede leerlo y el niño o la niña puede contarlo con sus palabras.", "default": draft},
            {"title": "Lo que ha contado el alumno o alumna", "help": "Opcional: escribe aquí una frase, una idea o una transcripción breve de su narración oral.", "default": ""},
        ]

    if level_key == "nivel_6_8":
        return [
            {"title": "Inicio", "help": "Presenta al personaje y el lugar.", "default": draft.split("\n\n")[0]},
            {"title": "Problema", "help": "Cuenta qué ocurrió y quién ayuda.", "default": draft.split("\n\n")[1]},
            {"title": "Aventura y final", "help": "Explica cómo se resuelve y cómo termina.", "default": "\n\n".join(draft.split("\n\n")[2:])},
        ]

    if level_key == "nivel_8_10":
        return [
            {"title": "Presentación", "help": "Presenta género, protagonista, rasgo y lugar.", "default": draft.split("\n\n")[0]},
            {"title": "Problema y causa", "help": "Explica qué ocurre y por qué.", "default": draft.split("\n\n")[1]},
            {"title": "Aventuras", "help": "Añade obstáculos, pistas y acciones.", "default": draft.split("\n\n")[2]},
            {"title": "Diálogo", "help": "Incluye una frase importante y quién la dice.", "default": draft.split("\n\n")[3]},
            {"title": "Solución y final", "help": "Cuenta qué aprende el personaje y cómo termina.", "default": "\n\n".join(draft.split("\n\n")[4:])},
        ]

    return [
        {"title": "Planteamiento", "help": "Presenta género, tema, protagonista, objetivo y conflictos.", "default": "\n\n".join(draft.split("\n\n")[:2])},
        {"title": "Escenarios y detonante", "help": "Sitúa la historia y explica qué pone todo en marcha.", "default": draft.split("\n\n")[2]},
        {"title": "Nudo", "help": "Desarrolla obstáculos, decisiones y giro narrativo.", "default": draft.split("\n\n")[3]},
        {"title": "Clímax", "help": "Escribe el momento más intenso de la historia.", "default": draft.split("\n\n")[4]},
        {"title": "Desenlace", "help": "Resuelve el conflicto y muestra el aprendizaje final.", "default": draft.split("\n\n")[5]},
    ]


def combine_sections(section_texts: Dict[str, str]) -> str:
    """Une secciones escritas por el alumnado en un texto final."""
    parts = [text.strip() for _, text in section_texts.items() if text and text.strip()]
    return "\n\n".join(parts)


def make_txt(title: str, author: str, level_label: str, story_text: str, selection_lines: List[str]) -> str:
    """Crea una versión descargable en texto plano."""
    author_line = author.strip() if author and author.strip() else "Autor/a sin indicar"
    elements = "\n".join(f"- {line}" for line in selection_lines)
    return (
        f"{title}\n"
        f"Autor/a: {author_line}\n"
        f"Nivel: {level_label}\n\n"
        f"ELEMENTOS DEL CUENTO\n{elements}\n\n"
        f"CUENTO\n{story_text.strip()}\n"
    )
