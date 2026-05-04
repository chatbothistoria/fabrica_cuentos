"""Comprobación de prioridad entre desplegable y campo libre.

Ejecutar desde la raíz del proyecto:
    python scripts/check_selection_priority.py

Qué valida:
1. Si el alumno elige una opción del desplegable y escribe una opción libre,
   la opción libre tiene prioridad y se usa en el cuento.
2. Si el campo libre está vacío, se usa la opción del desplegable.
3. La generación de borrador, secciones, resumen y PDF no falla.
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from core.pdf_exporter import story_to_pdf_bytes
from core.story_engine import (
    LEVEL_FILES,
    LEVEL_LABELS,
    combine_sections,
    empty_selection,
    generate_draft,
    guided_sections,
    load_level,
    make_title,
    planning_summary,
)


def apply_choice(selection: dict[str, str], key: str, selected_option: str, custom_value: str) -> dict[str, str]:
    """Replica la lógica usada en app.py para cada pieza narrativa."""
    custom_value = custom_value.strip()
    if custom_value:
        selection[key] = custom_value
    else:
        selection[key] = selected_option
    return selection


def build_outputs(level_key: str, level_data: dict, selection: dict[str, str]) -> tuple[str, str, list[str]]:
    title = make_title(level_key, selection)
    draft = generate_draft(level_key, selection)
    sections = guided_sections(level_key, draft)
    final_story = combine_sections({section["title"]: section["default"] for section in sections})
    summary = planning_summary(level_data, selection)
    return title, final_story, summary


def main() -> int:
    errors: list[dict[str, str]] = []
    stats: list[tuple[str, int, int, int]] = []
    custom_override_cases = 0
    dropdown_default_cases = 0

    for level_key in LEVEL_FILES:
        level_data = load_level(level_key)
        level_custom_cases = 0
        level_default_cases = 0

        for step in level_data["steps"]:
            key = step["key"]

            for option_index, selected_option in enumerate(step["options"]):
                # Caso 1: el desplegable tiene una opción y el campo libre también tiene texto.
                custom_value = f"opcion_libre_{level_key}_{key}_{option_index}"
                try:
                    selection = empty_selection(level_data)
                    apply_choice(selection, key, selected_option, custom_value)
                    assert selection[key] == custom_value

                    title, final_story, summary = build_outputs(level_key, level_data, selection)
                    joined_output = "\n".join([title, final_story, "\n".join(summary)]).lower()
                    assert custom_value.lower() in joined_output

                    custom_override_cases += 1
                    level_custom_cases += 1
                except Exception as exc:  # pragma: no cover - script diagnostic
                    errors.append(
                        {
                            "level": level_key,
                            "field": key,
                            "selected_option": selected_option,
                            "custom_value": custom_value,
                            "case": "custom_overrides_dropdown",
                            "error": repr(exc),
                        }
                    )

                # Caso 2: campo libre vacío. Debe usarse la opción del desplegable.
                try:
                    selection = empty_selection(level_data)
                    apply_choice(selection, key, selected_option, "")
                    assert selection[key] == selected_option

                    title, final_story, summary = build_outputs(level_key, level_data, selection)
                    assert final_story.strip()
                    assert summary

                    dropdown_default_cases += 1
                    level_default_cases += 1
                except Exception as exc:  # pragma: no cover - script diagnostic
                    errors.append(
                        {
                            "level": level_key,
                            "field": key,
                            "selected_option": selected_option,
                            "custom_value": "",
                            "case": "empty_custom_uses_dropdown",
                            "error": repr(exc),
                        }
                    )

        stats.append((level_key, len(level_data["steps"]), level_custom_cases, level_default_cases))

    # PDF: se comprueba una opción libre por cada campo narrativo, porque generar PDF
    # para todas las opciones ralentiza mucho el chequeo y la lógica PDF es común.
    pdf_cases = 0
    for level_key in LEVEL_FILES:
        level_data = load_level(level_key)
        for step in level_data["steps"]:
            key = step["key"]
            selection = empty_selection(level_data)
            selection[key] = f"opcion_libre_pdf_{level_key}_{key}"
            title, final_story, summary = build_outputs(level_key, level_data, selection)
            try:
                pdf = story_to_pdf_bytes(title, "Alumno/a de prueba", LEVEL_LABELS[level_key], final_story, summary)
                assert isinstance(pdf, (bytes, bytearray)) and len(pdf) > 100
                pdf_cases += 1
            except Exception as exc:  # pragma: no cover - script diagnostic
                errors.append(
                    {
                        "level": level_key,
                        "field": key,
                        "selected_option": "",
                        "custom_value": selection[key],
                        "case": "pdf_generation",
                        "error": repr(exc),
                    }
                )

    print("Comprobación de prioridad entre desplegable y campo libre")
    print("=" * 64)
    for level_key, fields, custom_cases, default_cases in stats:
        print(f"{level_key}: {fields} campos | {custom_cases} casos con campo libre | {default_cases} casos con campo libre vacío")
    print("-" * 64)
    print(f"Total casos con campo libre que sustituye al desplegable: {custom_override_cases}")
    print(f"Total casos con campo libre vacío que usan el desplegable: {dropdown_default_cases}")
    print(f"Comprobaciones de PDF: {pdf_cases}")
    print(f"Errores: {len(errors)}")

    if errors:
        print("\nPrimeros errores detectados:")
        for error in errors[:20]:
            print(error)
        return 1

    print("\nResultado: OK. El campo libre tiene prioridad y no rompe la generación del cuento.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
