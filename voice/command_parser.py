NEW_SCENE_KEYWORDS = ("show", "create", "display", "what is", "make me")
REFINE_KEYWORDS = ("bigger", "add", "remove", "color", "faster", "zoom", "rotate")


def classify_command(text: str) -> tuple[str, str]:
    lowered = text.lower()

    for keyword in REFINE_KEYWORDS:
        if keyword in lowered:
            return ("REFINE", text.strip())

    for keyword in NEW_SCENE_KEYWORDS:
        if keyword in lowered:
            return ("NEW_SCENE", text.strip())

    return ("NEW_SCENE", text.strip())
