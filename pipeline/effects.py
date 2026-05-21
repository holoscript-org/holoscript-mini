EFFECT_TEMPLATES = {
	"anim_orbit": {"type": "orbit", "center": [0, 0, 0], "axis": [0, 1, 0], "speed": 0.5},
	"anim_spin": {"type": "spin", "axis": [0, 1, 0], "speed": 1.0},
}


def apply_effect(objects: list[dict], handler: str, center=None, speed=None) -> list[dict]:
	template = dict(EFFECT_TEMPLATES.get(handler, {"type": "none"}))
	if center:
		template["center"] = center
	if speed is not None:
		template["speed"] = speed
	for obj in objects:
		if obj.get("animation", {}).get("type", "none") == "none":
			obj["animation"] = dict(template)
	return objects
