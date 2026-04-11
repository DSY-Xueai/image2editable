from __future__ import annotations


DIRECT_EFFECTS = {"shadow", "stroke", "glow", "opacity", "gradient"}


def map_effect_type(effect: dict) -> dict:
    effect_type = effect.get("effect_type")
    confidence = float(effect.get("confidence", 0.0))
    complexity = effect.get("complexity", "simple")
    if effect_type in DIRECT_EFFECTS and confidence >= 0.9 and complexity == "simple":
        return {"mapped": True, "effect_type": effect_type}
    return {"mapped": False, "effect_type": effect_type}


def filter_mappable_effects(effects: list[dict]) -> list[dict]:
    return [effect for effect in effects if map_effect_type(effect)["mapped"]]
