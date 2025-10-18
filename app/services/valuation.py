from typing import Optional


MULTIPLIERS = {
    "minor": 0.9,
    "moderate": 0.7,
    "severe": 0.5,
}


def apply_damage_multiplier(base_price: Optional[float], severity: str) -> Optional[float]:
    if base_price is None:
        return None
    factor = MULTIPLIERS.get(severity, 0.9)
    return round(base_price * factor, 2)


