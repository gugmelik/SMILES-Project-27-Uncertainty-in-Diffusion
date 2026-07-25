"""ImageNet class difficulty tiers used across Experiment 1 / geometric runs.

Tiers were assigned by per-class KID (clean-fid) against the pretrained
DiT-XL/2 samples: the 15 highest-KID classes are "hardest", the 15 lowest are
"easiest", and a middle band of 15 is "medium". IDs are read from
`images_classes/kid_difficulty_tiers.png`.

`KID_APPROX` values are digitised off that bar chart and are therefore
*approximate* (±~0.002). They are good enough for a difficulty axis / rank
correlation; if you have the exact per-class KID numbers, drop them into a CSV
and load those instead.
"""

# ordered hardest -> easiest within each tier (as plotted)
HARDEST = [63, 38, 34, 47, 51, 33, 979, 61, 21, 44, 36, 35, 67, 29, 65]
MEDIUM  = [10, 9, 91, 71, 49, 16, 57, 360, 974, 41, 83, 48, 26, 94, 69]
EASIEST = [15, 92, 19, 14, 37, 85, 25, 95, 13, 82, 88, 24, 22, 72, 90]

TIER_ORDER = ["hardest", "medium", "easiest"]
TIER_CLASSES = {"hardest": HARDEST, "medium": MEDIUM, "easiest": EASIEST}
ALL_CLASSES = HARDEST + MEDIUM + EASIEST

# approximate KID (clean-fid) read off images_classes/kid_difficulty_tiers.png
KID_APPROX = {
    63: 0.0875, 38: 0.0845, 34: 0.0720, 47: 0.0660, 51: 0.0615,
    33: 0.0610, 979: 0.0608, 61: 0.0598, 21: 0.0570, 44: 0.0560,
    36: 0.0538, 35: 0.0537, 67: 0.0534, 29: 0.0527, 65: 0.0522,
    10: 0.0312, 9: 0.0308, 91: 0.0305, 71: 0.0301, 49: 0.0298,
    16: 0.0292, 57: 0.0270, 360: 0.0269, 974: 0.0263, 41: 0.0257,
    83: 0.0253, 48: 0.0251, 26: 0.0248, 94: 0.0244, 69: 0.0232,
    15: 0.0097, 92: 0.0096, 19: 0.0084, 14: 0.0078, 37: 0.0075,
    85: 0.0070, 25: 0.0069, 95: 0.0069, 13: 0.0067, 82: 0.0061,
    88: 0.0058, 24: 0.0052, 22: 0.0051, 72: 0.0038, 90: 0.0017,
}

TIER_COLORS = {"hardest": "#c8324b", "medium": "#d9a125", "easiest": "#2e9153"}

_LOOKUP = {c: t for t, cs in TIER_CLASSES.items() for c in cs}


def tier_of(class_id):
    return _LOOKUP.get(int(class_id))


def kid_of(class_id):
    return KID_APPROX.get(int(class_id))


def sample_per_tier(n, seed=0):
    """Return a class list with `n` classes from each tier (deterministic)."""
    import random
    rng = random.Random(seed)
    out = []
    for t in TIER_ORDER:
        pool = list(TIER_CLASSES[t])
        out += sorted(rng.sample(pool, min(n, len(pool))))
    return out