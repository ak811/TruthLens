"""Probe prompt set ``P = {p_1, ..., p_N}`` (paper Section 2.1, Appendix B).

The nine prompt texts below are copied verbatim from Appendix B of the paper.
The prompt for ``overall_realism`` is also byte-identical to the single query
hard-coded in the originally released ``inference_image_chatunivi.py``.

Category keys are stable identifiers used in all intermediate files.
"""
from __future__ import annotations

from collections import OrderedDict
from typing import Iterable, List, Sequence, Union

# (key, human-readable title as in Appendix B / Table 4, prompt text)
_PROBES = [
    ("lighting", "Lighting and Shadows",
     "Describe the lighting in the image. Does it appear natural or does it show any inconsistencies, "
     "such as unrealistic shadows or lighting direction?"),
    ("texture", "Texture and Skin Details",
     "Analyze the texture of the skin in this image. Does the skin appear to have natural imperfections "
     "like pores, wrinkles, or blemishes, or is it unnaturally smooth?"),
    ("symmetry", "Symmetry and Proportions",
     "Describe the facial symmetry in the image. Are there any noticeable asymmetries in the eyes, nose, "
     "mouth, or face shape?"),
    ("reflections", "Reflections and Highlights",
     "Examine the reflections in the eyes or any shiny areas on the skin. Do they appear to be consistent "
     "with the environment, or do they seem artificial or inconsistent?"),
    ("facial_features", "Facial Features and Expression",
     "Describe the facial expression in the image. Does it appear natural, or are there any signs of a "
     "forced or unnatural expression?"),
    ("facial_hair", "Facial Hair",
     "If there is facial hair in the image, describe its appearance. Does it seem realistic in terms of "
     "texture, growth pattern, and interaction with the lighting?"),
    ("eyes", "Eyes and Pupils",
     "Describe the appearance of the eyes in the image. Do the pupils appear natural in size, shape, and "
     "positioning, or are there any abnormalities?"),
    ("background", "Background and Depth Perception",
     "Describe the background of the image. Does it seem well-integrated with the face in terms of depth, "
     "focus, and lighting, or does it appear artificially blurred or detached?"),
    ("overall_realism", "Overall Realism of the Face",
     "Taking into account the lighting, texture, symmetry, and other features, describe the overall "
     "realism of the face. Does it show any signs of being digitally manipulated or generated?"),
]

PROMPTS: "OrderedDict[str, str]" = OrderedDict((k, p) for k, _, p in _PROBES)
CATEGORY_TITLES: "OrderedDict[str, str]" = OrderedDict((k, t) for k, t, _ in _PROBES)
ALL_CATEGORIES: List[str] = list(PROMPTS)

# The seven per-category files concatenated by the originally released
# ``concatinate_jsons.py`` (in that order). Lighting and background were NOT
# included there although the paper lists nine categories. The legacy file
# stems were: eyes, faceattributes, facialhair, realism_2, reflections,
# symmetry_2, texture. The mapping faceattributes -> facial_features is
# inferred from the file name and should be confirmed by the authors.
RELEASED_7: List[str] = ["eyes", "facial_features", "facial_hair", "overall_realism",
                         "reflections", "symmetry", "texture"]

CATEGORY_PRESETS = {"all": ALL_CATEGORIES, "released_7": RELEASED_7}

# "Yes or No Question" baseline of Table 2. The paper (Figure 3) shows only the
# wording "Are these images fake?"; the exact per-image wording and answer
# format are not specified, so this default is an assumption (configurable).
YES_NO_PROMPT = "Is this image fake? Answer with yes or no."


def resolve_categories(spec: Union[str, Sequence[str], None]) -> List[str]:
    """Resolve a category specification into an ordered list of category keys.

    ``spec`` may be ``None``/"all", a preset name ("released_7"), a
    comma-separated string, or a list of keys/presets.
    """
    if spec is None:
        return list(ALL_CATEGORIES)
    if isinstance(spec, str):
        if spec in CATEGORY_PRESETS:
            return list(CATEGORY_PRESETS[spec])
        items: Iterable[str] = [s.strip() for s in spec.split(",") if s.strip()]
    else:
        items = spec
    out: List[str] = []
    for item in items:
        expanded = CATEGORY_PRESETS.get(item, [item])
        for key in expanded:
            if key not in PROMPTS:
                raise ValueError(f"Unknown probe category '{key}'. Valid: {ALL_CATEGORIES} or presets "
                                 f"{list(CATEGORY_PRESETS)}")
            if key not in out:
                out.append(key)
    if not out:
        raise ValueError("Empty category specification.")
    return out
