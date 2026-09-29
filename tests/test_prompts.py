import pytest

from truthlens.prompts import ALL_CATEGORIES, PROMPTS, RELEASED_7, resolve_categories

RELEASED_QUERY = ("Taking into account the lighting, texture, symmetry, and other features, describe the overall "
                  "realism of the face. Does it show any signs of being digitally manipulated or generated?")


def test_nine_probe_categories_in_appendix_b_order():
    assert ALL_CATEGORIES == ["lighting", "texture", "symmetry", "reflections", "facial_features",
                              "facial_hair", "eyes", "background", "overall_realism"]


def test_overall_realism_matches_released_script_exactly():
    assert PROMPTS["overall_realism"] == RELEASED_QUERY


def test_prompts_are_single_spaced_questions():
    for key, text in PROMPTS.items():
        assert "  " not in text and text.endswith("?"), key


def test_released_7_excludes_lighting_and_background():
    assert set(ALL_CATEGORIES) - set(RELEASED_7) == {"lighting", "background"}


@pytest.mark.parametrize("spec,expected", [
    (None, ALL_CATEGORIES), ("all", ALL_CATEGORIES), ("released_7", RELEASED_7),
    ("eyes,texture", ["eyes", "texture"]), (["eyes", "eyes"], ["eyes"]),
])
def test_resolve_categories(spec, expected):
    assert resolve_categories(spec) == expected


def test_resolve_categories_rejects_unknown():
    with pytest.raises(ValueError):
        resolve_categories("eyez")
