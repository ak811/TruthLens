import json

from truthlens.aggregate import aggregate_answers, collect_answers


def released_concatenate(json_paths):
    """Logic of the released concatinate_jsons.py (without its file output)."""
    combined = {}
    for p in json_paths:
        with open(p) as f:
            for img, desc in json.load(f).items():
                combined[img] = combined[img] + " | " + desc if img in combined else desc
    return combined


def test_pipe_mode_reproduces_released_concatenation(tmp_path):
    cats = ["eyes", "facial_features", "texture"]
    answers = {"eyes": "Pupils dilated.", "facial_features": "Natural smile.", "texture": "Smooth skin."}
    paths = []
    for c in cats:
        p = tmp_path / f"{c}.json"
        p.write_text(json.dumps({"/x/img.png": answers[c]}))
        paths.append(p)
    expected = released_concatenate(paths)["/x/img.png"]
    summary, used = aggregate_answers(answers, cats, "pipe")
    assert summary == expected and used == cats


def test_structured_mode_labels_each_block_in_category_order():
    summary, used = aggregate_answers({"texture": " smooth ", "eyes": "odd pupils"}, ["eyes", "texture"])
    assert summary == "[Eyes and Pupils]\nodd pupils\n\n[Texture and Skin Details]\nsmooth"
    assert used == ["eyes", "texture"]


def test_missing_categories_are_skipped():
    summary, used = aggregate_answers({"eyes": "a"}, ["eyes", "texture"], "pipe")
    assert summary == "a" and used == ["eyes"]


def test_error_records_are_not_used_as_answers():
    recs = [{"id": "a", "category": "eyes", "response": None, "error": "OOM"},
            {"id": "a", "category": "texture", "response": "fine", "error": None},
            {"id": "a", "category": "texture", "response": "retry answer", "error": None}]
    assert collect_answers(recs) == {"a": {"texture": "retry answer"}}
