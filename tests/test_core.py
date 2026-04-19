"""Unit tests for local_llm_tuner.core."""

from local_llm_tuner.core import chunk_text, merge_update, extract_json


# ---------- chunk_text ---------- #

def test_chunk_text_small_fits_one_chunk():
    chunks = chunk_text("short text", chunk_size=100, overlap=10)
    assert chunks == ["short text"]


def test_chunk_text_splits_into_multiple_chunks():
    text = "word " * 200  # 1000 chars of whitespace-separated words
    chunks = chunk_text(text, chunk_size=200, overlap=20)
    assert len(chunks) > 1
    # Reassembled (with overlap handled) content should contain all original words
    reassembled = "".join(chunks)
    assert "word" in reassembled
    # The chunker snaps to word boundaries — no chunk should mid-split a word
    # like "wo|rd" (first ending with "wo" AND second starting with "rd").
    for i in range(len(chunks) - 1):
        tail = chunks[i][-3:]
        head = chunks[i + 1][:3]
        # A mid-word split would produce tail "wor" next head "d w" etc. Since
        # our test corpus only has full "word " tokens, either tail or head
        # must contain whitespace if the boundary is respected.
        assert " " in tail or " " in head, f"mid-word split: {tail!r}|{head!r}"


def test_chunk_text_overlap():
    text = "a " * 500
    chunks = chunk_text(text, chunk_size=200, overlap=20)
    # Adjacent chunks should share some prefix/suffix overlap
    for i in range(len(chunks) - 1):
        assert chunks[i + 1][:10] in chunks[i] or chunks[i][-30:].strip() in chunks[i + 1]


# ---------- merge_update ---------- #

def test_merge_update_empty_noop():
    mem = {"k": [1, 2]}
    merge_update(mem, {})
    assert mem == {"k": [1, 2]}


def test_merge_update_string_replaces():
    mem = {"title": "old"}
    merge_update(mem, {"title": "new"})
    assert mem["title"] == "new"


def test_merge_update_empty_string_preserves():
    mem = {"title": "old"}
    merge_update(mem, {"title": "   "})
    assert mem["title"] == "old"


def test_merge_update_list_extends_dedup():
    mem = {"items": ["one", "two"]}
    merge_update(mem, {"items": ["two", "three", "TWO"]})  # case-insensitive dedup
    assert mem["items"] == ["one", "two", "three"]


def test_merge_update_dict_shallow_merge():
    mem = {"ratings": {"a": 1, "b": 2}}
    merge_update(mem, {"ratings": {"b": 99, "c": 3}})
    assert mem["ratings"] == {"a": 1, "b": 99, "c": 3}


def test_merge_update_accepts_new_keys():
    mem = {}
    merge_update(mem, {"new_key": "hello", "other": [1, 2]})
    assert mem["new_key"] == "hello"
    assert mem["other"] == [1, 2]


def test_merge_update_rejects_garbage_keys():
    # Very long / structured-looking keys are rejected (defensive)
    mem = {}
    merge_update(mem, {"x" * 100: "should_not_land"})
    assert "x" * 100 not in mem


def test_merge_update_non_dict_update_is_noop():
    mem = {"k": 1}
    merge_update(mem, "not a dict")  # type: ignore
    assert mem == {"k": 1}


# ---------- extract_json ---------- #

def test_extract_json_clean():
    assert extract_json('{"a": 1}') == {"a": 1}


def test_extract_json_with_prose():
    assert extract_json('here is the json: {"a": 1} end') == {"a": 1}


def test_extract_json_fenced():
    assert extract_json('```json\n{"a": 1}\n```') == {"a": 1}


def test_extract_json_trailing_comma():
    assert extract_json('{"a": 1,}') == {"a": 1}


def test_extract_json_empty():
    assert extract_json("") == {}
    assert extract_json("   ") == {}


def test_extract_json_no_json():
    assert extract_json("plain text no braces") == {}
