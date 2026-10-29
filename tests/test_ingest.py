"""Unit tests for ingestion planning and hashing."""

from app.ingest.ingest import hash_file, plan_ingestion


def test_hash_file_is_deterministic_and_sha256():
    assert hash_file(b"abc") == hash_file(b"abc")
    assert hash_file(b"abc") != hash_file(b"abd")
    assert len(hash_file(b"x")) == 64


def test_plan_ingestion_detects_all_transitions():
    existing = {"a.md": "h1", "b.md": "h2", "c.md": "h3"}
    current = {"a.md": "h1", "b.md": "changed", "d.md": "h4"}

    to_add, to_update, to_delete = plan_ingestion(existing, current)

    assert to_add == {"d.md"}
    assert to_update == {"b.md"}
    assert to_delete == {"c.md"}
    # a.md is unchanged, so it appears in none of the three sets.
    assert "a.md" not in to_add | to_update | to_delete


def test_plan_ingestion_empty_inputs():
    assert plan_ingestion({}, {}) == (set(), set(), set())


def test_plan_ingestion_first_run_adds_everything():
    current = {"a.md": "h1", "b.md": "h2"}
    assert plan_ingestion({}, current) == ({"a.md", "b.md"}, set(), set())


def test_plan_ingestion_removes_deleted_files():
    existing = {"a.md": "h1", "b.md": "h2"}
    assert plan_ingestion(existing, {}) == (set(), set(), {"a.md", "b.md"})
