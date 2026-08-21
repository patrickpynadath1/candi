from pathlib import Path

import pytest

from scripts.new_paper_method import scaffold, valid_slug


TEMPLATE = Path(__file__).resolve().parents[2] / "papers/templates/method-plan.md"


def test_scaffold_records_source_and_refuses_overwrite(tmp_path):
    paper = tmp_path / "source.pdf"
    paper.write_bytes(b"%PDF-1.7\nsmall test fixture")
    output_root = tmp_path / "methods"

    record = scaffold(
        paper=paper,
        slug="example-method",
        title="Example Method",
        paper_id="arXiv:1234.5678",
        output_root=output_root,
        template_path=TEMPLATE,
    )

    contents = record.read_text()
    assert "# Example Method" in contents
    assert "arXiv:1234.5678" in contents
    assert "source.pdf" in contents
    assert "{{" not in contents
    with pytest.raises(FileExistsError):
        scaffold(
            paper, "example-method", "Example Method", "", output_root, TEMPLATE
        )


def test_scaffold_rejects_a_non_pdf(tmp_path):
    source = tmp_path / "source.pdf"
    source.write_text("not actually a PDF")

    with pytest.raises(ValueError, match="not a PDF"):
        scaffold(source, "example", "Example", "", tmp_path / "methods", TEMPLATE)


@pytest.mark.parametrize("slug", ["Uppercase", "two_words", "-leading", "trailing-"])
def test_slug_rejects_unstable_names(slug):
    with pytest.raises(Exception):
        valid_slug(slug)
