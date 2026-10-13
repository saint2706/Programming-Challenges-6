import pytest
from helpers import make_pdf
from pdftoolkit.errors import PdfToolError
from pdftoolkit.files import expand_inputs, open_reader, plan_outputs, write_pdf
from pypdf import PdfWriter


def test_glob_is_sorted_naturally(tmp_path):
    for n in (10, 2, 1):
        make_pdf(tmp_path / f"scan{n}.pdf", 1)
    found = expand_inputs([str(tmp_path / "scan*.pdf")])
    assert [s.path.name for s in found] == ["scan1.pdf", "scan2.pdf", "scan10.pdf"]


def test_glob_without_match_is_an_error(tmp_path):
    with pytest.raises(PdfToolError, match="no files match"):
        expand_inputs([str(tmp_path / "*.pdf")])


def test_page_suffix_is_split_off(three):
    (src,) = expand_inputs([f"{three}[2-3]"])
    assert (src.path, src.pages) == (three, "2-3")


def test_real_file_with_brackets_wins(tmp_path):
    odd = make_pdf(tmp_path / "a[1].pdf", 1)
    (src,) = expand_inputs([str(odd)])
    assert (src.path, src.pages) == (odd, None)


def test_missing_input(tmp_path):
    with pytest.raises(PdfToolError, match="no such file"):
        expand_inputs([str(tmp_path / "nope.pdf")])


def test_relative_paths_use_base(tmp_path):
    make_pdf(tmp_path / "a.pdf", 1)
    assert expand_inputs(["a.pdf"], tmp_path)[0].path == tmp_path / "a.pdf"


def test_corrupt_file_reports_its_name(tmp_path):
    bad = tmp_path / "bad.pdf"
    bad.write_bytes(b"not a pdf")
    with pytest.raises(PdfToolError, match="bad.pdf"):
        open_reader(bad)


def test_write_refuses_overwrite_and_cleans_temp_files(tmp_path):
    writer = PdfWriter()
    writer.add_blank_page(100, 100)
    dst = tmp_path / "out.pdf"
    write_pdf(writer, dst)
    with pytest.raises(PdfToolError, match="already exists"):
        write_pdf(writer, dst)
    write_pdf(writer, dst, force=True)
    assert [p.name for p in tmp_path.iterdir()] == ["out.pdf"]


def test_plan_outputs_names_and_guards(tmp_path, three, five):
    sources = expand_inputs([str(three), str(five)])
    plan = plan_outputs(sources, output=None, out_dir=tmp_path / "o", suffix="_x")
    assert [dst.name for _, dst in plan] == ["three_x.pdf", "five_x.pdf"]
    with pytest.raises(PdfToolError, match="exactly one input"):
        plan_outputs(sources, output=tmp_path / "x.pdf", out_dir=None, suffix="_x")
    with pytest.raises(PdfToolError, match="also an input"):
        plan_outputs(sources[:1], output=three, out_dir=None, suffix="_x")
    with pytest.raises(PdfToolError, match="either"):
        plan_outputs(
            sources[:1], output=tmp_path / "a.pdf", out_dir=tmp_path, suffix="_x"
        )


def test_plan_detects_colliding_outputs(tmp_path):
    a = make_pdf(tmp_path / "x" / "same.pdf", 1)
    b = make_pdf(tmp_path / "y" / "same.pdf", 1)
    with pytest.raises(PdfToolError, match="two results"):
        plan_outputs(
            expand_inputs([str(a), str(b)]), output=None, out_dir=tmp_path, suffix="_x"
        )
