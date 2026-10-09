from readme_generator.document import has_blocks, merge_document, render_document, wrap
from readme_generator.render import Section


def S(id_, text="body"):
    return Section(id_, f"## {id_}\n\n{text}")


def test_render_wraps_each_section_in_markers():
    out = render_document([S("header"), S("stack")])
    assert out.startswith("<!-- readme-gen:begin header -->\n## header")
    assert out.endswith("<!-- readme-gen:end stack -->\n")
    assert has_blocks(out) and not has_blocks("# plain readme")


def test_merge_updates_in_place_and_leaves_hand_written_prose_untouched():
    existing = (
        "Intro I wrote.\n\n" + wrap(S("stack", "old")) + "\n\n## My notes\n\nKeep me.\n"
    )
    merged, report = merge_document(existing, [S("stack", "new")])
    assert (
        merged
        == "Intro I wrote.\n\n"
        + wrap(S("stack", "new"))
        + "\n\n## My notes\n\nKeep me.\n"
    )
    assert report.updated == ["stack"] and not report.added and not report.removed


def test_merge_is_a_no_op_when_nothing_changed():
    existing = render_document([S("header"), S("stack")])
    merged, report = merge_document(existing, [S("header"), S("stack")])
    assert merged == existing
    assert report == type(report)()


def test_merge_removes_stale_blocks_and_their_blank_lines():
    existing = render_document([S("header"), S("docker"), S("license")])
    merged, report = merge_document(existing, [S("header"), S("license")])
    assert merged == render_document([S("header"), S("license")])
    assert report.removed == ["docker"]


def test_new_sections_land_after_the_nearest_earlier_section():
    existing = render_document([S("header"), S("license")])
    merged, report = merge_document(
        existing, [S("header"), S("stack"), S("docker"), S("license")]
    )
    assert merged == render_document(
        [S("header"), S("stack"), S("docker"), S("license")]
    )
    assert report.added == ["stack", "docker"]


def test_new_first_section_goes_before_existing_later_ones_and_empty_files_get_appended():
    merged, _ = merge_document(render_document([S("stack")]), [S("header"), S("stack")])
    assert merged == render_document([S("header"), S("stack")])
    merged, _ = merge_document("only prose\n", [S("header")])
    assert merged == "only prose\n\n" + wrap(S("header")) + "\n"


def test_unknown_block_ids_in_the_file_are_treated_as_stale():
    existing = wrap(S("header")) + "\n\n" + wrap(Section("custom", "x"))
    merged, report = merge_document(existing, [S("header")])
    assert "custom" not in merged and report.removed == ["custom"]
