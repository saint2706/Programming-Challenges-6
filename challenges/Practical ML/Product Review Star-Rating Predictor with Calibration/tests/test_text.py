import pytest
from review_stars import text


@pytest.mark.parametrize(
    "title",
    [
        "Five Stars",
        "five stars",
        "5 stars",
        "5 Stars!",
        "One Star",
        "four-stars",
        "Two Stars.",
        " Three  stars ",
    ],
)
def test_auto_generated_star_titles_are_recognised(title):
    assert text.is_auto_title(title)


@pytest.mark.parametrize(
    "title",
    [
        "Perfect fit",
        "Works great",
        "Five Stars and a great price",
        "Stars",
        "6 stars",
        "",
        None,
        "Star wars fan",
    ],
)
def test_real_titles_are_not_mistaken_for_auto_titles(title):
    assert not text.is_auto_title(title)


def test_title_text_mode_blanks_only_the_auto_titles():
    assert (
        text.build_text("Perfect fit", "Fits my fridge.")
        == "Perfect fit. Fits my fridge."
    )
    assert text.build_text("Five Stars", "Fits my fridge.") == "Fits my fridge."


def test_text_only_mode_ignores_every_title():
    assert text.build_text("Perfect fit", "Fits.", "text_only") == "Fits."


def test_raw_title_mode_keeps_the_leaky_auto_title():
    assert text.build_text("Five Stars", "Fits.", "raw_title") == "Five Stars. Fits."


def test_a_title_only_review_is_just_the_title_with_no_stray_punctuation():
    assert text.build_text("Works great", "") == "Works great"
    assert text.build_text("Works great", None) == "Works great"
    assert (
        text.build_text("Five Stars", "") == ""
    )  # an auto title alone carries nothing


def test_empty_everything_is_the_empty_string_not_an_error():
    for mode in text.TEXT_MODES:
        assert text.build_text(None, None, mode) == ""
        assert text.build_text("", "   ", mode) == ""


def test_whitespace_and_html_line_breaks_are_collapsed():
    out = text.build_text("", "Line one<br />line two\n\n  three<br>four")
    assert out == "Line one line two three four"


def test_one_word_emoji_and_non_english_reviews_pass_through_unchanged():
    assert text.build_text("", "Great") == "Great"
    assert text.build_text("", "\U0001f44d\U0001f44d") == "\U0001f44d\U0001f44d"
    assert text.build_text("Très bien", "Parfait, merci") == "Très bien. Parfait, merci"


def test_unknown_mode_is_an_error():
    with pytest.raises(ValueError, match="text_mode"):
        text.build_text("a", "b", "nope")


def test_dedup_key_ignores_case_whitespace_and_unicode_form_but_not_content():
    k = text.dedup_key
    assert k("Great", "Works  fine") == k("great", "works fine")
    assert k("Caf\u00e9", "ok") == k("Cafe\u0301", "ok")
    assert k("Great", "Works fine") != k("Great", "Works fine!")
    assert k("a", "b") != k("b", "a")  # the title/text boundary matters


def test_dedup_key_is_a_stable_non_negative_64_bit_integer():
    v = text.dedup_key("Great", "Works fine")
    assert isinstance(v, int) and 0 <= v < 2**63
    assert v == text.dedup_key("Great", "Works fine")
