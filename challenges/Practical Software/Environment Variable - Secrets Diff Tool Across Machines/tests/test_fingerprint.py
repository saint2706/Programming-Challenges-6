from envdiff.fingerprint import (
    Hasher,
    classify_shape,
    is_secret_name,
    looks_like_placeholder,
    matches_any,
)


def test_same_key_name_and_value_give_the_same_fingerprint():
    h = Hasher(b"k" * 32)
    assert h.fingerprints("A", "v") == h.fingerprints("A", "v")


def test_different_keys_give_unrelated_fingerprints():
    assert (
        Hasher(b"a" * 32).fingerprints("A", "v").exact
        != Hasher(b"b" * 32).fingerprints("A", "v").exact
    )


def test_random_hashers_differ_between_runs():
    assert (
        Hasher.random().fingerprints("A", "v").exact
        != Hasher.random().fingerprints("A", "v").exact
    )


def test_fingerprints_are_bound_to_the_variable_name():
    h = Hasher(b"k" * 32)
    assert h.fingerprints("A", "same").exact != h.fingerprints("B", "same").exact


def test_variants_collapse_exactly_the_differences_they_are_meant_to():
    h = Hasher(b"k" * 32)
    plain, padded, quoted, upper = (
        h.fingerprints("A", v) for v in ["abc", " abc ", '"abc"', "ABC"]
    )
    assert plain.exact != padded.exact and plain.stripped == padded.stripped
    assert plain.exact != quoted.exact and plain.unquoted == quoted.unquoted
    assert plain.exact != upper.exact and plain.folded == upper.folded
    assert plain.stripped != quoted.stripped  # quotes are not whitespace


def test_variant_tags_stop_a_variant_fingerprint_colliding_with_the_exact_one():
    fp = Hasher(b"k" * 32).fingerprints("A", "abc")
    assert len(set(fp.as_list())) == 4


def test_passphrase_derivation_is_deterministic_and_passphrase_sensitive():
    a, b = Hasher.from_passphrase("one"), Hasher.from_passphrase("one")
    assert a.key_id == b.key_id and a.fingerprints("A", "v") == b.fingerprints("A", "v")
    assert Hasher.from_passphrase("two").key_id != a.key_id


def test_key_id_does_not_expose_the_key():
    h = Hasher(b"k" * 32)
    assert b"k" * 8 not in h.key_id.encode() and len(h.key_id) == 12


def test_secret_name_detection():
    for name in ["SECRET_KEY", "db_password", "STRIPE_API_KEY", "AWS_ACCESS_KEY_ID", "JWT_SIGNING_KEY", "SENTRY_DSN",
                 "GITHUB_TOKEN", "KEY", "SSH_PRIVATE_KEY", "SESSION_SALT"]:  # fmt: skip
        assert is_secret_name(name), name
    for name in [
        "PORT",
        "APP_ENV",
        "DEBUG",
        "DATABASE_HOST",
        "LOG_LEVEL",
        "KEYBOARD_LAYOUT",
    ]:
        assert not is_secret_name(name), name


def test_placeholder_detection():
    for value in [
        "changeme",
        "CHANGE_ME",
        "your-api-key-here",
        "<token>",
        "xxxxxxxx",
        "TODO",
        "${SECRET}",
        "password",
    ]:
        assert looks_like_placeholder(value), value
    for value in ["", "   ", "sk_live_abc123", "production", "8080"]:
        assert not looks_like_placeholder(value), value


def test_shape_classification_is_coarse():
    cases = {
        "": "empty", "true": "bool", "On": "bool", "8080": "integer", "-3": "integer", "1.5": "number",
        "123e4567-e89b-12d3-a456-426614174000": "uuid", "https://x.example/y": "url",
        "postgres://u:p@h/db": "url", "ops@example.com": "email", "/var/lib/app": "path", "./rel": "path",
        "deadbeefdeadbeef": "hex", "abcdefghijklmnopqrstuvwx": "token", "hello world": "text",
        "eyJhbGciOi.eyJzdWIi.sig": "jwt",
    }  # fmt: skip
    for value, shape in cases.items():
        assert classify_shape(value) == shape, value


def test_glob_matching():
    assert matches_any("AWS_REGION", ["AWS_*"]) and matches_any("PATH", ["PATH"])
    assert not matches_any("aws_region", ["AWS_*"])  # case-sensitive, like env names
    assert not matches_any("X", [])
