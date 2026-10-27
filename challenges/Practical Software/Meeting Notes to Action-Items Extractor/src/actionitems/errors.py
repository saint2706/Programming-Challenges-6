"""Exceptions that the CLI turns into short messages instead of tracebacks."""


class ActionItemsError(Exception):
    """Base class for expected, user-facing failures."""


class InputError(ActionItemsError):
    """An input file, stream or option is unusable (too big, not text, missing, unsafe)."""


class LLMError(ActionItemsError):
    """The optional local model endpoint failed or answered something unusable."""
