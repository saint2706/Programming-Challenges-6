"""One exception type for every failure the user can fix; the CLI prints it without a traceback."""


class InvoicerError(Exception):
    pass
