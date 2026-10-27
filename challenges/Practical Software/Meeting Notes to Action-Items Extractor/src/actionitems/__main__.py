"""``python -m actionitems`` runs the same entry point as the ``actionitems`` script."""

from actionitems.cli import main

if __name__ == "__main__":
    raise SystemExit(main())
