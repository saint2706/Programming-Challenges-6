"""Prefix the file paths in a Cobertura coverage.xml with a directory.

coverage.py writes paths relative to where it ran (src/pkg/cli.py). Codecov matches paths against
the repository tree, where that path exists in many challenges, so CI turns them into
challenges/<Category>/<Challenge>/src/pkg/cli.py before uploading.

    python3 coverage_prefix.py <coverage.xml> <directory> <output.xml>

The input is the coverage.xml that coverage.py wrote seconds earlier in the same job, so it is as
trusted as the test run that produced it and the stdlib parser is enough (no defusedxml).
"""

import sys
import xml.etree.ElementTree as ET


def main(source: str, prefix: str, output: str) -> None:
    tree = ET.parse(source)
    for element in tree.iter("class"):
        element.set("filename", f"{prefix.rstrip('/')}/{element.get('filename')}")
    tree.write(output, encoding="utf-8", xml_declaration=True)


if __name__ == "__main__":
    if len(sys.argv) != 4:
        sys.exit(__doc__)
    main(*sys.argv[1:])
