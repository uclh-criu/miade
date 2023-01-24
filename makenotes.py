"""
For extracting the note section from a CDA
"""


import typer

from typing import List
from pathlib import Path
from bs4 import BeautifulSoup
from bs4 import Tag


def display_children(tag, depth=1):
    indent = "\n" + " " * depth
    s = tag.name + "".join([
        indent + display_children(child, depth=depth + 1)
        for child in tag.children
        if ((type(child) == Tag) or (type(child) == BeautifulSoup))
    ])
    return s


def find_note(filepath):
    with open(filepath) as file:
        s = BeautifulSoup(file, 'xml')
        notes = [tag.parent.find("text") for tag in
                 s.find_all(lambda tag: tag.get('root') == "1.2.840.114350.1.72.1.200001")]
        return notes[0]


def convert(input_paths: List[Path], output_folder: Path):
    for filepath in input_paths:
        newfile = output_folder / filepath.name
        with open(newfile, "w") as file:
            file.write(str(find_note(filepath)))


if __name__ == "__main__":
    typer.run(convert)
