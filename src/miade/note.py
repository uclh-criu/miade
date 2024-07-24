import re
import io
import pkgutil
import logging
import pandas as pd

from typing import List, Optional, Dict

from .paragraph import Paragraph, ParagraphType


log = logging.getLogger(__name__)


def load_regex_config_mappings(filename: str) -> Dict:
    """
    Load regex configuration mappings from a file.

    Args:
        filename (str): The name of the file containing the regex configuration.

    Returns:
        A dictionary mapping paragraph types to their corresponding regex patterns.
    """
    regex_config = pkgutil.get_data(__name__, filename)
    data = (
        pd.read_csv(
            io.BytesIO(regex_config),
            index_col=0,
        )
        .squeeze("columns")
        .T.to_dict()
    )
    regex_lookup = {}

    for paragraph, regex in data.items():
        paragraph_enum = None
        try:
            paragraph_enum = ParagraphType(paragraph)
        except ValueError as e:
            log.warning(e)

        if paragraph_enum is not None:
            regex_lookup[paragraph_enum] = regex

    return regex_lookup


class Note(object):
    """
    Represents a note object.

    Attributes:
        text (str): The text content of the note.
        raw_text (str): The raw text content of the note.
        regex_config (str): The path to the regex configuration file.
        paragraphs (Optional[List[Paragraph]]): A list of paragraphs in the note.
    """

    def __init__(self, text: str, regex_config_path: str = "./data/regex_para_chunk.csv"):
        self.text = text
        self.raw_text = text
        self.regex_config = load_regex_config_mappings(regex_config_path)
        self.paragraphs: Optional[List[Paragraph]] = []

    def clean_text(self) -> None:
        """
        Cleans the text content of the note.

        This method performs various cleaning operations on the text content of the note,
        such as replacing spaces, removing punctuation, and removing empty lines.
        """

        # Replace all types of spaces with a single normal space, preserving "\n"
        self.text = re.sub(r"(?:(?!\n)\s)+", " ", self.text)

        # Remove en dashes that are not between two numbers
        self.text = re.sub(r"(?<![0-9])-(?![0-9])", "", self.text)

        # Remove all punctuation except full stops, question marks, dash and line breaks
        self.text = re.sub(r"[^\w\s.,?\n-]", "", self.text)

        # Remove spaces if the entire line (between two line breaks) is just spaces
        self.text = re.sub(r"(?<=\n)\s+(?=\n)", "", self.text)

    def get_paragraphs(self) -> None:
        """
        Splits the note into paragraphs.

        This method splits the text content of the note into paragraphs based on double line breaks.
        It also assigns a paragraph type to each paragraph based on matching patterns in the heading.
        """

        paragraphs = re.split(r"\n\n+", self.text)
        start = 0

        for text in paragraphs:
            # Default to prose
            paragraph_type = ParagraphType.prose

            # Use re.search to find everything before first \n
            match = re.search(r"^(.*?)(?:\n|$)([\s\S]*)", text)

            # Check if a match is found
            if match:
                heading = match.group(1)
                body = match.group(2)
            else:
                heading = text
                body = ""

            end = start + len(text)
            paragraph = Paragraph(heading=heading, body=body, type=paragraph_type, start=start, end=end)
            start = end + 2  # Account for the two newline characters

            # Convert the heading to lowercase for case-insensitive matching
            if heading:
                heading = heading.lower()
                # Iterate through the dictionary items and patterns
                for paragraph_type, pattern in self.regex_config.items():
                    if re.search(pattern, heading):
                        paragraph.type = paragraph_type
                        break  # Exit the loop if a match is found

            self.paragraphs.append(paragraph)

    def __str__(self):
        return self.text
