import re
import logging

from typing import List, Optional, Dict

from .paragraph import ListItem, NumberedList, Paragraph, ParagraphType


log = logging.getLogger(__name__)


class Note(object):
    """
    Represents a Note object

    Attributes:
        text (str): The text content of the note.
        raw_text (str): The raw text content of the note.
        paragraphs (Optional[List[Paragraph]]): A list of Paragraph objects representing the paragraphs in the note.
        numbered_lists (Optional[List[NumberedList]]): A list of NumberedList objects representing the numbered lists in the note.
    """

    def __init__(self, text: str):
        self.text = text
        self.raw_text = text
        self.paragraphs: Optional[List[Paragraph]] = []
        self.numbered_lists: Optional[List[NumberedList]] = []

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

    def get_numbered_lists(self):
        """
        Finds multiple lists of sequentially ordered numbers (with more than one item) that directly follow a newline character
        and captures the text following these numbers up to the next newline.

        Parameters:
            text (str): The input text in which to search for multiple lists of sequentially ordered numbers with more than one item and their subsequent text.

        Returns:
            list of lists: Each sublist contains tuples where each tuple includes the start index of the number,
            the end index of the line, and the captured text for each valid sequentially ordered list found. Returns an empty list if no such patterns are found.
        """
        # Regular expression to find numbers followed by any characters until a newline
        pattern = re.compile(r"(?<=\n)(\d+.*)")

        # Finding all matches
        matches = pattern.finditer(self.text)

        all_results = []
        results = []
        last_num = 0
        for match in matches:
            number_text = match.group(1)
            current_num = int(re.search(r"^\d+", number_text).group())

            # Check if current number is the next in sequence
            if current_num == last_num + 1:
                results.append(ListItem(content=number_text, start=match.start(1), end=match.end(1)))
            else:
                # If there is a break in the sequence, check if current list has more than one item
                if len(results) > 1:
                    numbered_list = NumberedList(items=results, list_start=results[0].start, list_end=results[-1].end)
                    all_results.append(numbered_list)
                results = [
                    ListItem(content=number_text, start=match.start(1), end=match.end(1))
                ]  # Start new results list with the current match
            last_num = current_num  # Update last number to the current

        # Add the last sequence if not empty and has more than one item
        if len(results) > 1:
            numbered_list = NumberedList(items=results, list_start=results[0].start, list_end=results[-1].end)
            all_results.append(numbered_list)

        self.numbered_lists = all_results

    def get_paragraphs(self, paragraph_regex: Dict) -> None:
        """
        Split the text into paragraphs and assign paragraph types based on regex patterns.

        Args:
            paragraph_regex (Dict): A dictionary containing paragraph types as keys and regex patterns as values.

        Returns:
            None
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
                for paragraph_type, pattern in paragraph_regex.items():
                    if re.search(pattern, heading):
                        paragraph.type = paragraph_type
                        break  # Exit the loop if a match is found

            self.paragraphs.append(paragraph)

    def merge_prose_sections(self) -> None:
        """
        Merges consecutive prose sections in the paragraphs list.

        Returns:
            A list of merged prose sections.
        """
        is_merge = False
        all_prose = []
        prose_section = []
        prose_indices = []

        for i, paragraph in enumerate(self.paragraphs):
            if paragraph.type == ParagraphType.prose:
                if is_merge:
                    prose_section.append((i, paragraph))
                else:
                    prose_section = [(i, paragraph)]
                    is_merge = True
            else:
                if len(prose_section) > 0:
                    all_prose.append(prose_section)
                    prose_indices.extend([idx for idx, _ in prose_section])
                is_merge = False

        if len(prose_section) > 0:
            all_prose.append(prose_section)
            prose_indices.extend([idx for idx, _ in prose_section])

        new_paragraphs = self.paragraphs[:]

        for section in all_prose:
            start = section[0][1].start
            end = section[-1][1].end
            new_prose_para = Paragraph(
                heading=self.text[start:end], body="", type=ParagraphType.prose, start=start, end=end
            )

            # Replace the first paragraph in the section with the new merged paragraph
            first_idx = section[0][0]
            new_paragraphs[first_idx] = new_prose_para

            # Mark other paragraphs in the section for deletion
            for _, paragraph in section[1:]:
                index = self.paragraphs.index(paragraph)
                new_paragraphs[index] = None

        # Remove the None entries from new_paragraphs
        self.paragraphs = [para for para in new_paragraphs if para is not None]

    def merge_empty_non_prose_with_next_prose(self) -> None:
        """
        This method checks if a Paragraph has an empty body and a type that is not prose,
        and merges it with the next Paragraph if the next paragraph is type prose.

        Returns:
            None
        """
        merged_paragraphs = []
        skip_next = False

        for i in range(len(self.paragraphs) - 1):
            if skip_next:
                # Skip this iteration because the previous iteration already handled merging
                skip_next = False
                continue

            current_paragraph = self.paragraphs[i]
            next_paragraph = self.paragraphs[i + 1]

            # Check if current paragraph has an empty body and is not of type prose
            if current_paragraph.body == "" and current_paragraph.type != ParagraphType.prose:
                # Check if the next paragraph is of type prose
                if next_paragraph.type == ParagraphType.prose:
                    # Create a new Paragraph with merged content and type prose
                    merged_paragraph = Paragraph(
                        heading=current_paragraph.heading,
                        body=next_paragraph.heading,
                        type=current_paragraph.type,
                        start=current_paragraph.start,
                        end=next_paragraph.end,
                    )
                    merged_paragraphs.append(merged_paragraph)
                    # Skip the next paragraph since it's already merged
                    skip_next = True
                    continue

            # If no merging is done, add the current paragraph to the list
            merged_paragraphs.append(current_paragraph)

        # Handle the last paragraph if it wasn't merged
        if not skip_next:
            merged_paragraphs.append(self.paragraphs[-1])

        # Update the paragraphs list with the merged paragraphs
        self.paragraphs = merged_paragraphs

    def process(self, lookup_dict: Dict, refine: bool = True):
        """
        Process the note by cleaning the text, extracting numbered lists, and getting paragraphs based on a lookup dictionary.

        Args:
            lookup_dict (Dict): A dictionary used to lookup specific paragraphs.
            refine (bool, optional): Flag indicating whether to refine the processed note - this will merge any consecutive prose
            paragraphs and then merge any structured paragraphs with empty body with the next prose paragraph (handles line break
            between heading and body). Defaults to True.
        """
        self.clean_text()
        self.get_numbered_lists()
        self.get_paragraphs(lookup_dict)
        if refine:
            self.merge_prose_sections()
            self.merge_empty_non_prose_with_next_prose()

    def __str__(self):
        return self.text
