import re
import logging

from typing import List, Optional, Dict

from .paragraph import Paragraph, ParagraphType


log = logging.getLogger(__name__)


class Note(object):
    """
    Represents a note object.

    Attributes:
        text (str): The text content of the note.
        raw_text (str): The raw text content of the note.
        regex_config (str): The path to the regex configuration file.
        paragraphs (Optional[List[Paragraph]]): A list of paragraphs in the note.
    """

    # TODO: refactor paragraph methods to a separate class. It's too much.

    def __init__(self, text: str):
        self.text = text
        self.raw_text = text
        self.paragraphs: Optional[List[Paragraph]] = []
        self.prose_paragraphs: Optional[List[Paragraph]] = []
        self.numbered_list: List[tuple] = []

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
                results.append((match.start(1), match.end(1), number_text))
            else:
                # If there is a break in the sequence, check if current list has more than one item
                if len(results) > 1:
                    all_results.append(results)
                results = [(match.start(1), match.end(1), number_text)]  # Start new results list with the current match
            last_num = current_num  # Update last number to the current

        # Add the last sequence if not empty and has more than one item
        if len(results) > 1:
            all_results.append(results)
        
        self.numbered_list = all_results

    def get_paragraphs(self, paragraph_regex: Dict) -> None:
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

    def merge_prose_sections(self):
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
                heading=self.text[start:end], 
                body="", 
                type=ParagraphType.prose, 
                start=start, 
                end=end)
            
            # Replace the first paragraph in the section with the new merged paragraph
            first_idx = section[0][0]
            new_paragraphs[first_idx] = new_prose_para
            
            # Mark other paragraphs in the section for deletion
            for _, paragraph in section[1:]:
                index = self.paragraphs.index(paragraph)
                new_paragraphs[index] = None
        
        # Remove the None entries from new_paragraphs
        self.paragraphs = [para for para in new_paragraphs if para is not None]
        
        return all_prose

    def is_in_numbered_list(self) -> bool:
        pass

    def refine_paragraghs(self):
        # do your whole manual check thing here etc.
        pass

    def process(self):
        # so this would be the call method for the refactored ParagraphSegmenter
        pass

    def __str__(self):
        return self.text
