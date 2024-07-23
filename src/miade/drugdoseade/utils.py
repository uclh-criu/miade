import re
import logging

from typing import Dict, List


log = logging.getLogger(__name__)


def word_replace(word: str, dictionary: Dict[str, str], processed_text: List[str]) -> List[str]:
    """
    Replaces words with entries from CALIBERdrugdose singleword dict

    Args:
        word (str): The word to be replaced.
        dictionary (Dict[str, str]): A dictionary containing word replacements.
        processed_text (List[str]): A list to store the processed text.

    Returns:
        The processed text with word replacements.

    """
    replacement = dictionary.get(word, None)
    if isinstance(replacement, str):
        # replace with dict entry
        processed_text.append(replacement)
        log.debug(f"Replaced word '{word}' with '{replacement}'")
    elif replacement is None and not word.replace(".", "", 1).isdigit():
        # 25mg to 25 mg
        word = re.sub(r"(\d+)([a-z]+)", r"\1 \2", word)
        # split further if e.g. 2-5
        for subword in re.findall(r"[\w']+|[.,!?;*&@>#/-]", word):
            replacement = dictionary.get(subword, None)
            if isinstance(replacement, str):
                processed_text.append(replacement)
                log.debug(f"Replaced word '{subword}' with '{replacement}'")
            elif replacement is None and not subword.replace(".", "", 1).isdigit():
                log.debug(f"Removed word '{subword}', not in singleword dict")
            else:
                processed_text.append(subword)
    else:
        # else the lookup returned Nan which means no change
        processed_text.append(word)

    return processed_text


def numbers_replace(text) -> str:
    """
    Replaces numbers and units in the given text according to specific patterns.

    Args:
        text (str): The input text to be processed.

    Returns:
        The processed text with numbers and units replaced.

    """
    # 10 ml etc
    text = re.sub(
        r" (\d+) o (ml|microgram|mcg|gram|mg) ",
        lambda m: " {:g} {} ".format(float(m.group(1)) * 10, m.group(2)),
        text,
    )
    # 1/2
    text = re.sub(r" 1 / 2 ", r" 0.5 ", text)
    # 1.5 times 2 ... (not used for 5ml doses, because this is treated as a separate dose units)
    if not re.search(r" ([\d.]+) (times|x) (\d+) 5 ml ", text):
        text = re.sub(
            r" ([\d.]+) (times|x) (\d+) ",
            lambda m: " {:g} ".format(int(m.group(1)) * int(m.group(3))),
            text,
        )

    # 1 mg x 2 ... (but not 1 mg x 5 days)
    if not re.search(
        r" ([\d.]+) (ml|mg|gram|mcg|microgram|unit) (times|x) (\d+) (days|month|week) ",
        text,
    ):
        text = re.sub(
            r" ([\d.]+) (ml|mg|gram|mcg|microgram|unit) (times|x) (\d+) ",
            lambda m: " {:g} {} ".format(int(m.group(1)) * int(m.group(4)), m.group(2)),
            text,
        )

    # 1 drop or 2...
    split_text = re.sub(
        r"^[\w\s]*([\d.]+) (tab|drops|cap|ml|puff|fiveml) (to|-|star) ([\d.]+)[\w\s]*$",
        r"MATCHED \1 \4",
        text,
    ).split(" ")
    if split_text[0] == "MATCHED":
        # check that upper dose limit is greater than lower, otherwise
        # the text may not actually represent a dose range
        if float(split_text[2]) > float(split_text[1]):
            text = re.sub(
                r" ([\d.]+) (tab|drops|cap|ml|puff|fiveml) (to|-|star) ([\d.]+) ",
                r" \1 \2 or \4 ",
                text,
            )
        else:
            # not a choice, two pieces of information (e.g. '25mg - 2 daily')
            text = re.sub(
                r" ([\d.]+) (tab|drops|cap|ml|puff|fiveml) (to|-|star) ([\d.]+) ",
                r" \1 \2 \4 ",
                text,
            )
    # 1 and 2...
    text = re.sub(
        r" ([\d.]+) (and|\\+) ([\d.]+) ",
        lambda m: " {:g} ".format(int(m.group(1)) + int(m.group(3))),
        text,
    )
    # 3 weeks...
    text = re.sub(r" ([\d.]+) (week) ", lambda m: " {:g} days ".format(int(m.group(1)) * 7), text)
    # 3 months ... NB assume 30 days in a month
    text = re.sub(
        r" ([\d.]+) (month) ",
        lambda m: " {:g} days ".format(int(m.group(1)) * 30),
        text,
    )
    # day 1 to day 14 ...
    text = re.sub(
        r" days (\d+) (to|-) day (\d+) ",
        lambda m: " for {:g} days ".format(int(m.group(3)) - int(m.group(1))),
        text,
    )
    # X times day to X times day
    # TODO: frequency ranges
    text = re.sub(
        r" (\d+) (times|x) day (to|or|-|upto|star) (\d+) (times|x) day ",
        lambda m: " every {:g} hours +-{:g} ".format(
            (24 / int(m.group(4)) + 24 / int(m.group(1))) / 2,
            24 / int(m.group(1)) - (24 / int(m.group(4)) + 24 / int(m.group(1))) / 2,
        ),
        text,
    )

    # days 1 to 14 ...
    text = re.sub(
        r" days (\d+) (to|-) (\d+) ",
        lambda m: " for {:g} days ".format(int(m.group(3)) - int(m.group(1))),
        text,
    )

    # 1 or 2 ...
    text = re.sub(
        r" ([\d.]+) (to|or|-|star) ([\d.]+) (tab|drops|cap|ml|puff|fiveml) ",
        r" \1 \4 \2 \3 \4 ",
        text,
    )

    # X times or X times ...deleted as want to have range
    # x days every x days
    text = re.sub(
        r" (for )*([\d\\.]+) days every ([\d\\.]+) days ",
        lambda m: " for {} days changeto 0 0 times day for {:g} days ".format(
            m.group(2), int(m.group(3)) - int(m.group(2))
        ),
        text,
    )

    return text
