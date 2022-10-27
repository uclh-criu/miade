""" run the drug dosage extractor"""

from miade.dosageextractor import DosageExtractor
from argparse import ArgumentParser
from devtools import debug


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "-t",
        "--text",
        dest="text",
    )

    args = parser.parse_args()

    dosage_extractor = DosageExtractor()
    dosage = dosage_extractor.extract(args.text)
    print(dosage)
    debug(dosage.dose)
    debug(dosage.frequency)
    debug(dosage.route)
    debug(dosage.duration)

