import re
import argparse


def convert_keywords_string_to_list(keywords_str):
    regex_filter = r"\s*\.*\s*Keywords\s*\d*:\s*"
    keywords_filtered = re.sub(regex_filter, "|", keywords_str)
    keywords_split = keywords_filtered.split("|")[1:]

    return keywords_split


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--keywords_str', type=str, help='Keywords in string form')
    args = parser.parse_args()

    print(convert_keywords_string_to_list(args.keywords_str))
