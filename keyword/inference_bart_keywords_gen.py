# Importing libraries
import json
import os
import re
import ast
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler
from torch import cuda

# Importing the T5 modules from huggingface/transformers
from transformers import BartTokenizer, BartForConditionalGeneration

# rich: for a better display on terminal
from rich.table import Column, Table
from rich import box
from rich.console import Console

# define a rich console logger
console = Console(record=True)


def clean_keywords(keywords_str):
    pattern = r"\s*\.*\s*Keywords\s*\d*\:*\s*\[.*?\]"
    # pattern = r"\s*\.*\s*Keywords\s*\d*\:*\s*\[(.*?)\]"
    keywords_match = re.findall(pattern, keywords_str)
    # print(f"keywords_match: {keywords_match}")

    # keyword_prefix_pattern = r"\s*\.*\s*Keywords\s*\d*:\s*"
    # piped_keywords = re.sub(keyword_prefix_pattern, "|", keywords_match)
    # keywords_split = piped_keywords.split("|")

    # return keywords_match
    return " ".join(keywords_match)


def fill_in_mask(bart_input, model, tokenizer, device):
    ids = tokenizer(bart_input, return_tensors="pt").input_ids.to(device)
    generated_ids = model.generate(
        input_ids=ids,
        max_length=512,
        min_length=200,
        num_beams=4,
        no_repeat_ngram_size=5,
        # topp = 0.9,
        # do_sample=True,
        repetition_penalty=5.8,
        length_penalty=1,
        early_stopping=True,
    )
    preds = [
        tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=True)
        for g in generated_ids
    ]

    return preds


def create_villanelle_keyword_masks(first_tercet):
    print(f"\ncreate_villanelle first_tercet: {first_tercet}")
    regex_filter = r"\s*\.*\s*Keywords\s*\d*:\s*"
    first_tercet_filtered = re.sub(regex_filter, "|", first_tercet)
    first_tercet_split = first_tercet_filtered.split("|")[1:]

    print(f"first_tercet_split: {first_tercet_split}")

    num_lines = 19
    result = ""

    first_line_repeat_indices = [0, 5, 11, 17]
    third_line_repeat_indices = [2, 8, 14, 18]

    for i in range(num_lines):
        prefix = " . Keywords " + str(i + 1) + ": "
        if i < 3:
            new_entry = prefix + first_tercet_split[i]
            result += new_entry
        elif i in first_line_repeat_indices:
            new_entry = prefix + first_tercet_split[0]
            result += new_entry
        elif i in third_line_repeat_indices:
            new_entry = prefix + first_tercet_split[2]
            result += new_entry
        else:
            new_entry = prefix + "['<MASK>', '<MASK>', '<MASK>']"
            result += new_entry

    print(f"villanelle mask: {result}\n")

    return result + " </s>"


def format_final_output(preds):
    print(f"\ncreate_villanelle first_tercet: {preds}")
    regex_filter = r"\s*\.*\s*Keywords\s*\d*:\s*"
    preds_filtered = re.sub(regex_filter, "|", preds)
    preds_split = preds_filtered.split("|")[1:]

    first_line_repeat_indices = [0, 5, 11, 17]
    third_line_repeat_indices = [2, 8, 14, 18]

    num_lines = 19
    result = ""

    for i in range(num_lines):
        prefix = "Keywords " + str(i + 1) + ": "
        if i in first_line_repeat_indices:
            new_entry = prefix + preds_split[0]
            result += new_entry
        elif i in third_line_repeat_indices:
            new_entry = prefix + preds_split[2]
            result += new_entry
        else:
            new_entry = prefix + preds_split[i]
            result += new_entry

    return result


def generate_keywords(title, model, tokenizer, device):
    print(f"generating keywords...")

    prompt = "Generate keywords for the title: "

    #  get keywords for first tercet
    first_tercet_placeholder = ". Keywords 1: ['<MASK>', '<MASK>', '<MASK>'] . Keywords 2: ['<MASK>', '<MASK>', '<MASK>'] . Keywords 3: ['<MASK>', '<MASK>', '<MASK>'] </s>"
    first_tercet_bart_input = prompt + title + first_tercet_placeholder
    first_tercet_preds = fill_in_mask(first_tercet_bart_input, model, tokenizer, device)

    print(f"Finished generating first tercet")
    print(f"first_tercet_preds: {first_tercet_preds}")

    first_tercet_preds[0] = clean_keywords(first_tercet_preds[0])
    print(f"cleaned first preds: {first_tercet_preds}")

    #  get keywords for rest of villanelle using first tercet keywords
    placeholder = create_villanelle_keyword_masks(first_tercet_preds[0])
    bart_input = prompt + title + placeholder
    preds = fill_in_mask(bart_input, model, tokenizer, device)

    print(f"Finished generating final preds")
    print(f"preds: {preds}")

    preds[0] = clean_keywords(preds[0])
    print(f"cleaned preds: {preds}")

    return preds


if __name__ == "__main__":
    # model_path = 'facebook/bart-large_batch_8_lr_3e-060503-mix-with-eos/model_files'
    print(f"Beginning")
    model_path = "FigoMe/sonnet_keyword_gen"  # for training
    tokenizer = BartTokenizer.from_pretrained(model_path)
    model = BartForConditionalGeneration.from_pretrained(model_path)
    device = "cuda:0" if cuda.is_available() else "cpu"
    model = model.to(device)

    title = "The Four Seasons"

    preds = generate_keywords(title, model, tokenizer, device)
    formatted_preds = format_final_output(preds[0])

    print(f"\nformatted_preds: {formatted_preds}")