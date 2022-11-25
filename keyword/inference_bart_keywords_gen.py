# Importing libraries
import json
import os
import re
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
    regex_filter = r"\s*\.*\s*Keywords\s*\d*:\s*"
    first_tercet_filtered = re.sub(regex_filter, "|", first_tercet[:-5])
    first_tercet_split = first_tercet_filtered.split()

    num_lines = 19
    result = ""

    result += ". " + first_tercet[:-4]

    first_line_repeat_indices = [5, 11, 17]
    third_line_repeat_indices = [8, 14, 18]

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

    return result + " </s>"


def generate_keywords(title, model, tokenizer, device):
    prompt = "Generate keywords for the title: "

    #  get keywords for first tercet
    first_tercet_placeholder = ". Keywords 1: ['<MASK>', '<MASK>', '<MASK>'] . Keywords 2: ['<MASK>', '<MASK>', '<MASK>'] . Keywords 3: ['<MASK>', '<MASK>', '<MASK>'] </s>"
    first_tercet_bart_input = prompt + title + first_tercet_placeholder
    first_tercet_preds = fill_in_mask(first_tercet_bart_input, model, tokenizer, device)

    print(f"Finished generating first tercet")
    print(f"first_tercet_preds: {first_tercet_preds}")

    #  get keywords for rest of villanelle using first tercet keywords
    placeholder = create_villanelle_keyword_masks(first_tercet_preds[0])
    bart_input = prompt + title + placeholder
    preds = fill_in_mask(bart_input, model, tokenizer)

    print(f"Finished generating final preds")
    print(f"preds: {preds}")

    return preds


if __name__ == "__main__":
    # model_path = 'facebook/bart-large_batch_8_lr_3e-060503-mix-with-eos/model_files'
    model_path = "FigoMe/sonnet_keyword_gen"  # for training
    tokenizer = BartTokenizer.from_pretrained(model_path)
    model = BartForConditionalGeneration.from_pretrained(model_path)
    device = "cuda:0" if cuda.is_available() else "cpu"
    model = model.to(device)

    title = "The Four Seasons"

    generate_keywords(title, model, tokenizer, device)