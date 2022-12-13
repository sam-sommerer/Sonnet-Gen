import random
import spacy

nlp = spacy.load("en_core_web_sm")
import os
import sys

if "polish" in os.getcwd():
    path = os.getcwd()
else:
    path = os.path.join(os.getcwd(), "polish")
sys.path.append(path)

import src
import src.data.data as data
import src.data.config as cfg
import src.interactive.functions as interactive

import utils.utils as utils
import argparse

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
cfg.device = "cpu"


### Imagery
# load model
model_file = os.path.join(path, "pretrained_models/reverse_comet_1e-05_adam_32_20000.pickle")
opt, state_dict = interactive.load_model_file(model_file)
data_loader, text_encoder = interactive.load_data("conceptnet", opt)

n_ctx = data_loader.max_e1 + data_loader.max_e2 + data_loader.max_r
n_vocab = len(text_encoder.encoder) + n_ctx

model = interactive.make_model(opt, 40543, 29, state_dict)

def getloss(input_e1, input_e2, relation, prnt=False):
    if relation not in data.conceptnet_data.conceptnet_relations:
        # if relation == "common":
        #     relation = common_rels
        # else:
        #     relation = "all"
        relation = "all"
    outputs = interactive.evaluate_conceptnet_sequence(
        input_e1, model, data_loader, text_encoder, relation, input_e2
    )

    for key, value in outputs.items():
        # if prnt:
        #     print(
        #         "{} \t {} {} {} \t\t norm: {:.4f} \t".format(
        #             input_e1,
        #             key,
        #             rel_formatting[key],
        #             input_e2,
        #             value["normalized_loss"],
        #         )
        #     )
        if prnt:
            print(
                "{} \t {} {} \t\t norm: {:.4f} \t".format(
                    input_e1,
                    key,
                    input_e2,
                    value["normalized_loss"],
                )
            )
        return round(value["normalized_loss"], 4)

def getPred(input_event, relation, prnt=True, sampling_algorithm="beam-2"):
    sampler = interactive.set_sampler(opt, sampling_algorithm, data_loader)
    outputs = interactive.get_conceptnet_sequence(
        input_event, model, sampler, data_loader, text_encoder, relation, prnt
    )
    return outputs

# randomly sample at most N=5 nouns, not from the same line
# then, select the most confident M candidates to do the replacement
N = 5
M = 2

def imagery_replacement(keywords):
    keywords = utils.convert_keywords_string_to_list(keywords)
    location_dict = {}
    for i, kws in enumerate(keywords):
        w1, w2, _ = kws
        ent = nlp(w1)[0]
        if ent.pos_ == "NOUN":
            location_dict[str(ent)] = [i, 0]
            continue
        ent = nlp(w2)[0]
        if ent.pos_ == "NOUN":
            location_dict[str(ent)] = [i, 1]
    samples = random.sample(location_dict.keys(), N)
    relations = ["SymbolOf"]
    score_dict = {}
    replace_dict = {}
    polished_lines = []
    flatten_list = [j for sub in keywords for j in sub]
    for ent in samples:
        result = getPred(
            ent, relation=relations, sampling_algorithm="topk-10", prnt=False
        )[relations[0]]["beams"]
        for i in range(len(result)):
            if result[i] not in flatten_list:
                result = result[i]
                break
        score_dict[ent] = getloss(ent, result, "SymbolOf", prnt=False)
        replace_dict[ent] = result

    selected = sorted(score_dict.items(), key=lambda item: item[1])[:M]
    print(f"replacing {replace_dict}")
    for ent in selected:
        ent = ent[0]
        location = location_dict[ent]
        polished_lines.append(location[0])
        keywords[location[0]][location[1]] = replace_dict[ent]

    return keywords

if __name__ == "__main__":

    villanelle_keywords = "Keywords 1: ['years', 'time', 'ago'] Keywords 2: ['life', 'happened', 'finally'] Keywords 3: ['family', 'love', 'go'] Keywords 4: ['year', 'long', 'au'] Keywords 5: ['day', 'couple','stickley'] Keywords 6: ['years', 'time', 'ago'] Keywords 7: ['night', 'n’t','au'] Keywords 8: ['bed', 'felt', 'thickly'] Keywords 9: ['family', 'love', 'go'] Keywords 10: ['window','staring', 'go'] Keywords 11: ['doorway', 'heard','prickly'] Keywords 12: ['years', 'time', 'ago'] Keywords 13: ['floor','slowly', 'slow'] Keywords 14: ['bedroom', 'closet', 'thickly'] Keywords 15: ['family', 'love', 'go'] Keywords 16: ['smiled', 'told', 'no'] Keywords 17: ['face', 'laughed', 'nicley'] Keywords 18: ['years', 'time', 'ago'] Keywords 19: ['family', 'love', 'go']"

    parser = argparse.ArgumentParser()
    parser.add_argument("--keywords", type=str, default=villanelle_keywords)
    args = parser.parse_args()

    print(imagery_replacement(args.keywords))