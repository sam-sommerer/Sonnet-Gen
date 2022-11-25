#!/usr/bin/env python
# coding: utf-8

# In[280]:


four_seasons_story_line = [
    ["snow", "falling", "future"],
    ["winter", "is", "coming"],
    ["gather", "honest", "humor"],
    ["spring", "happy", "blooming"],
    ["air", "heat", "warm"],
    ["little", "birds", "may"],
    ["flowers", "leaves", "storm"],
    ["summer", "moon", "day"],
    ["blue", "sky", "clouds"],
    ["sudden", "rain", "thunder"],
    ["Summer", "fill", "crowds"],
    ["Spring", "no", "wonder"],
    ["seasons", "years", "keep"],
    ["future", "months", "reap"],
]


# In[2]:


import random
import spacy

nlp = spacy.load("en_core_web_sm")
import os
import sys

sys.path.append(os.getcwd())

import src.data.data as data
import src.data.config as cfg
import src.interactive.functions as interactive

import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
cfg.device = "cpu"


# ### Imagery

# In[4]:


# load model
model_file = "pretrained_models/reverse_comet_1e-05_adam_32_20000.pickle"
opt, state_dict = interactive.load_model_file(model_file)
data_loader, text_encoder = interactive.load_data("conceptnet", opt)

n_ctx = data_loader.max_e1 + data_loader.max_e2 + data_loader.max_r
n_vocab = len(text_encoder.encoder) + n_ctx

model = interactive.make_model(opt, 40543, 29, state_dict)


# In[5]:


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


# In[6]:


def getPred(input_event, relation, prnt=True, sampling_algorithm="beam-2"):
    sampler = interactive.set_sampler(opt, sampling_algorithm, data_loader)
    outputs = interactive.get_conceptnet_sequence(
        input_event, model, sampler, data_loader, text_encoder, relation, prnt
    )
    return outputs


# In[33]:


# randomly sample at most N=5 nouns, not from the same line
# then, select the most confident M candidates to do the replacement
N = 5
M = 2

if __name__ == "__main__":
    location_dict = {}
    for i, keywords in enumerate(four_seasons_story_line):
        w1, w2, _ = keywords
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
    flatten_list = [j for sub in four_seasons_story_line for j in sub]
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
        four_seasons_story_line[location[0]][location[1]] = replace_dict[ent]


# Ashima TODO👆: instead of randomly select, include the user in the word selection process. again, list same for simile

# ### Simile

# In[34]:


from fairseq.models.bart import BARTModel
from transformers import AutoModelForCausalLM, AutoTokenizer
import time
import numpy as np
import pickle

tok = AutoTokenizer.from_pretrained("facebook/bart-large")
datadir = "simile"
cpdir = datadir + "/"
bart = BARTModel.from_pretrained(
    cpdir,
    checkpoint_file="checkpoint-simile/checkpoint_best.pt",
    data_name_or_path=datadir,
)

bart.cuda()
bart.eval()


# In[37]:


import pronouncing


# In[256]:


def get_stress(phone):
    stress = []
    for s in phone.split():
        if s[-1].isdigit():
            stress.append(int(s[-1]))
    for i, number in enumerate(stress):
        if number == 2:
            if i == 0:
                stress[0] = 1 - stress[1]
            else:
                stress[i] = 1 - stress[i - 1]
    return stress


# In[45]:


def alternating(stress):
    # Check if the stress and unstress are alternating
    check1 = len(set(stress[::2])) <= 1 and (len(set(stress[1::2])) <= 1)
    check2 = len(set(stress)) == 2 if len(stress) >= 2 else True
    return check1 and check2


# In[277]:


def is_none_phrase(vehicle):
    if vehicle.startswith("a "):
        vehicle = vehicle.replace("a ", "")
    doc = nlp(vehicle)
    flag = False
    for ent in doc:
        if ent.pos_ == "NOUN" or ent.pos_ == "PROPN":
            return True
    return False


# Ashima TODO👆: please check if you can install allennlp, if yes, we should use https://demo.allennlp.org/constituency-parsing to detect the noun phrase

# In[161]:


def check_meter(adj, vehicle):
    phone = pronouncing.phones_for_word(adj)[0]
    stress_adj = get_stress(phone)
    vehicle = vehicle.strip(",.<>")
    try:
        if len(vehicle.split()) == 1:
            phone = pronouncing.phones_for_word(vehicle)[0]
            stress_vehicle = get_stress(phone)
        else:
            stress_vehicle = []
            for word in vehicle.split():
                phone = pronouncing.phones_for_word(word)[0]
                stress_vehicle += get_stress(phone)
        # assume 'like' can be either stressed or unstressed
        if stress_vehicle[0] == stress_adj[-1] and alternating(stress_vehicle):
            return True
    except:
        pass
    return False


# In[272]:


check_meter("incredible", "a miracle")


# In[271]:


is_none_phrase("a miracle")


# In[269]:


def simile_vehicle(inp, t=0.7):
    simile_phrases = []
    prefix = inp + " like"
    inputs = [prefix] * 10
    l = len(tok(prefix)[0])
    hypotheses_batch = bart.sample(
        inputs,
        sampling=True,
        sampling_topk=5,
        temperature=t,
        max_len_b=l + 2,
        min_len=l,
    )
    for hypothesis in hypotheses_batch:
        vehicle = hypothesis.split(" like ")[1].split("<")[0].lower()
        print(vehicle)
        if is_none_phrase(vehicle) and check_meter(inp, vehicle):
            simile_phrases.append(" ".join([inp, "like", vehicle]))
    return list(set(simile_phrases))


# In[278]:


simile_vehicle("incredible")


# In[287]:


simile_vehicle("polite")


# Ashima TODO: involve the users in this decision process 👆

# In[283]:


if __name__ == "__main__":
    location_dict = {}
    for i, keywords in enumerate(four_seasons_story_line):
        if i not in polished_lines:
            w1, w2, _ = keywords
            ent = nlp(w1)[0]
            if ent.pos_ == "ADJ":
                location_dict[str(ent)] = [i, 0]
                continue
            ent = nlp(w2)[0]
            if ent.pos_ == "ADJ":
                location_dict[str(ent)] = [i, 1]

    samples = random.sample(location_dict.keys(), M)
    for ent in samples:
        simile_phrase = simile_vehicle(ent)
        print(simile_phrase)
        location = location_dict[ent]
        polished_lines.append(location[0])
        four_seasons_story_line[location[0]][location[1]] = simile_phrase


# In[ ]:
