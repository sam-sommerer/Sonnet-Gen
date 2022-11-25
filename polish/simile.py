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
