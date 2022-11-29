import torch
from transformers import GPT2Tokenizer, GPTNeoForCausalLM, GPTNeoModel
from transformers import GPT2Tokenizer, GPT2LMHeadModel
import pronouncing
from transformers import Trainer, TrainingArguments
from tqdm import tqdm
import random
from torch import Tensor
from torch.nn import functional as F

import warnings

warnings.filterwarnings("ignore")


def get_stress(phone):
    stress = []
    for s in phone.split():
        if s[-1].isdigit():
            if s[-1] == "2":
                stress.append(0)
            else:
                stress.append(int(s[-1]))
    return stress


def alternating(stress):
    # Check if the stress and unstress are alternating
    check1 = len(set(stress[::2])) <= 1 and (len(set(stress[1::2])) <= 1)
    check2 = len(set(stress)) == 2 if len(stress) >= 2 else True
    return check1 and check2


def get_phones(rhyme_word):
    phone = pronouncing.phones_for_word(rhyme_word)[0]
    stress = get_stress(phone)
    p_state = stress[0]
    n_syllables = len(stress)
    return p_state, n_syllables


def top_k_top_p_filtering(
    logits: Tensor,
    top_k: int = 0,
    top_p: float = 1.0,
    filter_value: float = -float("Inf"),
    min_tokens_to_keep: int = 1,
    return_index=False,
):
    """Filter a distribution of logits using top-k and/or nucleus (top-p) filtering
    Args:
        logits: logits distribution shape (batch size, vocabulary size)
        if top_k > 0: keep only top k tokens with highest probability (top-k filtering).
        if top_p < 1.0: keep the top tokens with cumulative probability >= top_p (nucleus filtering).
            Nucleus filtering is described in Holtzman et al. (http://arxiv.org/abs/1904.09751)
        Make sure we keep at least min_tokens_to_keep per batch example in the output
    From: https://gist.github.com/thomwolf/1a5a29f6962089e871b94cbd09daf317
    """
    if top_k > 0:
        top_k = min(max(top_k, min_tokens_to_keep), logits.size(-1))  # Safety check
        # Remove all tokens with a probability less than the last token of the top-k
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        indices_keep = logits >= torch.topk(logits, top_k)[0][..., -1, None]
        indices_keep = indices_keep[0].tolist()
        indices_keep = [i for i, x in enumerate(indices_keep) if x == True]
        logits[indices_to_remove] = filter_value

        if return_index:
            return logits, indices_keep

    if top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens with cumulative probability above the threshold (token with 0 are kept)
        sorted_indices_to_remove = cumulative_probs > top_p
        if min_tokens_to_keep > 1:
            # Keep at least min_tokens_to_keep (set to min_tokens_to_keep-1 because we add the first one below)
            sorted_indices_to_remove[..., :min_tokens_to_keep] = 0
        # Shift the indices to the right to keep also the first token above the threshold
        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        # scatter sorted tensors to original indexing
        indices_to_remove = sorted_indices_to_remove.scatter(
            -1, sorted_indices, sorted_indices_to_remove
        )
        logits[indices_to_remove] = filter_value

    # if return_index:
    #     return logits, indices_keep
    return logits


#  generates the next 10 words (assuming eos token isn't generated first) given an input_id
def generate_next_word(model, input_ids1, temperature=0.85, topk=100, device="cuda:0"):
    # print(f"enters generate_next_word")
    current_word = 0
    for i in range(10):
        # print(f"\t iteration i: {i}")
        outputs1 = model(input_ids1)
        next_token_logits1 = outputs1[0][:, -1, :]
        next_token_logits1 = top_k_top_p_filtering(next_token_logits1, top_k=topk)
        logit_zeros = torch.zeros(len(next_token_logits1)).cuda()
        # logit_zeros = torch.zeros(len(next_token_logits1), device=device)

        next_token_logits = next_token_logits1 * temperature
        probs = F.softmax(next_token_logits, dim=-1)
        next_tokens = torch.multinomial(probs, num_samples=1).squeeze(
            1
        )  # sampling next tokens from distribution
        # unfinished_sents = torch.ones(1, dtype=torch.long, device=device)
        unfinished_sents = torch.ones(1, dtype=torch.long).cuda()
        tokens_to_add = next_tokens * unfinished_sents + tokenizer.pad_token_id * (
            1 - unfinished_sents
        )  # same content as next_tokens

        if tokenizer.eos_token_id in next_tokens[0]:
            input_ids1 = torch.cat([input_ids1, tokens_to_add.unsqueeze(-1)], dim=-1)
            # return "", True
            return tokenizer.decode(input_ids1[0]).split()[-1], True

        #  what does this do? exit early if two spaces are generated in a row?
        if tokenizer.decode(tokens_to_add[0])[0] == " ":
            if current_word == 1:
                #  why decode here and nowhere else?
                return tokenizer.decode(input_ids1[0]).split()[-1], False
            current_word += 1

        input_ids1 = torch.cat([input_ids1, tokens_to_add.unsqueeze(-1)], dim=-1)
    # return None
    return tokenizer.decode(input_ids1[0]).split()[-1], False


def get_valid_samples(model, prompt, p_state, n_syllables, keywords):
    print(f"enters_valid_samples")
    print(f"\tkeywords: {keywords}")
    states = []
    all_n_syl = []

    prompts = []
    all_keywords = []
    # insert the keyword whenever possible
    for source_word in keywords:
        phone = pronouncing.phones_for_word(source_word)[0]
        stress = get_stress(phone)
        if not alternating(stress):
            continue

        # if the word is single syllable and can be either stressed or unstressed, flag = True
        can_be_stressed_or_unstressed = check_either_stress(stress, source_word)

        if stress[-1] == 1 - p_state or can_be_stressed_or_unstressed:
            states.append(stress[0])
            all_n_syl.append(n_syllables + len(stress))
            prompts.append(prompt + " " + source_word)
            copy = keywords.copy()
            copy.remove(source_word)
            all_keywords.append(copy)

    # The normal process of decoding
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.cuda()
    tokens = []
    # print(f"gets to valid_sample loop")
    while len(tokens) < 3:
        print(f"\ttokens: {tokens}")
        # print(f"\tinput_ids: {input_ids}")
        token, eos = generate_next_word(model, input_ids)
        print(f"\t\ttoken: {token}")
        if (token not in tokens) and (token not in keywords):
            # print(token, tokens)
            try:
                phone = pronouncing.phones_for_word(token)[0]
                stress = get_stress(phone)
                if not alternating(stress):
                    continue

                # if the word is single syllable and can be either stressed or unstressed, flag = True
                can_be_stressed_or_unstressed = check_either_stress(stress, token)

                if stress[-1] == 1 - p_state or can_be_stressed_or_unstressed:
                    print(f"\t\tadding token: {token}")
                    tokens.append(token)
                    states.append(stress[0])
                    all_n_syl.append(n_syllables + len(stress))
                    prompts.append(prompt + " " + token)
                    all_keywords.append(keywords)
            except:
                continue

    return prompts, states, all_n_syl, all_keywords


def check_either_stress(stress, source_word, loose=False):
    if loose:
        return len(stress) == 1
    if len(stress) == 1 and len(pronouncing.phones_for_word(source_word)) > 1:
        phone0 = pronouncing.phones_for_word(source_word)[0]
        phone1 = pronouncing.phones_for_word(source_word)[1]
        stress0 = [int(s[-1]) for s in phone0.split() if s[-1].isdigit()]
        stress1 = [int(s[-1]) for s in phone1.split() if s[-1].isdigit()]
        if stress0 + stress1 == 1 and stress0 * stress1 == 0:
            return True
    return False


def reverse_order(line):
    line = line.replace(", ", " , ")
    words = line.split()
    return " ".join(reversed(words)).replace(" , ", ", ")


def beam_search(model, true_beams, beam_size=5):
    beam_scorer = {}
    for sentence in true_beams:
        tokenize_input = tokenizer.tokenize(sentence)
        tensor_input = torch.tensor([tokenizer.convert_tokens_to_ids(tokenize_input)])

        tensor_input = tensor_input.to(device)
        loss = model(tensor_input, labels=tensor_input)
        avg_lp = torch.tensor(-loss[0].item() / len(tokenize_input))
        beam_scorer[sentence] = avg_lp
    beam_scorer = {
        k: v for k, v in sorted(beam_scorer.items(), key=lambda x: x[1], reverse=True)
    }
    return list(beam_scorer.keys())[:beam_size]


def gen_recursion(model, prompt, p_state, n_syllables, keywords, beam_size):
    global result_list
    """I modified this criterion to speed up the example.
    I suggest to add non-repeat-unigram (= 3) and keyword checking
    """
    if n_syllables >= 10:
        line = prompt.split(": ")[-1]
        reversed_words = reverse_order(line)
        result_list.append(reversed_words)
        # print(f'len of results list: {len(result_list)}')
        if len(result_list) > 0:
            # print('Going in Beam Search')
            result_list = beam_search(model, result_list, beam_size=beam_size)
            # print(result_list)
        return result_list
    prompts, states, all_n_sys, all_keywords = get_valid_samples(
        model, prompt, p_state, n_syllables, keywords
    )
    print(prompts)
    # prune the recursion tree by randomly selecting one prompt to decode, this speeds up the example for demo but compromises diversity
    k = random.randint(0, len(prompts))
    gen_recursion(
        model, prompts[0], states[0], all_n_sys[0], all_keywords[0], beam_size
    )
    # original code that explodes recursion exponentially
    # for prompt,p_state, n_syllables, keyword in zip(prompts, states, all_n_sys, all_keywords):
    #     gen_recursion(prompt,p_state, n_syllables, keywords)


def gen_villanelle(model, keywords_arr):
    result = ""
    first_line_repeat_indices = [5, 11, 17]
    first_line_repeat = ""
    third_line_repeat_indices = [8, 14, 18]
    third_line_repeat = ""
    for i, kws in enumerate(tqdm(keywords_arr)):
        # kws = four_seasons_story_line[2]
        print(f"keyword {i + 1}: {kws}")

        if i in first_line_repeat_indices:
            result += first_line_repeat + ","
        elif i in third_line_repeat_indices:
            result += third_line_repeat + ","
        else:
            rhyme_word = kws[-1]
            prefix = """Keywords: """ + "; ".join(kws) + ". Sentence in reverse order: "
            prompt = (
                """<|startoftext|> Title: """
                + example_title
                + " "
                + result
                + prefix
                + rhyme_word
            )
            p_state, n_syllables = get_phones(rhyme_word)
            result_list = []
            # to add hard constraints, specify keywords, otherwise use = []
            gen_recursion(model, prompt, p_state, n_syllables, keywords=[], beam_size=5)
            result = result + result_list[0] + ","

        if i == 0:
            first_line_repeat = result_list[0]

        if i == 2:
            third_line_repeat = result_list[0]

    return result


if __name__ == "__main__":
    # Download Finetuned GPT-Neo
    # Set the random seed to a fixed value to get reproducible results
    torch.manual_seed(42)
    tokenizer = GPT2Tokenizer.from_pretrained(
        "EleutherAI/gpt-neo-1.3B",
        bos_token="<|startoftext|>",
        eos_token="<|endoftext|>",
        pad_token="<|pad|>",
    )

    # Download the pre-trained GPT-Neo model and transfer it to the GPU
    model = GPTNeoForCausalLM.from_pretrained(
        "FigoMe/news-gpt-neo-1.3B-keywords-line-by-line-reverse"
    ).cuda()
    # Resize the token embeddings because we've just added 3 new tokens
    model.resize_token_embeddings(len(tokenizer))

    device = "cuda:0"
    score_model = model

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

    test_story = [
        ["computer", "scientists", "meetings"],
        ["years", "ago", "time"],
        ["today", "day", "dressings"],
        ["work", "lunch", "crime"],
        ["couple", "hours", "weeks"],
        ["happened", "n’t", "remember"],
        ["people", "talked", "leaks"],
        ["things", "wanted", "member"],
        ["answer", "told", "give"],
        ["confused", "thought", "stopped"],
        ["room", "looked", "live"],
        ["table", "sat", "cropped"],  # random words for rhyme scheme throw error
        ["screaming", "finally", "stood"],
        ["closer", "heard", "good"],
    ]

    #  not rhymed
    #  Keywords 1: ['years', 'time', 'ago'] Keywords 2: ['life', 'happened', 'finally'] Keywords 3: ['family', 'love', 'wanted'] Keywords 4: ['year', 'long', 'lived'] Keywords 5: ['day', 'couple','months'] Keywords 6: ['years', 'time', 'ago'] Keywords 7: ['night', 'n’t','sleep'] Keywords 8: ['bed', 'felt', 'cold'] Keywords 9: ['family', 'love', 'wanted'] Keywords 10: ['window','staring', 'darkness'] Keywords 11: ['doorway', 'heard','screaming'] Keywords 12: ['years', 'time', 'ago'] Keywords 13: ['floor','slowly', 'walked'] Keywords 14: ['bedroom', 'closet', 'turned'] Keywords 15: ['family', 'love', 'wanted'] Keywords 16: ['smiled', 'told', 'goodbye'] Keywords 17: ['face', 'laughed', 'gave'] Keywords 18: ['years', 'time', 'ago'] Keywords 19: ['family', 'love', 'wanted']

    example_title = "A Computer Scientist Meeting"

    result = gen_villanelle(model=model, keywords_arr=test_story)
    print(f"result: {result}")

    # previous = ""
    # for kws in tqdm(test_story):
    #     # kws = four_seasons_story_line[2]
    #     print(f"keyword: {kws}")
    #     rhyme_word = kws[-1]
    #     prefix = """Keywords: """ + "; ".join(kws) + ". Sentence in reverse order: "
    #     prompt = (
    #         """<|startoftext|> Title: """
    #         + example_title
    #         + " "
    #         + previous
    #         + prefix
    #         + rhyme_word
    #     )
    #     p_state, n_syllables = get_phones(rhyme_word)
    #     result_list = []
    #     # to add hard constraints, specify keywords, otherwise use = []
    #     gen_recursion(
    #         score_model, prompt, p_state, n_syllables, keywords=[], beam_size=5
    #     )
    #     previous = previous + result_list[0] + ","
    #
    # print(f"result: {previous}")
