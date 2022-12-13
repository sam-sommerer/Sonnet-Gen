from transformers import pipeline
import pronouncing
import random
import argparse


def get_rhyme_candidates(word, N):
    temp = pronouncing.rhymes(word)
    if len(temp) <= N:
        return temp
    else:
        return random.sample(temp, N)


def make_repeating_lines_rhyme(keywords):
    # repeating_lines_indices = [2, 8, 14, 18]

    word = keywords.split(" Keywords")[0].split("'")[5]
    candidates = get_rhyme_candidates(word, N=30)

    if len(candidates) == 0:
        candidates = get_rhyme_candidates("quickly", N=30)

    temp = keywords.split(" Keywords")[2].split("'")
    print(f"temp: {temp}")
    replace_word = keywords.split(" Keywords")[2].split("'")[5]
    print(f"replace_word: {replace_word}")

    mask_input = keywords.replace(replace_word, model.tokenizer.mask_token)
    result = model(mask_input, top_k=5000)
    # print(f"result: {result[]}")
    tokens = [res["token_str"] for res in result[0]]
    found = False
    rhyming_word = ""
    for t in tokens:
        if t in candidates:
            print("generated rhyme word:", t)
            found = True
            rhyming_word = t
            break

    if not found:
        candidates_target_str = [" " + c for c in candidates]
        result = model(mask_input, targets=candidates_target_str)
        tokens = [res["token_str"] for res in result]
        for t in tokens:
            if t in candidates_target_str:
                print("generated rhyme word:", t)
                found = True
                rhyming_word = t
                break
        if not found:
            rhyming_word = random.choice(candidates)
            print(
                f"couldn't generate rhyme with {word}, using {rhyming_word} from candidates instead"
            )

    keywords = keywords.replace(replace_word, rhyming_word)

    return keywords


def generate_rhymes(model, keywords, initial_rhyming_lines, countin_rhyming_lines):
    keywords = make_repeating_lines_rhyme(keywords)

    for i in range(len(initial_rhyming_lines)):
        # indices for keywords are 1,3,5.
        word = keywords.split(" Keywords")[initial_rhyming_lines[i]].split("'")[5]
        print(f"word: {word}")
        candidates = get_rhyme_candidates(word, N=30)
        # candidates = pronouncing.rhymes(word)
        # print(f"candidates: {candidates}")
        if len(candidates) == 0:
            candidates = get_rhyme_candidates("quickly", N=30)

        temp = keywords.split(" Keywords")[countin_rhyming_lines[i]].split("'")
        print(f"temp: {temp}")
        replace_word = keywords.split(" Keywords")[countin_rhyming_lines[i]].split("'")[
            5
        ]
        print(f"replace_word: {replace_word}")

        mask_input = keywords.replace(replace_word, model.tokenizer.mask_token)
        result = model(mask_input, top_k=5000)
        tokens = [res["token_str"] for res in result]
        found = False
        rhyming_word = ""
        for t in tokens:
            if t in candidates:
                print("generated rhyme word:", t)
                found = True
                rhyming_word = t
                break

        if not found:
            candidates_target_str = [" " + c for c in candidates]
            result = model(mask_input, targets=candidates_target_str)
            tokens = [res["token_str"] for res in result]
            for t in tokens:
                if t in candidates_target_str:
                    print("generated rhyme word:", t)
                    found = True
                    rhyming_word = t
                    break
            if not found:
                rhyming_word = random.choice(candidates)
                print(
                    f"couldn't generate rhyme with {word}, using {rhyming_word} from candidates instead"
                )

        keywords = keywords.replace(replace_word, rhyming_word)

    return keywords


def get_rhymes(keywords):
    path = "facebook/bart-base"
    model = pipeline("fill-mask", model=path)

    initial_rhyming_lines = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]
    countin_rhyming_lines = [3, 6, 9, 12, 15, 4, 7, 10, 13, 16]

    rhyming_keywords = generate_rhymes(
        model=model,
        keywords=keywords,
        initial_rhyming_lines=initial_rhyming_lines,
        countin_rhyming_lines=countin_rhyming_lines,
    )
    print("Generated rhyme words; ", rhyming_keywords, sep="\n")
    return rhyming_keywords


if __name__ == "__main__":
    path = "facebook/bart-base"
    model = pipeline("fill-mask", model=path)

    # zero index
    # initial_rhyming_lines = [0,1,4,5,8,9,12]
    # countin_rhyming_lines = [2,3,6,7,10,11,13]

    # A1 b A2 / a b A1 / a b A2 / a b A1 / a b A2 / a b A1 A2
    # a_rhyming_lines = [3, 6, 9, 12, 15]
    # b_rhyming_lines = [1, 4, 7, 10, 13, 16]

    # # initial_rhyming_lines = [3, 6, 9, 12, 1, 4, 7, 10, 13]
    # initial_rhyming_lines = [3, 3, 3, 3, 1, 1, 1, 1, 1]
    # # countin_rhyming_lines = [6, 9, 12, 15, 4, 7, 10, 13, 16]
    # countin_rhyming_lines = [6, 9, 12, 15, 4, 7, 10, 13, 16]

    # initial_rhyming_lines = []
    initial_rhyming_lines = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]
    countin_rhyming_lines = [3, 6, 9, 12, 15, 4, 7, 10, 13, 16]

    # example_keywords = "Keywords 1: ['wrong', 'things', 'trade'] . Keywords 2: ['Silhouettes', 'yard', 'bright'] . Keywords 3: ['safe', 'feel', 'cozy'] . Keywords 4: ['air', 'lifted', 'today'] . Keywords 5: ['grounds', 'spirits', 'inhabit'] . Keywords 6: ['mist', 'wrong', 'deeply'] . Keywords 7: ['mind', 'cottage', 'engulfed'] . Keywords 8: ['legs', 'dog', 'tail'] . Keywords 9: ['reached', 'house', 'cold'] . Keywords 10: ['silence', 'air', 'shook'] . Keywords 11: ['rumble', 'thunder', 'localized'] . Keywords 12: ['animal', 'slumber', 'horrible'] . Keywords 13: ['cottage', 'glance', 'dashed'] . Keywords 14: ['ran', 'life', 'cabin'] </s>"
    # example_keywords = "Keywords 1: ['years', 'time', 'ago'] . Keywords 2: ['life', 'happened', 'quickly'] . Keywords 3: ['family', 'love', 'wanted'] . Keywords 4: ['year', 'long', 'lived'] . Keywords 5: ['day', 'couple','months'] . Keywords 6: ['home','sitting', 'room'] . Keywords 7: ['night', 'n’t','sleep'] . Keywords 8: ['bed', 'felt', 'cold'] . Keywords 9: ['eyes', 'looked', 'back'] . Keywords 10: ['window','staring', 'darkness'] . Keywords 11: ['doorway', 'heard','screaming'] . Keywords 12: ['open', 'opened','stairs'] . Keywords 13: ['floor','slowly', 'walked'] . Keywords 14: ['bedroom', 'closet', 'turned'] . Keywords 15: ['moment','shook', 'head'] . Keywords 16: ['smiled', 'told', 'goodbye'] . Keywords 17: ['face', 'laughed', 'gave'] . Keywords 18: ['breath','started', 'running'] . Keywords 19: ['house', 'ran', 'downstairs']"
    example_keywords = "Keywords 1: ['years', 'time', 'ago'] Keywords 2: ['life', 'happened', 'finally'] Keywords 3: ['family', 'love', 'wanted'] Keywords 4: ['year', 'long', 'lived'] Keywords 5: ['day', 'couple','months'] Keywords 6: ['years', 'time', 'ago'] Keywords 7: ['night', 'n’t','sleep'] Keywords 8: ['bed', 'felt', 'cold'] Keywords 9: ['family', 'love', 'wanted'] Keywords 10: ['window','staring', 'darkness'] Keywords 11: ['doorway', 'heard','screaming'] Keywords 12: ['years', 'time', 'ago'] Keywords 13: ['floor','slowly', 'walked'] Keywords 14: ['bedroom', 'closet', 'turned'] Keywords 15: ['family', 'love', 'wanted'] Keywords 16: ['smiled', 'told', 'goodbye'] Keywords 17: ['face', 'laughed', 'gave'] Keywords 18: ['years', 'time', 'ago'] Keywords 19: ['family', 'love', 'wanted']"

    parser = argparse.ArgumentParser()
    parser.add_argument("--keywords", type=str, default=example_keywords)
    args = parser.parse_args()

    rhyming_keywords = generate_rhymes(
        model=model,
        keywords=args.keywords,
        initial_rhyming_lines=initial_rhyming_lines,
        countin_rhyming_lines=countin_rhyming_lines,
    )
    print("Generated rhyme words; ", rhyming_keywords, sep="\n")
