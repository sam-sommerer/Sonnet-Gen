from transformers import pipeline
import pronouncing
import random


def get_rhyme_candidates(word, N):
    temp = pronouncing.rhymes(word)
    if len(temp) <= N:
        return temp
    else:
        return random.sample(temp, N)


def generate_rhymes(model, keywords, initial_rhyming_lines, countin_rhyming_lines):
    for i in range(7):
        # indices for keywords are 1,3,5.
        word = keywords.split(" Keywords")[initial_rhyming_lines[i]].split("'")[5]
        print(f"word: {word}")
        candidates = get_rhyme_candidates(word, N=30)
        # candidates = pronouncing.rhymes(word)

        temp = keywords.split(" Keywords")[countin_rhyming_lines[i]].split("'")
        print(f"temp: {temp}")
        replace_word = keywords.split(" Keywords")[countin_rhyming_lines[i]].split("'")[5]
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
            candidates = [" " + c for c in candidates]
            result = model(mask_input, targets=candidates)
            tokens = [res["token_str"] for res in result]
            for t in tokens:
                if t in candidates:
                    print("generated rhyme word:", t)
                    found = True
                    rhyming_word = t
                    break
            if not found:
                print("not found", word)

        if found:
            keywords = keywords.replace(replace_word, rhyming_word)

    return keywords


if __name__ == "__main__":
    path = "facebook/bart-base"
    model = pipeline("fill-mask", model=path)

    # zero index
    # initial_rhyming_lines = [0,1,4,5,8,9,12]
    # countin_rhyming_lines = [2,3,6,7,10,11,13]

    # A1 b A2 / a b A1 / a b A2 / a b A1 / a b A2 / a b A1 A2
    # a_rhyming_lines = [3, 6, 9, 12, 15]
    # b_rhyming_lines = [1, 4, 7, 10, 13, 16]

    initial_rhyming_lines = [3, 6, 9, 12, 1, 4, 7, 10, 13]
    countin_rhyming_lines = [6, 9, 12, 15, 4, 7, 10, 13, 16]

    example_keywords = "Keywords 1: ['wrong', 'things', 'trade'] . Keywords 2: ['Silhouettes', 'yard', 'bright'] . Keywords 3: ['safe', 'feel', 'cozy'] . Keywords 4: ['air', 'lifted', 'today'] . Keywords 5: ['grounds', 'spirits', 'inhabit'] . Keywords 6: ['mist', 'wrong', 'deeply'] . Keywords 7: ['mind', 'cottage', 'engulfed'] . Keywords 8: ['legs', 'dog', 'tail'] . Keywords 9: ['reached', 'house', 'cold'] . Keywords 10: ['silence', 'air', 'shook'] . Keywords 11: ['rumble', 'thunder', 'localized'] . Keywords 12: ['animal', 'slumber', 'horrible'] . Keywords 13: ['cottage', 'glance', 'dashed'] . Keywords 14: ['ran', 'life', 'cabin'] </s>"

    rhyming_keywords = generate_rhymes(model=model, keywords=example_keywords, initial_rhyming_lines=initial_rhyming_lines, countin_rhyming_lines=countin_rhyming_lines)
    print("Generated rhyme words; ", rhyming_keywords, sep="\n")
