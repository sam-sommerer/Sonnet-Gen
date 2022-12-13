from keyword_gen import inference_bart_keywords_gen
from keyword_gen import generate_rhymes
# import inference_bart_keywords_gen
# import generate_rhymes
import decode

import argparse

if __name__ == "__main__":
    print(f"Begins main process")

    parser = argparse.ArgumentParser()
    parser.add_argument("--title", type=str)
    args = parser.parse_args()

    raw_keywords = inference_bart_keywords_gen.get_keywords(title=args.title)
    rhyming_keywords = generate_rhymes.get_rhymes(keywords=raw_keywords)
    poem = decode.get_poem(title=args.title, keywords=rhyming_keywords)

    print(f"final poem: {poem}")

    output_dir = "generated_poems/"
    output_filename = args.title + ".txt"
    output_path = output_dir + output_filename

    with open(output_path, "w") as f:
        f.write(args.title + "\n\n")
        f.write(poem)
