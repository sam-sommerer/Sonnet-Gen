import numpy as np
import pandas as pd
import yake
import json
from tqdm import tqdm
from transformers import BartTokenizer


def has_numbers(input_string):
    return any(char.isdigit() for char in input_string)


def filter_input(unfiltered_news, unfiltered_headlines):
    tokens = []
    sentences = []
    paragraphs = []
    count = 0
    filtered_stories = []
    filtered_prompts = []
    for s, p in zip(unfiltered_news, unfiltered_headlines):
        if has_numbers(p) or pd.isna(s) or "@" in s or "#" in s:
            continue
        try:
            s = s.strip("\n").replace("?s", "'s").replace("?", "").replace("..", ".")
            s = s.replace("..", ".").replace("\n", "").replace("'", "'")
            s = s.replace("..", ".")
            token_l = len(s.split())
            # batch = tok(s, return_tensors="pt")
            sent_l = len(s.split("."))
            # sents = s.split(".")
            # flag = True
            # print(len(batch['input_ids'][0]) /token_l)
            if 8 <= sent_l <= 50:
                # if len(batch['input_ids'][0]) < 1.5*token_l and 8<=sent_l<=50:
                count += 1
                tokens.append(token_l)
                sentences.append(sent_l)
                filtered_stories.append(s)
                filtered_prompts.append(p)
        except:
            continue

    return tokens, sentences, paragraphs, filtered_stories, filtered_prompts, count


def get_and_dump_keywords(stories, prompts):
    all_story = []
    for s, p in tqdm(zip(stories, prompts)):
        p = p.replace("\n", "")
        story = dict()
        story["Theme"] = p
        sents = s.split(".")
        story_keywords = []
        informative_lines = []
        sentiments = []
        for sent in sents:
            kws = kw_extractor.extract_keywords(sent)
            keywords = [kw[0] for kw in kws]
            if len(keywords) >= 3:
                informative_lines.append(sent)
                story_keywords.append(keywords)
        story["sentiments"] = sentiments
        story["keywords"] = story_keywords
        story["sentences"] = informative_lines
        all_story.append(story)
        # if len(all_story) % 500 == 1:
        #     print(len(all_story))
        #     with open("../data/all_news_short_theme.json", "w") as f:
        #         json.dump(all_story, f, indent=4)
    with open("../data/all_news_short_theme.json", "w") as f:
        json.dump(all_story, f, indent=4)


if __name__ == "__main__":
    tok = BartTokenizer.from_pretrained("facebook/bart-base")
    kw_extractor = yake.KeywordExtractor(n=1, top=3)

    path = "../data/news_summary/"
    filename = "news_summary_more.csv"
    df = pd.read_csv(path + filename)

    headlines = df["headlines"]
    news = df["text"]

    # for i in news[:10]:
    #     print(i)

    tokens, sentences, paragraphs, filtered_stories, filtered_prompts, count = filter_input(news, headlines)

    print(count)
    print("average tokens of story:", np.mean(tokens))
    print("average sentences of story:", np.mean(sentences))
    print("average tokens per sentence:", np.mean(tokens) / np.mean(sentences))

    print(f"len(filtered_stories), len(filtered_prompts): {len(filtered_stories)}, {len(filtered_prompts)}")

    get_and_dump_keywords(filtered_stories, filtered_prompts)

