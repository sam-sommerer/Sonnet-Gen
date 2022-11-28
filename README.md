# Sonnet-Gen
NAACL 2022: Zero-shot Sonnet Generation with Discourse-level Planning and Aesthetics Features [https://aclanthology.org/2022.naacl-main.262/](https://aclanthology.org/2022.naacl-main.262/)

- Step 1 - Keyword generation
- Step 2 - Rhyme word generation
- Step 3 - Add simile and imagery
- Step 4 - Decoding

A few notes:
- Both step 1&2 are located in the keyword folder. 
- To directly use the pretrained model, run inference_bart_keywords_gen.ipynb and load the model from [https://huggingface.co/FigoMe/sonnet_keyword_gen](https://huggingface.co/FigoMe/sonnet_keyword_gen). Then at decoding time, load the pretrained model from [https://huggingface.co/FigoMe/news-gpt-neo-1.3B-keywords-line-by-line-reverse](https://huggingface.co/FigoMe/news-gpt-neo-1.3B-keywords-line-by-line-reverse)
- To train the keyword model yourself, run train-keywords-bart.ipynb (we shifted from T5 to bart)

## Citations
Please cite our paper if they are helpful to your work !
```bibtex 
    @inproceedings{tian-peng-2022-zero,
    title = "Zero-shot Sonnet Generation with Discourse-level Planning and Aesthetics Features",
    author = "Tian, Yufei  and Peng, Nanyun",
    booktitle = "Proceedings of the 2022 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language  Technologies",
    year = "2022",
    address = "Seattle, United States"
} 
```

## Villanelle Notes
- `keyword/inference_bart_keywords_gen.ipynb` contains code to generate keywords
- `keyword/generate_rhyme.ipynb` contains code to make lines rhyme
- Look at `decoding.ipynb` or `demo/decode.py` for examples of how to decode
  - `generate.py` seems to be a copy of the two above, safe to ignore
  - Villlanelles have no restraints on pentameter, can get rid of those checks
- Ignore `keyword/get_keywords.py` and `keyword/train-keywords-bart.ipynb`/`keyword/TrainKeywords.py`
- For polish functionality, the following links contain the files we need:
  - https://github.com/tuhinjubcse/SimileGeneration-EMNLP2020
    - https://drive.google.com/drive/folders/1KSANJ7XiPo0xqFCUG5WDhB3763EgEVnC
  - https://github.com/NinaTian98369/HypoGen
    - https://drive.google.com/drive/folders/1aexFfPMD8mRSaq_pQukD8NSTemxp1A0u

`salloc --time=5:00 --partition=gpu --nodes=1 --ntasks=1 --cpus-per-task=8 --mem=16GB --gres=gpu:v100:1`

