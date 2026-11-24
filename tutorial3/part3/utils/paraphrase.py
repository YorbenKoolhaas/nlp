import time
import argparse
import pickle
import json
import os
import random

import nltk
from nltk.tokenize import sent_tokenize
import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration

# Prefer project NLTK data (home quota is full; ~/nltk_data/punkt_tab may be incomplete)
_NLTK_DATA = os.environ.get("NLTK_DATA", "/projects/0/gusr0466/eilia/nltk_data")
if _NLTK_DATA not in nltk.data.path:
    nltk.data.path.insert(0, _NLTK_DATA)


#This coding block for the DipperParaphraser class is borrowed by the authors of the original paper as the suggested
#method to use their model. See https://huggingface.co/kalpeshk2011/dipper-paraphraser-xxl for more details.
class DipperParaphraser(object):
    def __init__(self, model="kalpeshk2011/dipper-paraphraser-xxl", verbose=True):
        time1 = time.time()
        self.tokenizer = T5Tokenizer.from_pretrained('google/t5-v1_1-xxl')
        self.model = T5ForConditionalGeneration.from_pretrained(model, device_map="auto", torch_dtype=torch.float16)
        if verbose:
            print(f"{model} model loaded in {time.time() - time1}")
        self.model.cuda()
        self.model.eval()

    def paraphrase(self, input_text, lex_diversity, order_diversity, prefix="", sent_interval=3, **kwargs):
        """Paraphrase a text using the DIPPER model.

        Args:
            input_text (str): The text to paraphrase. Make sure to mark the sentence to be paraphrased between <sent> and </sent> blocks, keeping space on either side.
            lex_diversity (int): The lexical diversity of the output, choose multiples of 20 from 0 to 100. 0 means no diversity, 100 means maximum diversity.
            order_diversity (int): The order diversity of the output, choose multiples of 20 from 0 to 100. 0 means no diversity, 100 means maximum diversity.
            **kwargs: Additional keyword arguments like top_p, top_k, max_length.
        """
        assert lex_diversity in [0, 20, 40, 60, 80, 100], "Lexical diversity must be one of 0, 20, 40, 60, 80, 100."
        assert order_diversity in [0, 20, 40, 60, 80, 100], "Order diversity must be one of 0, 20, 40, 60, 80, 100."

        lex_code = int(100 - lex_diversity)
        order_code = int(100 - order_diversity)

        input_text = " ".join(input_text.split())
        sentences = sent_tokenize(input_text)
        prefix = " ".join(prefix.replace("\n", " ").split())
        output_text = ""

        for sent_idx in range(0, len(sentences), sent_interval):
            curr_sent_window = " ".join(sentences[sent_idx:sent_idx + sent_interval])
            final_input_text = f"lexical = {lex_code}, order = {order_code}"
            if prefix:
                final_input_text += f" {prefix}"
            final_input_text += f" <sent> {curr_sent_window} </sent>"

            final_input = self.tokenizer([final_input_text], return_tensors="pt")
            final_input = {k: v.cuda() for k, v in final_input.items()}

            with torch.inference_mode():
                outputs = self.model.generate(**final_input, **kwargs)
            outputs = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
            prefix += " " + outputs[0]
            output_text += " " + outputs[0]

        return output_text

if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--paraphrase_model", type=str, default="DipperParaphraser")
    args.add_argument("--fold", type=int, default=0)
    args.add_argument("--input_path", type=str, default="/home/eilia/nlp1/part3/data/cross_validation_sets.pkl")
    args.add_argument("--output_path", type=str, default="/home/eilia/nlp1/part3/data/paraphrases")
    args.add_argument("--lex_diversity", type=int, default=40)
    args.add_argument("--order_diversity", type=int, default=20)
    args.add_argument("--top_p", type=float, default=0.75)
    args.add_argument("--top_k", type=int, default=None)
    args.add_argument("--max_length", type=int, default=512)
    args.add_argument("--do_sample", type=bool, default=True)
    args.add_argument("--num_samples_per_class", type=int, default=100)
    args.add_argument("--seed", type=int, default=42)
    args = args.parse_args()

    output_path = os.path.join(args.output_path, args.paraphrase_model, str(args.fold))
    os.makedirs(output_path, exist_ok=True)

    if args.paraphrase_model == "DipperParaphraser":
        dp = DipperParaphraser()
    else:
        raise ValueError(f"Paraphrase model {args.paraphrase_model} not supported")

    # Load cross validation sets
    with open(args.input_path, "rb") as f:
        sets = pickle.load(f)

    set_validation = sets[0][args.fold]["validation"]
    paraphrases = []
    gen_text = []

    #Sample 100 sentences from each class for the purposes of this analysis
    unique_labels = set(label for sentence, label in set_validation)
    random.seed(args.seed)
    
    for label in unique_labels:
        #We are sampling num_samples_per_class sentences from each class to paraphrase, for our ablation study.
        idx_class = [idx for idx, (sentence, l) in enumerate(set_validation) if l == label]
        idx_class_sampled = random.sample(idx_class, args.num_samples_per_class)
        
        for idx in idx_class_sampled:
            sentence = set_validation[idx][0]
            label = set_validation[idx][1]
            paraphrased_text = dp.paraphrase(sentence, lex_diversity=args.lex_diversity, order_diversity=args.order_diversity, do_sample=args.do_sample, top_p=args.top_p, top_k=args.top_k, max_length=args.max_length)        
            paraphrases.append({"sentence": sentence, "paraphrased_sentence": paraphrased_text, "label": label, "original_order_idx": idx})
            gen_text.append((paraphrased_text, label))
            
            if len(paraphrases) % 100 == 0:
                with open(f"{output_path}/paraphrases.json", "w") as f:
                    json.dump(paraphrases, f)
                with open(f"{output_path}/paraphrases_for_generation.pkl", "wb") as f:
                    pickle.dump(gen_text, f)
    
    with open(f"{output_path}/paraphrases.json", "w") as f:
        json.dump(paraphrases, f)
    with open(f"{output_path}/paraphrases_for_generation.pkl", "wb") as f:
        pickle.dump(gen_text, f)

    #Save experiments metadata
    metadata = {
        "paraphrase_model": args.paraphrase_model,
        "fold": args.fold,
        "lex_diversity": args.lex_diversity,
        "order_diversity": args.order_diversity,
        "top_p": args.top_p,
        "top_k": args.top_k,
        "max_length": args.max_length,
        "do_sample": args.do_sample,
        "num_samples_per_class": args.num_samples_per_class
    }
    with open(f"{output_path}/metadata.json", "w") as f:
        json.dump(metadata, f)