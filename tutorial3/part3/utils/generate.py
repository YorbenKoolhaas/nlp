# Import relevant packages
import torch
import pickle
import argparse
import os
import json
import random
from transformers import AutoTokenizer, AutoModelForCausalLM, set_seed
from peft import PeftModel
from tqdm import tqdm

class Generation:
    def __init__(self, model_name: str, device: str, 
                max_new_tokens: int = 300, use_bf16: bool = True, adapter_path: str = None):
        
        self.device = device
        self.max_new_tokens = max_new_tokens
        self.adapter_path = adapter_path
        self.tokenizer, self.model = self._load_tokenizer_and_model(
            model_name, use_bf16=use_bf16 and self.device == "cuda", adapter_path=adapter_path
        )
        self.model.eval()

    def _load_tokenizer_and_model(self, model_name: str, use_bf16: bool, adapter_path: str):
        kwargs = {}
        if use_bf16 and torch.cuda.is_available():
            kwargs["torch_dtype"] = torch.bfloat16
        
        if model_name == "llama-3b-instruct":
            mid = "meta-llama/Llama-3.2-3B-Instruct"
        elif model_name == "qwen-0.5b-instruct":
            mid = "Qwen/Qwen2.5-0.5B-Instruct"
        elif model_name == "qwen-0.5b-instruct-ft-lora":
            mid = "Qwen/Qwen2.5-0.5B-Instruct"
        else:
            raise ValueError(f"Model {model_name} not found")

        if "ft-lora" in model_name:
            adapter_path = self.adapter_path
            base_model = AutoModelForCausalLM.from_pretrained(mid, **kwargs)
            model = PeftModel.from_pretrained(base_model, adapter_path)
            tokenizer = AutoTokenizer.from_pretrained(adapter_path)
        else:       
            tokenizer = AutoTokenizer.from_pretrained(mid)
            model = AutoModelForCausalLM.from_pretrained(mid, **kwargs)

        # Left-padding is required for correct batched generation with decoder-only LMs
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "left"
        
        model.to(self.device)
        return tokenizer, model

    def _prompt_ids(self, question: str, instruction: str) -> list:
        messages = [
            {"role": "system", "content": instruction},
            {"role": "user", "content": f"Input: {question}\nAnswer:"},
        ]
        t = self.tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=True
        )
        if isinstance(t, torch.Tensor):
            t = t.tolist()
        if isinstance(t, dict):
            t = t["input_ids"]
        # apply_chat_template may return a nested batch of size 1
        if len(t) > 0 and isinstance(t[0], list):
            t = t[0]
        return t

    def _inputs(self, question: str, instruction: str) -> torch.Tensor:
        return torch.tensor([self._prompt_ids(question, instruction)], device=self.device)

    def _batch_inputs(self, questions: list, instruction: str):
        input_ids_list = [self._prompt_ids(q, instruction) for q in questions]
        padded = self.tokenizer.pad(
            {"input_ids": input_ids_list},
            padding=True,
            return_tensors="pt",
        )
        return (
            padded["input_ids"].to(self.device),
            padded["attention_mask"].to(self.device),
        )

    def _gen_kw(self):
        return {
            "max_new_tokens": self.max_new_tokens,
            "num_beams": 1,
            "pad_token_id": self.tokenizer.pad_token_id,
            "use_cache": True,
        }

    def _decode_new_tokens(self, outputs: torch.Tensor, input_len: int):
        generated_tokens = outputs[:, input_len:]
        generated_strings = self.tokenizer.batch_decode(
            generated_tokens, skip_special_tokens=True
        )
        return generated_tokens, generated_strings

    def generate_greedy(self, question: str, instruction: str):
        generated_tokens, generated_strings = self.generate_greedy_batch(
            [question], instruction
        )
        return generated_tokens, generated_strings[0]

    def generate_greedy_batch(self, questions: list, instruction: str):
        input_ids, attention_mask = self._batch_inputs(questions, instruction)
        with torch.inference_mode():
            outputs = self.model.generate(
                input_ids,
                attention_mask=attention_mask,
                do_sample=False,
                temperature=0.0,
                top_k=0,
                **self._gen_kw(),
            )
        return self._decode_new_tokens(outputs, input_ids.shape[-1])

    def generate_samples(self, question: str, instruction: str, num_return_sequences: int = 1, seed: int = 42, 
                         temperature: float = 1.0, top_p: float = 1.0):
        generated_tokens, generated_strings = self.generate_samples_batch(
            [question],
            instruction,
            num_return_sequences=num_return_sequences,
            seed=seed,
            temperature=temperature,
            top_p=top_p,
        )
        return generated_tokens, generated_strings

    def generate_samples_batch(
        self,
        questions: list,
        instruction: str,
        num_return_sequences: int = 1,
        seed: int = 42,
        temperature: float = 1.0,
        top_p: float = 1.0,
    ):
        torch.manual_seed(seed)
        set_seed(seed)

        input_ids, attention_mask = self._batch_inputs(questions, instruction)
        with torch.inference_mode():
            outputs = self.model.generate(
                input_ids,
                attention_mask=attention_mask,
                do_sample=True,
                temperature=temperature,
                top_p=top_p,
                num_return_sequences=num_return_sequences,
                **self._gen_kw(),
            )

        generated_tokens, generated_strings = self._decode_new_tokens(
            outputs, input_ids.shape[-1]
        )
        # Flatten to one string per prompt when num_return_sequences == 1
        if num_return_sequences == 1:
            return generated_tokens, generated_strings

        # Group strings per prompt: [s0_r0, s0_r1, ..., s1_r0, ...]
        grouped = []
        for i in range(len(questions)):
            start = i * num_return_sequences
            grouped.append(generated_strings[start : start + num_return_sequences])
        return generated_tokens, grouped

def parse_generated_string(generated_string: str) -> int:
    if "A" in generated_string:
        return 0
    elif "B" in generated_string:
        return 1
    elif "C" in generated_string:
        return 2
    elif "D" in generated_string:
        return 3
    else:
        return -1


if __name__ == "__main__":
    # Parse arguments
    args = argparse.ArgumentParser()
    args.add_argument("--fold", type=int, default=None)
    args.add_argument("--instruction_type", type=str, default="0-shot", choices=["0-shot", "few-shot"])
    args.add_argument("--device", type=str, default="cuda")
    args.add_argument("--model_name", type=str, default="qwen-0.5b-instruct", choices=["qwen-0.5b-instruct", "llama-3b-instruct", "qwen-0.5b-instruct-ft-lora"])
    args.add_argument("--input_path", type=str, default="/home/eilia/nlp1/part3/data/cross_validation_sets.pkl")
    args.add_argument("--output_path", type=str, default="/home/eilia/nlp1/part3/predictions")
    args.add_argument("--seed", type=int, default=0)
    args.add_argument("--adapter_path", type=str, default=None)
    args.add_argument("--num_examples_per_class", type=int, default=1)
    args.add_argument("--batch_size", type=int, default=16)
    args = args.parse_args()

    random.seed(args.seed)

    # Load cross validation sets
    with open(args.input_path, "rb") as f:
        sets = pickle.load(f)
    
    original_model_name = args.model_name.split("-ft")[0]

    instruction = """You are a bot that predicts the category of a news article. 
            The options are: (A) World, (B) Sports, (C) Business, (D) Science/Technology. 
            Just respond with the letter of the category (A/B/C/D)."""
    if args.instruction_type == "few-shot":
        # We use 1 sampled instance from each class from the training set to create the few-shot examples
        examples_folds = {}
        for fold, data in sets[0].items():
            #We randomly choose examples from each class from the training set to create the few-shot examples
            examples_text = "Here are some examples of inputs and outputs:\n"
            examples = data["training"]
            
            #Sample 1 example from each class
            gold_labels = [x[1] for x in examples]
            unique_labels = list(set(gold_labels))
            # Match instruction / user turn / fine-tuning: Answer: + letter A/B/C/D
            map_label_to_letter = {0: "A", 1: "B", 2: "C", 3: "D"}
            for l in unique_labels:
                index_for_class = [i for i, x in enumerate(gold_labels) if x == l]   
                chosen_index = random.sample(index_for_class, args.num_examples_per_class)
                letter = map_label_to_letter[l]
                examples_text += f"Input: {examples[chosen_index[0]][0]}\nAnswer: {letter}\n"
            
            examples_folds[fold] = instruction + examples_text
            print(examples_folds)
    elif args.instruction_type != "0-shot":
        raise ValueError(f"Instruction type {args.instruction_type} not found")

    os.makedirs(args.output_path, exist_ok=True)

    def generate_predictions(gen, data, save_data_dict, fold: int, instruction: str, seed: int, batch_size: int):
        save_data_dict[fold] = {}
        data = list(data)
        n = len(data)

        # Pass 1: batched greedy decoding
        for start in tqdm(range(0, n, batch_size), desc=f"fold {fold} greedy"):
            batch = data[start : start + batch_size]
            sentences = [s for s, _ in batch]
            labels = [l for _, l in batch]

            _, generated_strings = gen.generate_greedy_batch(sentences, instruction)

            for offset, (sentence, label, generated_string) in enumerate(
                zip(sentences, labels, generated_strings)
            ):
                idx = start + offset
                save_data_dict[fold][idx] = {
                    "sentence": sentence,
                    "generated_string": generated_string,
                    "predicted_label": parse_generated_string(generated_string),
                    "label": label,
                }

        # Pass 2: batched sampling retries for invalid predictions
        for try_count in range(3):
            failed_idxs = [
                i for i in range(n)
                if save_data_dict[fold][i]["predicted_label"] == -1
            ]
            if not failed_idxs:
                break

            for start in tqdm(
                range(0, len(failed_idxs), batch_size),
                desc=f"fold {fold} retry {try_count + 1}",
            ):
                batch_idxs = failed_idxs[start : start + batch_size]
                sentences = [save_data_dict[fold][i]["sentence"] for i in batch_idxs]

                _, generated_strings = gen.generate_samples_batch(
                    sentences,
                    instruction,
                    num_return_sequences=1,
                    seed=(seed + try_count),
                )

                for i, generated_string in zip(batch_idxs, generated_strings):
                    save_data_dict[fold][i]["generated_string"] = generated_string
                    save_data_dict[fold][i]["predicted_label"] = parse_generated_string(
                        generated_string
                    )
        
        #Save the predictions for fold
        with open(os.path.join(args.output_path, f"{args.model_name}_{args.instruction_type}_{fold}.json"), "w") as f:
            # JSON keys must be strings
            json.dump({str(k): v for k, v in save_data_dict[fold].items()}, f, indent=4)

    predictions = {}

    def resolve_adapter_path(fold):
        # Base models ignore adapters; only FT LoRA needs a fold-specific path.
        if "ft-lora" not in args.model_name:
            return None
        if args.adapter_path is None:
            raise ValueError("--adapter_path is required when using a fine-tuned LoRA model")
        return f"{args.adapter_path}/{original_model_name}-fold-{fold}/lora-adapter"

    def free_generation(gen):
        # Drop model/tokenizer refs so the next fold can load without OOM.
        if gen is None:
            return
        del gen.model
        del gen.tokenizer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if "paraphrases" in args.input_path:
        #For the purposes of the paraphrasing analysis experiment
        fold = args.input_path.split("/")[-2]
        
        # Initialize generation (use fold from input path, same as predictions key)
        gen = Generation(
            args.model_name, device=args.device, adapter_path=resolve_adapter_path(fold)
        )

        if args.instruction_type == "few-shot":
            instruction = examples_folds[fold]
        else:
            instruction = instruction
        generate_predictions(gen, sets, predictions, fold, instruction, seed=args.seed, batch_size=args.batch_size)
    else:
        if args.fold is None:
            # pickle layout: [ {fold: {"training": ..., "validation": ...}, ...} ]
            gen = None
            for fold, data in tqdm(sets[0].items()):
                # FT LoRA: one adapter per fold. Base model: load once and reuse.
                if gen is None or "ft-lora" in args.model_name:
                    free_generation(gen)
                    gen = Generation(
                        args.model_name,
                        device=args.device,
                        adapter_path=resolve_adapter_path(fold),
                    )

                if args.instruction_type == "few-shot":
                    fold_instruction = examples_folds[fold]
                else:
                    fold_instruction = instruction
                generate_predictions(
                    gen, data["validation"], predictions, fold, fold_instruction,
                    seed=args.seed, batch_size=args.batch_size,
                )
            free_generation(gen)
        else:
            if args.instruction_type == "few-shot":
                instruction = examples_folds[args.fold]
            else:
                instruction = instruction
            
            # Initialize generation
            gen = Generation(
                args.model_name,
                device=args.device,
                adapter_path=resolve_adapter_path(args.fold),
            )

            generate_predictions(
                gen, sets[0][args.fold]["validation"], predictions, args.fold, instruction,
                seed=args.seed, batch_size=args.batch_size,
            )
