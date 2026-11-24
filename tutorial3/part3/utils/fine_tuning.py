import argparse
import random
import pickle
import os
import json

from datasets import Dataset
import torch
from transformers import AutoTokenizer,AutoModelForCausalLM, DataCollatorForLanguageModeling
from peft import LoraConfig, get_peft_model
from trl import SFTConfig, SFTTrainer

class FineTuning:
    def __init__(self, device: str, model_name: str, use_bf16: bool):
        self.device = device
        self.model_name = model_name
        self.use_bf16 = use_bf16

        self._load_tokenizer_and_model()
    
    def _load_tokenizer_and_model(self):
        # Load the tokenizer and model
        kwargs = {}
        if self.use_bf16 and torch.cuda.is_available():
            kwargs["torch_dtype"] = torch.bfloat16
        
        if self.model_name == "llama-3b-instruct":
            mid = "meta-llama/Llama-3.2-3B-Instruct"
        elif self.model_name == "qwen-0.5b-instruct":
            mid = "Qwen/Qwen2.5-0.5B-Instruct"
        else:
            raise ValueError(f"Model {self.model_name} not found")

        self.tokenizer = AutoTokenizer.from_pretrained(mid)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(mid, **kwargs)
        self.model.to(self.device)
      
    def load_lora_config(self, r: int, lora_alpha: int, lora_dropout: float, target_modules: list[str], bias: str, task_type: str):
        # Configure LoRA parameters
        self.lora_config = LoraConfig(
            r=r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            target_modules=target_modules,
            bias=bias,
            task_type=task_type,
        )

    def load_training_arguments(self,output_dir: str, num_train_epochs: int, per_device_train_batch_size: int, 
                                per_device_eval_batch_size: int, learning_rate: float, logging_steps: int, save_strategy: str, 
                                report_to: str, eval_steps: int, eval_strategy: str, save_steps: int):
        # Configure training parameters
        self.training_args = SFTConfig(
        output_dir=output_dir,
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=per_device_train_batch_size,
        per_device_eval_batch_size=per_device_eval_batch_size,
        learning_rate=learning_rate,
        eval_strategy=eval_strategy,
        eval_steps=eval_steps,
        save_strategy=save_strategy,
        save_steps=save_steps,
        logging_steps=logging_steps,
        report_to=report_to, 
        bf16=self.use_bf16, 
        disable_tqdm=False)

    def _prepare_peft_model(self):
        # Load a new LoRA adapter for each fold; unload any previous adapter without merging
        # so base weights stay as the original pretrained model.
        if hasattr(self.model, "unload"):
            self.model = self.model.unload()

        self.model = get_peft_model(self.model, self.lora_config)
        self.model.to(self.device)
        self.model.print_trainable_parameters()

    def _create_text_from_example(self, example):
        instruction = """You are a bot that predicts the category of a news article. 
        The options are: (A) World, (B) Sports, (C) Business, (D) Science/Technology. 
        Just respond with the letter of the category (A/B/C/D).\n\n"""

        #{0: "World", 1: "Sports", 2: "Business", 3: "Science/Technology"}
        map_label_to_topic = {0: "A", 1: "B", 2: "C", 3: "D"}
        topic = map_label_to_topic[example[1]]

        instance = f"Input: {example[0]}\nAnswer:"
        datapoint = [{"role": "system", "content": f"{instruction}"},
                    {"role": "user", "content": instance},
                    {"role": "assistant", "content": topic}]
        
        return datapoint
    
    def _format_chat(self, example):
        # Format the training and validation prompts
        return {
            "text": self.tokenizer.apply_chat_template(
                example["messages"],
                tokenize=False,
                add_generation_prompt=False,
            )
        }
    
    def _tokenize(self, example):
        # Tokenize without padding; DataCollatorForLanguageModeling pads dynamically per batch
        return self.tokenizer(example["text"], truncation=True, max_length=256, padding=False)

    def train(self, training_set: list[dict], validation_set: list[dict], fold: int, output_dir: str):
        self._prepare_peft_model()

        # Create the training and validation prompts
        training_set = [self._create_text_from_example(x) for x in training_set]
        validation_set = [self._create_text_from_example(x) for x in validation_set]

        training_set = {"messages": training_set}
        validation_set = {"messages": validation_set}

        training_dataset = Dataset.from_dict(training_set)
        validation_dataset = Dataset.from_dict(validation_set)

        training_dataset = training_dataset.map(self._format_chat)
        validation_dataset = validation_dataset.map(self._format_chat)

        tokenized_training_dataset = training_dataset.map(self._tokenize, batched=True, batch_size=16, remove_columns=training_dataset.column_names)
        tokenized_validation_dataset = validation_dataset.map(self._tokenize, batched=True, batch_size=16, remove_columns=validation_dataset.column_names)

        tokenized_training_dataset = tokenized_training_dataset.with_format("torch")
        tokenized_validation_dataset = tokenized_validation_dataset.with_format("torch")

        # Create the data collator
        data_collator = DataCollatorForLanguageModeling(tokenizer=self.tokenizer, mlm=False)

        # Train the LoRA adapter
        trainer = SFTTrainer(model=self.model, args=self.training_args, train_dataset=tokenized_training_dataset, 
                            eval_dataset=tokenized_validation_dataset, data_collator=data_collator, processing_class=self.tokenizer)

        trainer.train()

        # Save train/eval losses logged during training
        fold_dir = f"{output_dir}/{self.model_name}-fold-{fold}"
        os.makedirs(fold_dir, exist_ok=True)
        with open(f"{fold_dir}/trainer_log_history.json", "w") as f:
            json.dump(trainer.state.log_history, f, indent=2)

        # Save the LoRA adapter
        self.model.save_pretrained(f"{fold_dir}/lora-adapter")
        self.tokenizer.save_pretrained(f"{fold_dir}/lora-adapter")

        #Save meta-data with arguments
        meta_data = {}
        meta_data["train_config"] = self.training_args.to_dict()
        meta_data["lora_config"] = self.lora_config.__dict__
        meta_data["model_name"] = self.model_name
        meta_data["fold"] = fold

        with open(f"{fold_dir}/meta_data.json", "w") as f:
            json.dump(meta_data, f, indent=2, default=str)

if __name__ == "__main__":
    print("CUDA available:", torch.cuda.is_available())
    print("CUDA device:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)

    # Parse arguments
    args = argparse.ArgumentParser()
    args.add_argument("--fold", type=int, default=None)
    args.add_argument("--device", type=str, default="cuda")
    args.add_argument("--model_name", type=str, default="qwen-0.5b-instruct")
    args.add_argument("--input_path", type=str, default="/home/eilia/nlp1/part3/data/cross_validation_sets.pkl")
    args.add_argument("--use_bf16", action=argparse.BooleanOptionalAction, default=True)
    args.add_argument("--output_dir", type=str, default="/home/eilia/nlp1/part3/lora-models")
    args.add_argument("--seed", type=int, default=0)
    args.add_argument("--num_train_epochs", type=int, default=1)
    args.add_argument("--per_device_train_batch_size", type=int, default=64)
    args.add_argument("--per_device_eval_batch_size", type=int, default=64)
    args.add_argument("--learning_rate", type=float, default=2e-4)
    args.add_argument("--logging_steps", type=int, default=1000)
    args.add_argument("--save_strategy", type=str, default="epoch")
    args.add_argument("--report_to", type=str, default="none")
    args.add_argument("--eval_steps", type=int, default=1000)
    args.add_argument("--eval_strategy", type=str, default="steps")
    args.add_argument("--save_steps", type=int, default=1000)
    args.add_argument("--r", type=int, default=8)
    args.add_argument("--lora_alpha", type=int, default=16)
    args.add_argument("--lora_dropout", type=float, default=0.05)
    args.add_argument("--target_modules", nargs="+", default=["q_proj", "k_proj", "v_proj", "o_proj"])
    args.add_argument("--bias", type=str, default="none")
    args.add_argument("--task_type", type=str, default="CAUSAL_LM")
    args = args.parse_args()

    random.seed(args.seed)

    # Create the output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    ft = FineTuning(device=args.device, model_name=args.model_name, use_bf16=args.use_bf16)

    ft.load_lora_config(r=args.r, lora_alpha=args.lora_alpha, lora_dropout=args.lora_dropout, target_modules=args.target_modules, bias=args.bias, task_type=args.task_type)
    ft.load_training_arguments(output_dir=args.output_dir, num_train_epochs=args.num_train_epochs, per_device_train_batch_size=args.per_device_train_batch_size, per_device_eval_batch_size=args.per_device_eval_batch_size, learning_rate=args.learning_rate, logging_steps=args.logging_steps, save_strategy=args.save_strategy, report_to=args.report_to, eval_steps=args.eval_steps, eval_strategy=args.eval_strategy, save_steps=args.save_steps)

    
    # Load the cross validation sets
    with open(args.input_path, "rb") as f:
        cross_validation_sets = pickle.load(f)

    # pickle layout: [ {fold: {"training": ..., "validation": ...}, ...} ]
    folds = cross_validation_sets[0] if isinstance(cross_validation_sets, list) else cross_validation_sets

    def training_set_from_cross_validation_sets(folds: dict, fold: int, output_dir: str):
        training_set = folds[fold]["training"]
        validation_set = folds[fold]["validation"]

        ft.train(training_set=training_set, validation_set=validation_set, fold=fold, output_dir=output_dir)

    if args.fold is not None:
        training_set_from_cross_validation_sets(folds=folds, fold=args.fold, output_dir=args.output_dir)
    else:
        for fold in folds.keys():
            training_set_from_cross_validation_sets(folds=folds, fold=fold, output_dir=args.output_dir)

        

        
        
        