from __future__ import annotations

import copy
import json
import random
from pathlib import Path
from urllib.request import urlretrieve

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score
from tqdm.auto import tqdm

import clslib

from cfg import parse_args
from dataset import collate_biographies, prepare_data
from model import RecurrentClassifier

EMB_DIR = Path.home() / ".cache" / "embeddings"
RESULT_DIR = Path("results")

EMBEDDINGS = {
    "glove": (
        "https://downloads.cs.stanford.edu/nlp/data/wordvecs/"
        "glove.2024.wikigiga.300d.zip",
        300,
        False,
    ),
    "fasttext": (
        "https://dl.fbaipublicfiles.com/fasttext/vectors-english/"
        "wiki-news-300d-1M.vec.zip",
        300,
        True,
    ),
    "numberbatch": (
        "https://conceptnet.s3.amazonaws.com/downloads/2019/"
        "numberbatch/numberbatch-en-19.08.txt.gz",
        300,
        True,
    ),
}

# ---------------------------------------------------------------------
# Seeding
# ---------------------------------------------------------------------

def seed_all(seed: int = 42):
    """Seed all random number generators."""

    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    # Prefer deterministic operations when available.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except TypeError:
        # Compatibility with older PyTorch versions.
        torch.use_deterministic_algorithms(True)

# ---------------------------------------------------------------------
# Embeddings
# ---------------------------------------------------------------------

def load_embeddings(name: str):
    """Load one of the supported pretrained embedding models."""

    if name not in EMBEDDINGS:
        raise ValueError(
            f"Unknown embedding '{name}'. "
            f"Choose from {list(EMBEDDINGS)}."
        )

    url, dims, skip_first_line = EMBEDDINGS[name]

    EMB_DIR.mkdir(parents=True, exist_ok=True)
    path = EMB_DIR / url.split("/")[-1]

    if not path.is_file():
        print(f"Downloading {name} embeddings...")
        urlretrieve(url, path)

    return clslib.PretrainedEmbeddings(
        str(path),
        dims,
        skip_first_line=skip_first_line,
    )


# ---------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------

@torch.no_grad()
def evaluate_model(model, dataloader, device, loss_module=None):
    """Evaluate a model and return loss, accuracy and macro-F1."""

    model.eval()

    all_predictions = []
    all_targets = []

    total_loss = 0.0
    total_samples = 0

    for tokens, lengths, targets in dataloader:
        tokens = tokens.to(device)
        targets = targets.to(device)

        predictions = model(tokens, lengths)

        if loss_module is not None:
            loss = loss_module(predictions, targets)
            total_loss += loss.item() * len(targets)

        predicted_classes = predictions.argmax(dim=1)

        all_predictions.extend(
            predicted_classes.cpu().tolist()
        )
        all_targets.extend(targets.cpu().tolist())

        total_samples += len(targets)

    metrics = {
        "accuracy": accuracy_score(
            all_targets,
            all_predictions,
        ),
        "macro_f1": f1_score(
            all_targets,
            all_predictions,
            average="macro",
            zero_division=0,
        ),
    }

    if loss_module is not None:
        metrics["loss"] = total_loss / total_samples

    return metrics


# ---------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------

def train_one_run(
    args,
    model_class,
    embedding_matrix,
    datasets,
    seed,
):
    """Train and evaluate one configuration with one random seed."""

    seed_all(seed)

    device = args.device

    generator = torch.Generator()
    generator.manual_seed(seed)

    train_loader = DataLoader(
        datasets["train"],
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_biographies,
        generator=generator,
        num_workers=args.num_workers,
        pin_memory=device == "cuda",
    )

    val_loader = DataLoader(
        datasets["dev"],
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_biographies,
        num_workers=args.num_workers,
        pin_memory=device == "cuda",
    )

    test_loader = DataLoader(
        datasets["test"],
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_biographies,
        num_workers=args.num_workers,
        pin_memory=device == "cuda",
    )

    model = RecurrentClassifier(
        embedding_matrix=embedding_matrix,
        model=model_class,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        bidirectional=args.bidirectional,
        dropout=args.dropout,
        train_embeddings=args.train_embeddings,
        projection_dim=args.projection_dim,
        num_classes=args.num_classes,
    ).to(device)

    loss_module = nn.CrossEntropyLoss()

    if args.optimizer == "adam":
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=args.lr,
            weight_decay=args.weight_decay,
        )
    elif args.optimizer == "adamw":
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=args.lr,
            weight_decay=args.weight_decay,
        )
    elif args.optimizer == "adadelta":
        optimizer = torch.optim.Adadelta(
            model.parameters(),
            lr=args.lr,
            weight_decay=args.weight_decay,
        )
    else:
        raise ValueError(f"Unknown optimizer: {args.optimizer}")

    history = {
        "training_loss": [],
        "validation_accuracy": [],
        "validation_macro_f1": [],
    }

    best_val_f1 = -1.0
    best_epoch = 0
    best_state = None

    epoch_bar = tqdm(range(args.epochs), desc="Epochs")

    for epoch in epoch_bar:
        model.train()

        total_loss = 0.0
        total_samples = 0

        for tokens, lengths, targets in train_loader:
            tokens = tokens.to(device)
            targets = targets.to(device)

            optimizer.zero_grad(set_to_none=True)

            predictions = model(tokens, lengths)
            loss = loss_module(predictions, targets)

            loss.backward()

            if args.grad_clip > 0:
                nn.utils.clip_grad_norm_(
                    model.parameters(),
                    args.grad_clip,
                )

            optimizer.step()

            total_loss += loss.item() * len(targets)
            total_samples += len(targets)

        avg_train_loss = total_loss / total_samples

        val_metrics = evaluate_model(
            model,
            val_loader,
            device,
            loss_module,
        )

        history["training_loss"].append(avg_train_loss)
        history["validation_accuracy"].append(
            val_metrics["accuracy"]
        )
        history["validation_macro_f1"].append(
            val_metrics["macro_f1"]
        )

        # Keep the best validation checkpoint.
        if val_metrics["macro_f1"] > best_val_f1:
            best_val_f1 = val_metrics["macro_f1"]
            best_epoch = epoch + 1
            best_state = copy.deepcopy(model.state_dict())

        epoch_bar.set_postfix({
            "train_loss": f"{avg_train_loss:.4f}",
            "val_acc": f"{val_metrics['accuracy']:.4f}",
            "val_f1": f"{val_metrics['macro_f1']:.4f}",
        })

    # Test using the best validation checkpoint.
    model.load_state_dict(best_state)

    test_metrics = evaluate_model(
        model,
        test_loader,
        device,
        loss_module,
    )

    return {
        "seed": seed,
        "best_epoch": best_epoch,
        "best_val_macro_f1": best_val_f1,
        "test": test_metrics,
        "history": history,
    }, best_state


# ---------------------------------------------------------------------
# Experiment runner
# ---------------------------------------------------------------------

def normalize_models(models):
    if not isinstance(models, (list, tuple)):
        models = [models]

    normalized = []

    for model in models:
        if isinstance(model, str):
            model = {
                "gru": nn.GRU,
                "lstm": nn.LSTM,
            }[model.lower()]

        if model not in (nn.GRU, nn.LSTM):
            raise ValueError(f"Unsupported recurrent model: {model}")

        normalized.append(model)

    return normalized


def normalize_embeddings(embeddings):
    if isinstance(embeddings, str):
        embeddings = [embeddings]

    embeddings = [name.lower() for name in embeddings]

    invalid = set(embeddings) - set(EMBEDDINGS)

    if invalid:
        raise ValueError(f"Unsupported embeddings: {invalid}")

    return embeddings


def train(args):
    """Run all model/embedding combinations and repeated seeds."""

    models = normalize_models(args.model)
    embedding_names = normalize_embeddings(args.embedding)

    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    if args.save_model: (RESULT_DIR / "models").mkdir(parents=True, exist_ok=True)

    all_results = {}

    for embedding_name in embedding_names:
        print(f"\n{'=' * 60}")
        print(f"Loading {embedding_name}")
        print(f"{'=' * 60}")

        pretrained = load_embeddings(embedding_name)

        datasets, embedding_matrix, vocab_size = prepare_data(
            embedding_name,
            pretrained,
            max_length=args.max_length,
            max_vocab_size=args.max_vocab_size,
        )

        print(
            f"Vocabulary size: {vocab_size:,} | "
            f"Embedding shape: {tuple(embedding_matrix.shape)}"
        )

        for model_class in models:
            model_name = model_class.__name__.lower()

            config = {
                "model": model_name,
                "embedding": embedding_name,
                "hidden_size": args.hidden_size,
                "num_layers": args.num_layers,
                "bidirectional": args.bidirectional,
                "dropout": args.dropout,
                "train_embeddings": args.train_embeddings,
                "projection_dim": args.projection_dim,
                "batch_size": args.batch_size,
                "lr": args.lr,
                "weight_decay": args.weight_decay,
                "optimizer": args.optimizer,
                "epochs": args.epochs,
                "max_length": args.max_length,
                "max_vocab_size": args.max_vocab_size,
                "num_classes": args.num_classes,
                "grad_clip": args.grad_clip,
                "seed": args.seed,
                "num_runs": args.num_runs,
            }

            # Generate a descriptive, filesystem-safe filename.
            filename = (
                f"{model_name}_{embedding_name}"
                f"_h{args.hidden_size}"
                f"_l{args.num_layers}"
                f"_bi{int(args.bidirectional)}"
                f"_drop{args.dropout}"
                f"_embtrain{int(args.train_embeddings)}"
                f"_proj{args.projection_dim}"
                f"_bs{args.batch_size}"
                f"_lr{args.lr}"
                f"_opt{args.optimizer}"
                f"_s{args.seed}"
                f"_n{args.num_runs}.json"
            )

            result_path = RESULT_DIR / filename

            if result_path.exists() and not args.overwrite:
                print(f"Skipping existing result: {result_path}")
                continue

            print(f"\nRunning {model_name} with {embedding_name}")


            model_path = (RESULT_DIR / "models" / filename).with_suffix(".pt")

            best_model_state = None
            overall_best_macro = 0
            experiment_results = []

            for run_idx in range(args.num_runs):
                # Same sequence of seeds for every configuration.
                # This makes comparisons across configurations paired.
                run_seed = args.seed + run_idx

                print(
                    f"\nRun {run_idx + 1}/{args.num_runs} "
                    f"(seed={run_seed})"
                )

                result, best_run_state = train_one_run(
                    args,
                    model_class,
                    embedding_matrix,
                    datasets,
                    run_seed,
                )

                # Save best model accross runs
                if args.save_model and result["best_val_macro_f1"] > overall_best_macro:
                    best_model_state = copy.deepcopy(best_run_state)

                experiment_results.append(result)

                # Save after every run, so interrupted experiments
                # do not lose completed results.
                payload = {
                    "config": config,
                    "results": experiment_results,
                }

                with open(result_path, "w") as f:
                    json.dump(payload, f, indent=4)

            # Aggregate metrics across seeds.
            test_accs = [
                r["test"]["accuracy"]
                for r in experiment_results
            ]

            test_f1s = [
                r["test"]["macro_f1"]
                for r in experiment_results
            ]

            summary = {
                "accuracy_mean": float(np.mean(test_accs)),
                "accuracy_std": float(np.std(test_accs, ddof=1))
                if len(test_accs) > 1 else 0.0,
                "macro_f1_mean": float(np.mean(test_f1s)),
                "macro_f1_std": float(np.std(test_f1s, ddof=1))
                if len(test_f1s) > 1 else 0.0,
            }

            payload["summary"] = summary

            with open(result_path, "w") as f:
                json.dump(payload, f, indent=4)

            all_results[f"{model_name}_{embedding_name}"] = summary

            print("\nExperiment summary:")
            print(json.dumps(summary, indent=4))

            if args.save_model and best_model_state:
                torch.save(best_model_state, model_path)
                print(f"\nModel saved to {'/'.join(model_path.parts[-2:])}")

    return all_results

if __name__ == "__main__":
    args = parse_args()

    if torch.cuda.is_available():
        args.device = "cuda"
    elif torch.backends.mps.is_available():
        args.device = "mps"
    else:
        args.device = "cpu"

    print(f"Using device: {args.device}")

    results = train(args)

    print("\nAll experiments completed.")
    print(json.dumps(results, indent=4))