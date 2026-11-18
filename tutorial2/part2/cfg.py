import argparse

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train recurrent models on Bias in Bios."
    )

    # Model and embeddings
    parser.add_argument(
        "--model",
        nargs="+",
        choices=["gru", "lstm"],
        default=["gru"],
        help="One or more recurrent architectures.",
    )

    parser.add_argument(
        "--embedding",
        nargs="+",
        choices=["glove", "fasttext", "numberbatch"],
        default=["glove"],
        help="One or more pretrained embedding models.",
    )

    # Architecture
    parser.add_argument("--hidden-size", type=int, default=128)
    parser.add_argument("--num-layers", type=int, default=1)
    parser.add_argument("--bidirectional", action="store_true")
    parser.add_argument("--dropout", type=float, default=0.2)

    parser.add_argument(
        "--train-embeddings",
        action="store_true",
        help="Fine-tune pretrained embeddings during training.",
    )

    parser.add_argument(
        "--projection-dim",
        type=int,
        default=0,
        help="Optional embedding projection dimension. 0 disables it.",
    )

    # Data
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument(
        "--max-vocab-size",
        type=int,
        default=0,
        help="Maximum vocabulary size. 0 means unlimited.",
    )
    parser.add_argument("--num-classes", type=int, default=28)

    # Training
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument(
        "--optimizer",
        choices=["adam", "adamw", "adadelta"],
        default="adam",
    )
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--num-workers", type=int, default=0)

    # Experiment management
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-runs", type=int, default=3)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing experiment results.",
    )

    return parser.parse_args()