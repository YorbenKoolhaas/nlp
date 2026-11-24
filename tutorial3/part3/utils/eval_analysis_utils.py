import os
import json
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from datasets import load_dataset


def evaluate_predictions(output_path, model_name, instruction_type, print_results=True):
    """A function that processes the predictions given the path you have used to save 
    the predictions and the model name and instruction type you have used,
    and returns the classification scores for each assessed fold"""
    predictions = os.listdir(output_path)
    predictions = [p for p in predictions if (model_name + "_") in p and instruction_type in p]

    #Load predictions
    predictions_list = [(p.split("_")[-1].replace(".json", ""),json.load(open(f"{output_path}/{p}", "r"))) for p in predictions]
    
    dict_performance = {}
    p = []
    for i in range(len(predictions_list)):
        fold, predictions = predictions_list[i]
        p.append(predictions)

        preds = []
        gold = []
        for v in predictions.values():
            preds.append(v["predicted_label"])
            gold.append(v["label"])

        dict_performance[fold] = {
            "accuracy": accuracy_score(gold, preds),
            "precision": precision_score(gold, preds, average="macro"),
            "recall": recall_score(gold, preds, average="macro"),
            "f1": f1_score(gold, preds, average="macro")
        }
    
    if print_results:
        for fold, result in dict_performance.items():
            print(f"Fold {fold}:")
            print(f"Accuracy: {result['accuracy']}")
            print(f"Precision: {result['precision']}")
            print(f"Recall: {result['recall']}")
            print(f"F1: {result['f1']}")
            print("--------------------------------")
        
    return dict_performance

