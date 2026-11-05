from scipy.sparse import csr_matrix
from sklearn.linear_model import LogisticRegression
import pickle


class LogLinearModel:
    """
    A wrapper for sklearn's LogisticRegression to be used in our codebase.
    """
    def load(path):
        """
        Load a logistic model from {path}.pickle and return it.
        """
        with open(f"{path}.pickle", 'rb') as f:
            logistic_model = pickle.load(f)
        return logistic_model

    def fit(training_set, encoder, save_to=None, max_iter=100, n_jobs=-1):
        """
        Constructs, trains and returns a softmax regressor.
        
        training_set: list of text-label pairs (a text is a string, a label is a 0-based integer)
        encoder: an instance of the Encoder class (such as TokenIndicator)    
        save_to: (str) if given, save the model to {save_to}.pickle
        """
        # Convert the training and validation sets to vectors
        training_vectors = [encoder(text) for text, _ in training_set]
        training_labels = [label for _, label in training_set]

        #To train the model more efficiently, we transform the training vectors to a sparse matrix
        X_train_sparse = csr_matrix(training_vectors)

        # Train the logistic model
        logistic_model = LogisticRegression(solver="saga", max_iter=max_iter, n_jobs=n_jobs) #sparse matrix frinedly solver 
        logistic_model.fit(X_train_sparse, training_labels)

        if save_to is not None:        
            with open(f"{save_to}.pickle", 'wb') as f:
                pickle.dump(logistic_model, f, protocol=pickle.HIGHEST_PROTOCOL)
            
        return logistic_model        

    def predict(texts, model: LogisticRegression, encoder):
        """
        Predicts labels (via the most probable class decision rule) for a collection of texts. 
        (scikit-learn hard-codes the most probable class as the decision rule of choice)
        
        texts: a list of texts, each a string
        model: a trained instance of sklearn's LogisticRegression
        encoder: the Encoder used for in the model's training
        """
        validation_vectors = [encoder(text) for text in texts]
        X_val_sparse = csr_matrix(validation_vectors)
        predictions = model.predict(X_val_sparse)
        return predictions

    def predict_probs(texts, model: LogisticRegression, encoder):
        """
        Predicts labels (via the most probable class decision rule) for a collection of texts. 
        (scikit-learn hard-codes the most probable class as the decision rule of choice)
        
        texts: a list of texts, each a string
        model: a trained instance of sklearn's LogisticRegression
        encoder: the Encoder used for in the model's training
        """
        validation_vectors = [encoder(text) for text in texts]
        X_val_sparse = csr_matrix(validation_vectors)
        probs = model.predict_proba(X_val_sparse)
        return probs


from torch.utils.data import Dataset
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
import time
import scipy
import scipy.sparse
import json
import numpy as np


class SparseMLP(nn.Module):
    """One-hidden-layer MLP that accepts sparse input tensors.

    Inputs: feature vectors of size `input_dim` (e.g. bag-of-words).
    Process: sparse matmul into a dense hidden layer, ReLU + dropout, linear (num_classes) and log-softmax outputs
        log-softmax, rather than softmax, for numerical stability
    Output: logprobs over `num_classes`.
    """

    def __init__(self, input_dim, hidden_dim, num_classes, dropout=0.2):
        """
        input_dim: size of the input feature space (e.g. vocabulary size)
        hidden_dim: width of the hidden layer
        num_classes: number of output labels
        dropout: fraction of hidden units zeroed during training (regularisation)
        """
        super().__init__()

        # First-layer weights stay dense; inputs arriving here are sparse.
        self.weight = nn.Parameter(
            torch.empty(input_dim, hidden_dim)
        )
        self.bias = nn.Parameter(
            torch.zeros(hidden_dim)
        )

        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

        # Everything after the first layer is dense.
        self.output = nn.Sequential(
            nn.Linear(hidden_dim, num_classes),            
            nn.LogSoftmax(dim=-1) # for numerical stability, it is common to use log softmax, rather than softmax
        )

        nn.init.xavier_uniform_(self.weight) # why xavier?

    def forward(self, x):
        """Map a sparse batch [batch, input_dim] to class log-probabilities [batch, num_classes]."""
        # First matmul keeps x sparse; result is dense from here on.
        x = torch.sparse.mm(x, self.weight)
        x = x + self.bias

        x = self.activation(x)
        x = self.dropout(x)
        # logprobs
        return self.output(x)


class DeepSparseMLP(nn.Module):
    """Modified MLP: same sparse first layer as SparseMLP, followed by extra dense hidden layers.

    Inputs: feature vectors of size `input_dim` (e.g. bag-of-words).
    Process: sparse matmul into the first hidden layer, then for every extra hidden layer:
        Linear -> LayerNorm -> ReLU -> dropout, and finally linear (num_classes) and log-softmax outputs
    Output: logprobs over `num_classes`.
    """

    def __init__(self, input_dim, hidden_dims, num_classes, dropout=0.2):
        """
        input_dim: size of the input feature space (e.g. vocabulary size)
        hidden_dims: widths of the hidden layers, e.g. [256, 128] for two hidden layers
        num_classes: number of output labels
        dropout: fraction of hidden units zeroed during training, applied after every hidden layer
        """
        super().__init__()
        hidden_dims = list(hidden_dims)

        # First layer is identical to SparseMLP: dense weights, sparse inputs.
        # Named `weight` so predict_probs can find the device via model.weight.device.
        self.weight = nn.Parameter(
            torch.empty(input_dim, hidden_dims[0])
        )
        self.bias = nn.Parameter(
            torch.zeros(hidden_dims[0])
        )
        self.first_norm = nn.LayerNorm(hidden_dims[0])
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

        # Extra dense hidden layers; LayerNorm keeps activations on a stable scale as depth grows.
        layers = []
        for d_in, d_out in zip(hidden_dims[:-1], hidden_dims[1:]):
            layers += [nn.Linear(d_in, d_out), nn.LayerNorm(d_out), nn.ReLU(), nn.Dropout(dropout)]
        self.hidden = nn.Sequential(*layers)

        self.output = nn.Sequential(
            nn.Linear(hidden_dims[-1], num_classes),
            nn.LogSoftmax(dim=-1)
        )

        nn.init.xavier_uniform_(self.weight)

    def forward(self, x):
        """Map a sparse batch [batch, input_dim] to class log-probabilities [batch, num_classes]."""
        x = torch.sparse.mm(x, self.weight)
        x = x + self.bias

        x = self.first_norm(x)
        x = self.activation(x)
        x = self.dropout(x)

        x = self.hidden(x)
        # logprobs
        return self.output(x)


def scipy_to_torch_sparse(x):
    """Convert a SciPy sparse matrix to a coalesced PyTorch sparse COO tensor.

    Input: SciPy sparse matrix.
    Output: torch sparse tensor with the same shape and non-zero values.
    """
    x = x.tocoo()
    indices = torch.tensor(np.vstack([x.row, x.col]), dtype=torch.long)
    values = torch.tensor(x.data, dtype=torch.float32)

    return torch.sparse_coo_tensor(indices, values, size=x.shape).coalesce()

class SparseDataset(Dataset):
    """Wraps a SciPy sparse feature matrix and label list for use with DataLoader.

    Inputs: X (sparse matrix of examples), y (integer labels).
    Output per item: one sparse row and its label tensor.
    """

    def __init__(self, X, y):
        self.X = X
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

def sparse_collate(batch):
    """Collate individual sparse rows into one PyTorch sparse batch.

    Input: list of (sparse row, label) pairs from SparseDataset.
    Process: stack SciPy rows, convert to a torch sparse tensor, stack labels.
    Output: (X_batch sparse tensor, y_batch long tensor).
    """
    X_rows, y_rows = zip(*batch)

    # Stack SciPy sparse rows, then convert the whole batch at once.
    X_batch = scipy.sparse.vstack(X_rows)
    X_batch = scipy_to_torch_sparse(X_batch)

    y_batch = torch.stack(y_rows)

    return X_batch, y_batch

def train_one_epoch(model, dataloader, optimizer, criterion, device="cpu"):
    """Run one training epoch: forward, loss, backward, optimizer step.

    Inputs: model, train dataloader, optimizer, loss function, device.
    Output: dict with average train loss and accuracy for the epoch.
    """
    if isinstance(device, str):
        device = torch.device(device)
    
    model.train()  # enable dropout / training behaviour

    total_loss = 0.0
    total_correct = 0
    total_examples = 0

    for X_batch, y_batch in dataloader:
        # move the batch to the device
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)

        optimizer.zero_grad()  # clear gradients from the previous step

        logprobs = model(X_batch) # forward pass
        loss = criterion(logprobs, y_batch) # compute the loss

        loss.backward()   # compute gradients
        optimizer.step()  # update parameters

        # accumulate weighted loss and correct predictions for epoch averages
        total_loss += loss.item() * y_batch.size(0)
        predictions = logprobs.argmax(dim=1) # get the predicted class
        total_correct += (predictions == y_batch).sum().item() # count the number of correct predictions
        total_examples += y_batch.size(0) # count the number of examples

    return {
        "loss": total_loss / total_examples, # average loss
        "accuracy": total_correct / total_examples, # average accuracy
    }

def evaluate(model, dataloader, criterion, device="cpu"):
    """Evaluate the model without updating weights.

    Inputs: model, eval dataloader, loss function, device.
    Output: (metrics dict with loss/accuracy, numpy array of predicted labels).
    """
    if isinstance(device, str):
        device = torch.device(device)
        
    model.eval()  # disable dropout / training behaviour

    total_loss = 0.0
    total_correct = 0
    total_examples = 0

    all_preds = []
    with torch.no_grad():  # no gradients needed for evaluation

        for X_batch, y_batch in dataloader:
            # move the batch to the device
            X_batch = X_batch.to(device) 
            y_batch = y_batch.to(device)

            logprobs = model(X_batch) # forward pass

            loss = criterion(logprobs, y_batch) # compute the loss

            # accumulate metrics; also keep predictions for later use
            total_loss += loss.item() * y_batch.size(0)
            predictions = logprobs.argmax(dim=1) # get the predicted class
            all_preds.append(predictions)

            total_correct += (predictions == y_batch).sum().item() # count the number of correct predictions
            total_examples += y_batch.size(0)

    # concatenate batch predictions into one array
    return {
        "loss": total_loss / total_examples,
        "accuracy": total_correct / total_examples
    }, torch.cat([t.detach().cpu() for t in all_preds]).numpy()

def train_neural_network_model(training_set, validation_set, num_classes, encoder, save_to,   
                                hidden_dim=128, dropout=0.1, batch_size=64, learning_rate=1e-3, 
                                weight_decay=1e-4, num_epochs=20, device=torch.device("cpu"), hidden_dims=None):
    """Encode texts, train SparseMLP (or DeepSparseMLP), and checkpoint the best validation model.

    Inputs: train/val (text, label) pairs, num_classes, encoder (text -> sparse vector), path prefix to save weights,
            plus standard training hyperparameters.
            hidden_dims: if None, train the single-hidden-layer SparseMLP of width `hidden_dim`;
                         if a list (e.g. [256, 128]), train a DeepSparseMLP with those hidden layers.
    Process: vectorise documents, build sparse dataloaders, train for `num_epochs`, save best-by-val-loss.
    Output: per-epoch metrics list, best epoch index, and validation predictions from the best model.
    """
    if isinstance(device, str):
        device = torch.device(device)
        
    time_start = time.time()
    # Convert the training and validation sets to vectors
    print("Transforming documents...")
    training_vectors = [encoder(text) for text, label in training_set]
    training_labels = [int(label) for text, label in training_set]
    validation_vectors = [encoder(text) for text, label in validation_set]
    validation_labels = [int(label) for text, label in validation_set]

    time_document_to_vector = time.time()
    print(f"Document transformation done! Time taken: {time_document_to_vector - time_start:.3f} seconds")
    
    # Keep SciPy sparse here; conversion to torch happens in sparse_collate
    X_train_sparse = csr_matrix(training_vectors)
    X_val_sparse = csr_matrix(validation_vectors)

    train_dataset = SparseDataset(X_train_sparse, training_labels) # create the training dataset
    val_dataset = SparseDataset(X_val_sparse, validation_labels) # create the validation dataset

    # create the data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=sparse_collate,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=sparse_collate,
    )

    time_dataloader = time.time()
    print(f"Dataloader done! Time taken: {time_dataloader - time_document_to_vector:.3f} seconds")

    # save architecture-defining arguments
    if hidden_dims is None:
        args = {
            'input_dim': len(encoder),
            'hidden_dim': hidden_dim,
            'num_classes': num_classes,
            'dropout': dropout,
        }
        architecture = SparseMLP
    else:
        args = {
            'input_dim': len(encoder),
            'hidden_dims': list(hidden_dims),
            'num_classes': num_classes,
            'dropout': dropout,
        }
        architecture = DeepSparseMLP

    with open(f"{save_to}.json", "w", encoding="utf-8") as f:
        json.dump({**args, 'architecture': architecture.__name__}, f, indent=2, ensure_ascii=False)

    # Create the model based on the input dimension, chosen hidden dimension(s), number of classification classes and dropout rate
    model = architecture(**args).to(device)
    criterion = nn.NLLLoss() # create the loss function
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay) # create the optimizer

    metrics = []
    print("Training...")
    predictions = None
    for epoch in tqdm(range(num_epochs)):
        # Train the model for one epoch
        train_metrics = train_one_epoch(model, train_loader, optimizer, criterion, device)
        # Evaluate the model on the validation set
        val_metrics, preds = evaluate(model, val_loader, criterion, device)
        # Store the metrics for this epoch
        metrics.append({"train_metrics": train_metrics, "val_metrics": val_metrics})

        print(
            f"Epoch {epoch + 1:02d} | "
            f"train loss: {train_metrics['loss']:.4f} | "
            f"train acc: {train_metrics['accuracy']:.4f} | "
            f"val loss: {val_metrics['loss']:.4f} | "
            f"val acc: {val_metrics['accuracy']:.4f}"
            )
        
        # Saving the best model according to validation loss
        if epoch != 0:
            if val_metrics["loss"] <= best_val_loss:
                best_val_loss = val_metrics["loss"]
                torch.save(model.state_dict(), f"{save_to}.pt")
                best_epoch = epoch
                predictions = preds
        else:
            best_val_loss = val_metrics["loss"]
            torch.save(model.state_dict(), f"{save_to}.pt")
            best_epoch = epoch
            predictions = preds
    
    time_training = time.time()
    print(f"Training done! Time taken: {time_training - time_dataloader:.3f} seconds")

    return metrics, best_epoch, predictions


def load(path, device="cpu"):
    """
    Reconstruct the model architecture (using {path}.json) and load trained weights (using {path}.pt) and return it.
    """

    with open(f"{path}.json", "r", encoding="utf-8") as f:
        args = json.load(f)

    if isinstance(device, str):
        device = torch.device(device)

    with open(f"{path}.pt", "rb") as f:
        state_dict = torch.load(f, map_location=device)

    # older checkpoints have no 'architecture' key and are always SparseMLP
    architecture = {"SparseMLP": SparseMLP, "DeepSparseMLP": DeepSparseMLP}[args.pop("architecture", "SparseMLP")]
    model = architecture(**args)
    model.load_state_dict(state_dict)
    model = model.to(device)
    return model


def predict_probs(texts, model, encoder, batch_size=10):
    """
    Return a numpy array with shape [num_texts, num_classes] with the pmfs of the input texts as predicted by the model.
    
    texts: list of texts, each a string
    model: trained model
    encoder: encoder used in training
    batch_size: how many at a time
    """
    
    device = model.weight.device
        
    vectors = [encoder(x) for x in texts]
    labels = [0 for _ in texts]

    x_sparse = csr_matrix(vectors)

    dl = DataLoader(
        SparseDataset(x_sparse, labels),
        batch_size=batch_size,
        shuffle=False,
        collate_fn=sparse_collate,
    )
    
    model.eval()  # disable dropout / training behaviour
    
    all_probs = []
    with torch.no_grad():  # no gradients needed for evaluation

        for x_batch, y_batch in dl:
            # move the batch to the device
            x_batch = x_batch.to(device) 
            y_batch = y_batch.to(device)
            logprobs = model(x_batch) # forward pass
            # accumulate metrics; also keep predictions for later use
            all_probs.extend(logprobs.exp().detach().cpu().numpy())
            
    return np.array(all_probs)


import os
import random
import tempfile


class MLPModel:
    """
    A wrapper around SparseMLP (single hidden layer, as in the tutorial) with the same interface as LogLinearModel,
    so that all models can be used interchangeably in the cross-validation loop.

    After fit, the returned model also carries:
        model.history: per-epoch train/val loss and accuracy
        model.best_epoch: epoch (0-based) of the checkpoint that was kept
    """
    default_hidden_dims = None  # None -> SparseMLP

    @staticmethod
    def _split_validation(training_set, val_fraction, seed):
        """Hold out a random `val_fraction` of the training set for checkpoint selection."""
        indices = list(range(len(training_set)))
        random.Random(seed).shuffle(indices)
        n_val = max(1, int(len(indices) * val_fraction))
        validation_set = [training_set[i] for i in indices[:n_val]]
        training_set = [training_set[i] for i in indices[n_val:]]
        return training_set, validation_set

    @classmethod
    def load(cls, path, device="cpu"):
        """
        Load a model from {path}.json and {path}.pt and return it.
        """
        return load(path, device=device)

    @classmethod
    def fit(cls, training_set, encoder, validation_set=None, save_to=None, num_classes=None,
            hidden_dim=128, hidden_dims="default", dropout=0.1, batch_size=64, learning_rate=1e-3,
            weight_decay=1e-4, num_epochs=20, val_fraction=0.1, seed=42, device=None):
        """
        Constructs, trains and returns an MLP (the checkpoint with the lowest validation loss).

        training_set: list of text-label pairs (a text is a string, a label is a 0-based integer)
        encoder: an instance of the Encoder class (such as TokenIndicator)
        validation_set: list of text-label pairs used to pick the best epoch;
            if None, `val_fraction` of the training set is held out for this
            (never pass the test fold of a cross-validation split here, that would leak test data into model selection)
        save_to: (str) if given, save the model to {save_to}.json and {save_to}.pt
        num_classes: number of labels; if None, inferred as max label + 1
        hidden_dim: width of the hidden layer (only used by MLPModel)
        hidden_dims: widths of the hidden layers (only used by DeepMLPModel); "default" uses the class default
        device: "cuda" / "cpu"; if None, use the GPU when available
        """
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        if hidden_dims == "default":
            hidden_dims = cls.default_hidden_dims
        if validation_set is None:
            training_set, validation_set = cls._split_validation(training_set, val_fraction, seed)
        if num_classes is None:
            num_classes = max(int(label) for _, label in training_set + validation_set) + 1

        torch.manual_seed(seed)

        with tempfile.TemporaryDirectory() as tmp_dir:
            path = save_to if save_to is not None else os.path.join(tmp_dir, "model")
            metrics, best_epoch, _ = train_neural_network_model(
                training_set, validation_set, num_classes, encoder, path,
                hidden_dim=hidden_dim, dropout=dropout, batch_size=batch_size, learning_rate=learning_rate,
                weight_decay=weight_decay, num_epochs=num_epochs, device=device, hidden_dims=hidden_dims,
            )
            model = load(path, device=device)  # reload the best checkpoint, not the last epoch

        model.history = metrics
        model.best_epoch = best_epoch
        return model

    @staticmethod
    def predict_probs(texts, model, encoder, batch_size=256):
        """
        Returns a numpy array [num_texts, num_classes] with the predicted class probabilities.

        texts: a list of texts, each a string
        model: a trained SparseMLP / DeepSparseMLP
        encoder: the Encoder used for in the model's training
        """
        return predict_probs(texts, model, encoder, batch_size=batch_size)

    @staticmethod
    def predict(texts, model, encoder, batch_size=256):
        """
        Predicts labels (via the most probable class decision rule) for a collection of texts.

        texts: a list of texts, each a string
        model: a trained SparseMLP / DeepSparseMLP
        encoder: the Encoder used for in the model's training
        """
        return predict_probs(texts, model, encoder, batch_size=batch_size).argmax(axis=1)


class DeepMLPModel(MLPModel):
    """
    The modified MLP (third architecture): DeepSparseMLP with two hidden layers and LayerNorm.
    Same interface as MLPModel and LogLinearModel; pass hidden_dims=[...] to fit to change the depth/widths.
    """
    default_hidden_dims = [256, 128]
