import argparse
import pandas as pd
import numpy as np
import torch
from torch import nn
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from fasterkan.fasterkan import FasterKAN, FasterKANvolver
from efficient_kan import KAN
from torchkan import KAL_Net
from fastkan.fastkan import FastKAN as FastKANORG
from torchkan import KANvolver
import time

# --- Dataset from CSV ---
def create_dataset_from_csv(csv_path, vocab=None, embedding_dim=128, device='cpu'):
    df = pd.read_csv(csv_path)
    texts = df['text'].tolist()
    labels = df['label'].astype(int).tolist()
    num_classes = len(set(labels))
    
    if vocab is None:
        all_words = set(word for text in texts for word in text.split())
        vocab = {w: i for i, w in enumerate(all_words)}
    
    vocab_size = len(vocab)
    embeddings = torch.randn(vocab_size, embedding_dim)
    X_embed = []
    for text in texts:
        idxs = [vocab[w] for w in text.split() if w in vocab]
        if len(idxs) == 0:
            idxs = [0]
        vec = embeddings[idxs].mean(dim=0)
        X_embed.append(vec)
    X_embed = torch.stack(X_embed).to(device)
    
    y_tensor = torch.tensor(labels)
    y_onehot = nn.functional.one_hot(y_tensor, num_classes).float().to(device)
    
    return X_embed, y_onehot, num_classes

# --- MLP ---
class MLP(nn.Module):
    def __init__(self, layers: tuple, device='cpu'):
        super().__init__()
        self.layer1 = nn.Linear(layers[0], layers[1], device=device)
        self.layer2 = nn.Linear(layers[1], layers[2], device=device)

    def forward(self, x):
        x = self.layer1(x)
        x = nn.functional.relu(x)
        x = self.layer2(x)
        x = nn.functional.softmax(x, dim=-1)
        return x

# --- Benchmark time/memory ---
def benchmark(dataset, device, bs, loss_fn, model, reps):
    forward_times, backward_times = [], []
    for k in range(1 + reps):
        idxs = np.random.choice(dataset['train_input'].shape[0], bs, replace=False)
        x = dataset['train_input'][idxs].to(device)
        y = dataset['train_label'][idxs].to(device)

        if device == 'cpu':
            t0 = time.time()
            pred = model(x)
            t1 = time.time()
            if k > 0:
                forward_times.append((t1 - t0)*1000)
            loss = loss_fn(pred, y)
            t2 = time.time()
            loss.backward()
            t3 = time.time()
            if k > 0:
                backward_times.append((t3 - t2)*1000)
        elif device == 'cuda':
            torch.cuda.reset_peak_memory_stats()
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            pred = model(x)
            end.record()
            torch.cuda.synchronize()
            if k > 0:
                forward_times.append(start.elapsed_time(end))
            loss = loss_fn(pred, y)
            start.record()
            loss.backward()
            end.record()
            torch.cuda.synchronize()
            if k > 0:
                backward_times.append(start.elapsed_time(end))
    return {'forward': np.mean(forward_times), 'backward': np.mean(backward_times)}

# --- Compute classification metrics ---
def benchmark_classification(dataset, device, model):
    model.eval()
    x_test = dataset['test_input'].to(device)
    y_test = dataset['test_label'].to(device)
    with torch.no_grad():
        y_pred = model(x_test)
        y_true = y_test.argmax(dim=1).cpu().numpy()
        y_pred_labels = y_pred.argmax(dim=1).cpu().numpy()
        acc = accuracy_score(y_true, y_pred_labels)
        precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred_labels, average='weighted')
    return {'accuracy': acc, 'precision': precision, 'recall': recall, 'f1': f1}

# --- Count parameters ---
def count_params(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable

# --- K-Fold Cross Validation ---
def run_kfold_all(csv_path, k=5, batch_size=64, hid_size=64, reps=5, device='cpu', bool_flag=False):
    X, Y, num_classes = create_dataset_from_csv(csv_path, device=device)
    kf = KFold(n_splits=k, shuffle=True, random_state=0)

    model_constructors = {
        'mlp': lambda: MLP((X.shape[1], hid_size*8, num_classes), device=device),
        'fasterkan': lambda: FasterKAN(
            layers_hidden=[X.shape[1], hid_size, num_classes],
            grid_min=-1.2, grid_max=1.2, num_grids=8, exponent=2,
            train_grid=bool_flag, train_inv_denominator=bool_flag
        ),
        'fastkanorg': lambda: FastKANORG(layers_hidden=[X.shape[1], hid_size, num_classes], grid_min=-1.2, grid_max=1.2, num_grids=8),
        'efficientkan': lambda: KAN(layers_hidden=[X.shape[1], hid_size, num_classes], grid_size=5, spline_order=3),
        'kalnet': lambda: KAL_Net(layers_hidden=[X.shape[1], hid_size, num_classes], polynomial_order=3, base_activation=nn.SiLU),
        'kanvolve': lambda: KANvolver(layers_hidden=[hid_size, num_classes], polynomial_order=2, base_activation=nn.ReLU),
        'fasterkanvolver': lambda: FasterKANvolver(layers_hidden=[hid_size, num_classes], grid_min=-1.2, grid_max=0.2, num_grids=8, exponent=2, train_grid=bool_flag, train_inv_denominator=bool_flag)
    }

    loss_fn = lambda x, y: torch.mean((x - y)**2)

    for model_name, constructor in model_constructors.items():
        print(f"\nRunning 5-fold CV for {model_name.upper()}")
        fold_results = []
        fold_metrics = []
        for fold, (train_idx, test_idx) in enumerate(kf.split(X)):
            dataset = {
                'train_input': X[train_idx],
                'train_label': Y[train_idx],
                'test_input': X[test_idx],
                'test_label': Y[test_idx]
            }
            model = constructor()
            model.to(device)
            # Benchmark speed
            res = benchmark(dataset, device, batch_size, loss_fn, model, reps)
            res['params'], res['train_params'] = count_params(model)
            # Benchmark classification metrics
            metrics = benchmark_classification(dataset, device, model)
            res.update(metrics)
            print(f"Fold {fold+1}: {res}")
            fold_results.append(res)
            fold_metrics.append(metrics)
        
        # Average metrics across folds
        avg_forward = np.mean([r['forward'] for r in fold_results])
        avg_backward = np.mean([r['backward'] for r in fold_results])
        avg_acc = np.mean([m['accuracy'] for m in fold_metrics])
        avg_precision = np.mean([m['precision'] for m in fold_metrics])
        avg_recall = np.mean([m['recall'] for m in fold_metrics])
        avg_f1 = np.mean([m['f1'] for m in fold_metrics])
        print(f"\nAverage for {model_name.upper()}: Forward={avg_forward:.2f} ms, Backward={avg_backward:.2f} ms, Acc={avg_acc:.4f}, P={avg_precision:.4f}, R={avg_recall:.4f}, F1={avg_f1:.4f}")

# --- Main ---
if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv', type=str, required=True)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--hid-size', type=int, default=64)
    parser.add_argument('--reps', type=int, default=5)
    parser.add_argument('--k', type=int, default=5)
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--bool-flag', action='store_true')
    args = parser.parse_args()
    run_kfold_all(args.csv, k=args.k, batch_size=args.batch_size, hid_size=args.hid_size, reps=args.reps, device=args.device, bool_flag=args.bool_flag)
