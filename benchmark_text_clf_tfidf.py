import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from datasets import load_dataset
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import time
from fasterkan.fasterkan import FasterKAN
from efficient_kan import KAN
from torchkan import KAL_Net
from fastkan.fastkan import FastKAN as FastKANORG

# --- Load Myanmar News dataset with TF-IDF features ---
def load_myanmar_news(device='cpu', max_features=1000):
    dataset = load_dataset("ThuraAung1601/myanmar_news")
    texts_train = list(dataset['train']['text'])
    labels_train = list(dataset['train']['label'])
    texts_test = list(dataset['test']['text'])
    labels_test = list(dataset['test']['label'])

    # Map string labels to integers
    unique_labels = sorted(set(labels_train + labels_test))
    label2id = {label: i for i, label in enumerate(unique_labels)}
    labels_train_int = [label2id[l] for l in labels_train]
    labels_test_int = [label2id[l] for l in labels_test]

    # Use TF-IDF vectorizer for better text representation
    vectorizer = TfidfVectorizer(
        max_features=max_features,
        stop_words=None,  # Keep all words for Myanmar text
        ngram_range=(1, 2),  # Use unigrams and bigrams
        min_df=2,  # Ignore terms that appear in less than 2 documents
        max_df=0.95  # Ignore terms that appear in more than 95% of documents
    )
    
    # Fit on training data and transform both train and test
    X_train_tfidf = vectorizer.fit_transform(texts_train).toarray()
    X_test_tfidf = vectorizer.transform(texts_test).toarray()
    
    # Convert to tensors
    X_train = torch.tensor(X_train_tfidf, dtype=torch.float32, device=device)
    X_test = torch.tensor(X_test_tfidf, dtype=torch.float32, device=device)
    y_train = torch.tensor(labels_train_int, dtype=torch.long, device=device)
    y_test = torch.tensor(labels_test_int, dtype=torch.long, device=device)

    print(f"Feature dimension: {X_train.shape[1]}")
    print(f"Number of classes: {len(unique_labels)}")
    print(f"Training samples: {X_train.shape[0]}")
    print(f"Test samples: {X_test.shape[0]}")

    return {
        'train_input': X_train, 'train_label': y_train,
        'test_input': X_test, 'test_label': y_test
    }, len(unique_labels)

# --- Simple MLP ---
class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim // 2, output_dim)
        )

    def forward(self, x):
        return self.layers(x)

# --- Training function ---
def train_model(model, dataset, device, epochs=20, batch_size=32, lr=1e-3):
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    loss_fn = nn.CrossEntropyLoss()
    
    # Create data loader
    train_dataset = TensorDataset(dataset['train_input'], dataset['train_label'])
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    for epoch in range(epochs):
        total_loss = 0
        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = loss_fn(outputs, batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        if (epoch + 1) % 5 == 0:
            avg_loss = total_loss / len(train_loader)
            print(f"  Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")

# --- Speed benchmark ---
def benchmark_speed(model, dataset, device, batch_size=64, reps=10):
    model.eval()
    forward_times, backward_times = [], []
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()

    # Warm up
    for _ in range(3):
        idxs = np.random.choice(dataset['train_input'].shape[0], batch_size, replace=False)
        x = dataset['train_input'][idxs]
        y = dataset['train_label'][idxs]
        pred = model(x)
        loss = loss_fn(pred, y)
        loss.backward()
        optimizer.zero_grad()

    for _ in range(reps):
        idxs = np.random.choice(dataset['train_input'].shape[0], batch_size, replace=False)
        x = dataset['train_input'][idxs]
        y = dataset['train_label'][idxs]

        # Forward timing
        torch.cuda.synchronize() if device == 'cuda' else None
        t0 = time.time()
        pred = model(x)
        torch.cuda.synchronize() if device == 'cuda' else None
        t1 = time.time()
        forward_times.append((t1 - t0) * 1000)

        # Backward timing
        optimizer.zero_grad()
        torch.cuda.synchronize() if device == 'cuda' else None
        t2 = time.time()
        loss = loss_fn(pred, y)
        loss.backward()
        torch.cuda.synchronize() if device == 'cuda' else None
        t3 = time.time()
        backward_times.append((t3 - t2) * 1000)

    return {
        'forward_ms': np.mean(forward_times),
        'backward_ms': np.mean(backward_times)
    }

# --- Classification evaluation ---
def evaluate_model(model, dataset, device):
    model.eval()
    with torch.no_grad():
        logits = model(dataset['test_input'])
        y_pred = logits.argmax(dim=1).cpu().numpy()
        y_true = dataset['test_label'].cpu().numpy()
        
        acc = accuracy_score(y_true, y_pred)
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true, y_pred, average='weighted', zero_division=0
        )
        
    return {
        'accuracy': acc,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }

# --- Count parameters ---
def count_params(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable

# --- Run all models ---
def run_all_models(device='cpu', batch_size=32, hidden_size=128, epochs=20):
    print("Loading dataset...")
    dataset, num_classes = load_myanmar_news(device=device, max_features=1000)
    input_dim = dataset['train_input'].shape[1]
    
    model_constructors = {
        'mlp': lambda: MLP(input_dim, hidden_size, num_classes),
        'fasterkan': lambda: FasterKAN(
            layers_hidden=[input_dim, hidden_size, num_classes],
            grid_min=-2.0, grid_max=2.0, num_grids=8, exponent=2,
            train_grid=True, train_inv_denominator=True
        ),
        'fastkanorg': lambda: FastKANORG(
            layers_hidden=[input_dim, hidden_size, num_classes],
            grid_min=-2.0, grid_max=2.0, num_grids=8
        ),
        'efficientkan': lambda: KAN(
            layers_hidden=[input_dim, hidden_size, num_classes],
            grid_size=5, spline_order=3
        ),
        'kalnet': lambda: KAL_Net(
            layers_hidden=[input_dim, hidden_size, num_classes],
            polynomial_order=3, base_activation=nn.SiLU
        )
    }

    results = {}
    
    for model_name, constructor in model_constructors.items():
        print(f"\n{'='*50}")
        print(f"Running model: {model_name.upper()}")
        print(f"{'='*50}")
        
        try:
            model = constructor().to(device)
            total_params, trainable_params = count_params(model)
            print(f"Parameters - Total: {total_params:,}, Trainable: {trainable_params:,}")
            
            # Train the model
            print("Training...")
            train_start = time.time()
            train_model(model, dataset, device, epochs=epochs, batch_size=batch_size)
            train_time = time.time() - train_start
            
            # Evaluate performance
            print("Evaluating...")
            metrics = evaluate_model(model, dataset, device)
            
            # Speed benchmark
            print("Speed benchmark...")
            speed_metrics = benchmark_speed(model, dataset, device, batch_size=batch_size)
            
            # Combine results
            result = {
                'total_params': total_params,
                'trainable_params': trainable_params,
                'train_time_sec': train_time,
                **metrics,
                **speed_metrics
            }
            
            results[model_name] = result
            
            print(f"Results: {result}")
            
        except Exception as e:
            print(f"Error with {model_name}: {str(e)}")
            results[model_name] = {'error': str(e)}
    
    # Print summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"{'Model':<12} {'Accuracy':<8} {'F1':<6} {'Params':<8} {'Train(s)':<8} {'Fwd(ms)':<7} {'Bwd(ms)':<7}")
    print("-" * 60)
    
    for name, result in results.items():
        if 'error' not in result:
            print(f"{name:<12} {result['accuracy']:<8.3f} {result['f1']:<6.3f} "
                  f"{result['total_params']:<8,} {result['train_time_sec']:<8.1f} "
                  f"{result['forward_ms']:<7.2f} {result['backward_ms']:<7.2f}")
        else:
            print(f"{name:<12} ERROR: {result['error']}")

if __name__ == "__main__":
    run_all_models(device='cuda', batch_size=32, hidden_size=128, epochs=15)