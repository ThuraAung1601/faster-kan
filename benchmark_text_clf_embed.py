import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from datasets import load_dataset
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from collections import Counter
import numpy as np
import time
import re
from fasterkan.fasterkan import FasterKAN
from efficient_kan import KAN
from torchkan import KAL_Net
from fastkan.fastkan import FastKAN as FastKANORG

# --- Text preprocessing ---
def preprocess_text(text):
    """Basic text preprocessing"""
    # Convert to lowercase
    text = text.lower()
    # Remove special characters but keep Myanmar script
    text = re.sub(r'[^\u1000-\u109f\u0020-\u007e]', ' ', text)  # Keep Myanmar and ASCII
    # Remove extra whitespace
    text = ' '.join(text.split())
    return text

# --- Build vocabulary ---
def build_vocab(texts, vocab_size=10000, min_freq=2):
    """Build vocabulary from texts"""
    word_freq = Counter()
    for text in texts:
        words = text.split()
        word_freq.update(words)
    
    # Keep most frequent words
    vocab = {'<PAD>': 0, '<UNK>': 1}
    for word, freq in word_freq.most_common(vocab_size - 2):
        if freq >= min_freq:
            vocab[word] = len(vocab)
    
    return vocab

def texts_to_sequences(texts, vocab, max_length=256):
    """Convert texts to sequences of token IDs"""
    sequences = []
    for text in texts:
        words = text.split()[:max_length]  # Truncate to max_length
        seq = [vocab.get(word, vocab['<UNK>']) for word in words]
        # Pad sequences
        if len(seq) < max_length:
            seq.extend([vocab['<PAD>']] * (max_length - len(seq)))
        sequences.append(seq)
    return torch.tensor(sequences, dtype=torch.long)

# --- Load Myanmar News dataset with embeddings ---
def load_myanmar_news(device='cpu', vocab_size=10000, max_length=256):
    dataset = load_dataset("ThuraAung1601/myanmar_news")
    
    # Preprocess texts
    texts_train = [preprocess_text(text) for text in dataset['train']['text']]
    texts_test = [preprocess_text(text) for text in dataset['test']['text']]
    labels_train = list(dataset['train']['label'])
    labels_test = list(dataset['test']['label'])

    # Build vocabulary
    vocab = build_vocab(texts_train + texts_test, vocab_size=vocab_size)
    
    # Convert to sequences
    X_train = texts_to_sequences(texts_train, vocab, max_length).to(device)
    X_test = texts_to_sequences(texts_test, vocab, max_length).to(device)

    # Map string labels to integers
    unique_labels = sorted(set(labels_train + labels_test))
    label2id = {label: i for i, label in enumerate(unique_labels)}
    y_train = torch.tensor([label2id[l] for l in labels_train], dtype=torch.long, device=device)
    y_test = torch.tensor([label2id[l] for l in labels_test], dtype=torch.long, device=device)

    print(f"Vocabulary size: {len(vocab)}")
    print(f"Sequence length: {max_length}")
    print(f"Number of classes: {len(unique_labels)}")
    print(f"Training samples: {X_train.shape[0]}")
    print(f"Test samples: {X_test.shape[0]}")

    return {
        'train_input': X_train, 'train_label': y_train,
        'test_input': X_test, 'test_label': y_test,
        'vocab_size': len(vocab), 'max_length': max_length
    }, len(unique_labels)

# --- Text Classification Models ---
class EmbeddingMLP(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, output_dim, max_length):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.dropout1 = nn.Dropout(0.3)
        
        # Use global average pooling instead of just mean
        self.fc = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim // 2, output_dim)
        )

    def forward(self, x):
        # x: [batch_size, seq_length]
        embedded = self.embedding(x)  # [batch_size, seq_length, embed_dim]
        embedded = self.dropout1(embedded)
        
        # Global average pooling (ignore padding tokens)
        mask = (x != 0).unsqueeze(-1).float()  # [batch_size, seq_length, 1]
        masked_embedded = embedded * mask
        pooled = masked_embedded.sum(dim=1) / mask.sum(dim=1).clamp(min=1)  # [batch_size, embed_dim]
        
        return self.fc(pooled)

class TextCNN(nn.Module):
    def __init__(self, vocab_size, embed_dim, output_dim, num_filters=100):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        
        # Multiple filter sizes for n-gram features
        self.convs = nn.ModuleList([
            nn.Conv1d(embed_dim, num_filters, kernel_size=k)
            for k in [3, 4, 5]
        ])
        
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(len(self.convs) * num_filters, output_dim)

    def forward(self, x):
        embedded = self.embedding(x)  # [batch_size, seq_length, embed_dim]
        embedded = embedded.transpose(1, 2)  # [batch_size, embed_dim, seq_length]
        
        conv_outputs = []
        for conv in self.convs:
            conv_out = torch.relu(conv(embedded))  # [batch_size, num_filters, new_length]
            pooled = torch.max_pool1d(conv_out, conv_out.size(2))  # [batch_size, num_filters, 1]
            conv_outputs.append(pooled.squeeze(2))  # [batch_size, num_filters]
        
        concatenated = torch.cat(conv_outputs, dim=1)  # [batch_size, len(convs) * num_filters]
        output = self.dropout(concatenated)
        return self.fc(output)

class EmbeddingKAN(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, output_dim, max_length, kan_type='fasterkan'):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.dropout1 = nn.Dropout(0.3)
        
        if kan_type == 'fasterkan':
            self.kan = FasterKAN(
                layers_hidden=[embed_dim, hidden_dim, output_dim],
                grid_min=-2.0, grid_max=2.0, num_grids=8, exponent=2,
                train_grid=True, train_inv_denominator=True
            )
        elif kan_type == 'efficientkan':
            self.kan = KAN(
                layers_hidden=[embed_dim, hidden_dim, output_dim],
                grid_size=5, spline_order=3
            )
        elif kan_type == 'kalnet':
            self.kan = KAL_Net(
                layers_hidden=[embed_dim, hidden_dim, output_dim],
                polynomial_order=3, base_activation=nn.SiLU
            )

    def forward(self, x):
        embedded = self.embedding(x)
        embedded = self.dropout1(embedded)
        
        # Global average pooling
        mask = (x != 0).unsqueeze(-1).float()
        masked_embedded = embedded * mask
        pooled = masked_embedded.sum(dim=1) / mask.sum(dim=1).clamp(min=1)
        
        return self.kan(pooled)

# --- Training function with better optimization ---
def train_model(model, dataset, device, epochs=25, batch_size=32, lr=1e-3):
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    loss_fn = nn.CrossEntropyLoss()
    
    train_dataset = TensorDataset(dataset['train_input'], dataset['train_label'])
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    
    best_val_acc = 0
    patience = 5
    patience_counter = 0
    
    for epoch in range(epochs):
        total_loss = 0
        model.train()
        
        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = loss_fn(outputs, batch_y)
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            total_loss += loss.item()
        
        scheduler.step()
        
        # Validation on subset for early stopping
        if (epoch + 1) % 3 == 0:
            model.eval()
            with torch.no_grad():
                val_outputs = model(dataset['test_input'][:500])  # Use subset for speed
                val_pred = val_outputs.argmax(dim=1)
                val_acc = (val_pred == dataset['test_label'][:500]).float().mean().item()
            
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                patience_counter = 0
            else:
                patience_counter += 1
            
            avg_loss = total_loss / len(train_loader)
            current_lr = optimizer.param_groups[0]['lr']
            print(f"  Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}, Val Acc: {val_acc:.4f}, LR: {current_lr:.6f}")
            
            if patience_counter >= patience and epoch > epochs // 2:
                print(f"  Early stopping at epoch {epoch+1}")
                break

# --- Speed benchmark ---
def benchmark_speed(model, dataset, device, batch_size=64, reps=10):
    model.eval()
    forward_times, backward_times = [], []
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()

    # Warm up
    for _ in range(3):
        idxs = np.random.choice(dataset['train_input'].shape[0], min(batch_size, dataset['train_input'].shape[0]), replace=False)
        x = dataset['train_input'][idxs]
        y = dataset['train_label'][idxs]
        pred = model(x)
        loss = loss_fn(pred, y)
        loss.backward()
        optimizer.zero_grad()

    for _ in range(reps):
        idxs = np.random.choice(dataset['train_input'].shape[0], min(batch_size, dataset['train_input'].shape[0]), replace=False)
        x = dataset['train_input'][idxs]
        y = dataset['train_label'][idxs]

        # Forward timing
        if device == 'cuda':
            torch.cuda.synchronize()
        t0 = time.time()
        pred = model(x)
        if device == 'cuda':
            torch.cuda.synchronize()
        t1 = time.time()
        forward_times.append((t1 - t0) * 1000)

        # Backward timing
        optimizer.zero_grad()
        if device == 'cuda':
            torch.cuda.synchronize()
        t2 = time.time()
        loss = loss_fn(pred, y)
        loss.backward()
        if device == 'cuda':
            torch.cuda.synchronize()
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
        # Process in batches to avoid memory issues
        batch_size = 128
        all_preds = []
        
        for i in range(0, len(dataset['test_input']), batch_size):
            end_i = min(i + batch_size, len(dataset['test_input']))
            batch_input = dataset['test_input'][i:end_i]
            logits = model(batch_input)
            preds = logits.argmax(dim=1)
            all_preds.append(preds)
        
        y_pred = torch.cat(all_preds).cpu().numpy()
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
def run_all_models(device='cpu', batch_size=32, embed_dim=128, hidden_size=128, epochs=25):
    print("Loading dataset...")
    dataset, num_classes = load_myanmar_news(device=device, vocab_size=10000, max_length=256)
    vocab_size = dataset['vocab_size']
    max_length = dataset['max_length']
    
    model_constructors = {
        'embedding_mlp': lambda: EmbeddingMLP(vocab_size, embed_dim, hidden_size, num_classes, max_length),
        # 'text_cnn': lambda: TextCNN(vocab_size, embed_dim, num_classes, num_filters=100),
        'embedding_fasterkan': lambda: EmbeddingKAN(vocab_size, embed_dim, hidden_size, num_classes, max_length, 'fasterkan'),
        'embedding_efficientkan': lambda: EmbeddingKAN(vocab_size, embed_dim, hidden_size, num_classes, max_length, 'efficientkan'),
        # 'embedding_kalnet': lambda: EmbeddingKAN(vocab_size, embed_dim, hidden_size, num_classes, max_length, 'kalnet'),
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
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    print(f"{'Model':<18} {'Accuracy':<8} {'F1':<6} {'Params':<8} {'Train(s)':<8} {'Fwd(ms)':<7} {'Bwd(ms)':<7}")
    print("-" * 70)
    
    for name, result in results.items():
        if 'error' not in result:
            print(f"{name:<18} {result['accuracy']:<8.3f} {result['f1']:<6.3f} "
                  f"{result['total_params']:<8,} {result['train_time_sec']:<8.1f} "
                  f"{result['forward_ms']:<7.2f} {result['backward_ms']:<7.2f}")
        else:
            print(f"{name:<18} ERROR: {result['error']}")

if __name__ == "__main__":
    run_all_models(device='cuda', batch_size=32, embed_dim=300, hidden_size=128, epochs=20)