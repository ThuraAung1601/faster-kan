import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from datasets import load_dataset
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from transformers import AutoTokenizer, AutoModel
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
    text = text.strip()
    # Remove excessive whitespace but preserve text structure for transformers
    text = ' '.join(text.split())
    return text

# --- Load Myanmar News dataset with transformer tokenization ---
def load_myanmar_news_transformers(model_name='xlm-roberta-base', device='cpu', max_length=512):
    dataset = load_dataset("ThuraAung1601/myanmar_news")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Preprocess texts
    texts_train = [preprocess_text(text) for text in dataset['train']['text']]
    texts_test = [preprocess_text(text) for text in dataset['test']['text']]
    labels_train = list(dataset['train']['label'])
    labels_test = list(dataset['test']['label'])

    # Tokenize texts
    print("Tokenizing training data...")
    train_encodings = tokenizer(
        texts_train,
        truncation=True,
        padding=True,
        max_length=max_length,
        return_tensors='pt'
    )
    
    print("Tokenizing test data...")
    test_encodings = tokenizer(
        texts_test,
        truncation=True,
        padding=True,
        max_length=max_length,
        return_tensors='pt'
    )

    # Map string labels to integers
    unique_labels = sorted(set(labels_train + labels_test))
    label2id = {label: i for i, label in enumerate(unique_labels)}
    y_train = torch.tensor([label2id[l] for l in labels_train], dtype=torch.long)
    y_test = torch.tensor([label2id[l] for l in labels_test], dtype=torch.long)

    print(f"Model: {model_name}")
    print(f"Max sequence length: {max_length}")
    print(f"Number of classes: {len(unique_labels)}")
    print(f"Training samples: {len(texts_train)}")
    print(f"Test samples: {len(texts_test)}")

    return {
        'train_input_ids': train_encodings['input_ids'].to(device),
        'train_attention_mask': train_encodings['attention_mask'].to(device),
        'train_label': y_train.to(device),
        'test_input_ids': test_encodings['input_ids'].to(device),
        'test_attention_mask': test_encodings['attention_mask'].to(device),
        'test_label': y_test.to(device),
    }, len(unique_labels), tokenizer

# --- Transformer-based Models ---
class TransformerMLP(nn.Module):
    def __init__(self, model_name, num_classes, dropout=0.3, freeze_backbone=False):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(model_name)
        
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
        
        hidden_size = self.backbone.config.hidden_size
        
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, num_classes)
        )

    def forward(self, input_ids, attention_mask):
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        # Use [CLS] token representation
        pooled_output = outputs.last_hidden_state[:, 0]  # [batch_size, hidden_size]
        return self.classifier(pooled_output)

class TransformerKAN(nn.Module):
    def __init__(self, model_name, num_classes, kan_type='fasterkan', freeze_backbone=False):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(model_name)
        
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
        
        hidden_size = self.backbone.config.hidden_size
        kan_hidden = min(hidden_size // 2, 256)  # Reduce KAN size for efficiency
        
        self.dropout = nn.Dropout(0.3)
        
        if kan_type == 'fasterkan':
            self.kan = FasterKAN(
                layers_hidden=[hidden_size, kan_hidden, num_classes],
                grid_min=-2.0, grid_max=2.0, num_grids=6, exponent=2,
                train_grid=True, train_inv_denominator=True
            )
        elif kan_type == 'efficientkan':
            self.kan = KAN(
                layers_hidden=[hidden_size, kan_hidden, num_classes],
                grid_size=5, spline_order=3
            )
        elif kan_type == 'kalnet':
            self.kan = KAL_Net(
                layers_hidden=[hidden_size, kan_hidden, num_classes],
                polynomial_order=3, base_activation=nn.SiLU
            )

    def forward(self, input_ids, attention_mask):
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.last_hidden_state[:, 0]
        pooled_output = self.dropout(pooled_output)
        return self.kan(pooled_output)

class TransformerCNN(nn.Module):
    def __init__(self, model_name, num_classes, freeze_backbone=False, num_filters=100):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(model_name)
        
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
        
        hidden_size = self.backbone.config.hidden_size
        
        # CNN layers
        self.convs = nn.ModuleList([
            nn.Conv1d(hidden_size, num_filters, kernel_size=k, padding=k//2)
            for k in [3, 4, 5]
        ])
        
        self.dropout = nn.Dropout(0.5)
        self.classifier = nn.Linear(len(self.convs) * num_filters, num_classes)

    def forward(self, input_ids, attention_mask):
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        hidden_states = outputs.last_hidden_state  # [batch_size, seq_len, hidden_size]
        
        # Apply attention mask to hidden states
        mask_expanded = attention_mask.unsqueeze(-1).expand(hidden_states.size()).float()
        hidden_states = hidden_states * mask_expanded
        
        # Transpose for CNN: [batch_size, hidden_size, seq_len]
        hidden_states = hidden_states.transpose(1, 2)
        
        conv_outputs = []
        for conv in self.convs:
            conv_out = torch.relu(conv(hidden_states))  # [batch_size, num_filters, seq_len]
            # Global max pooling
            pooled = torch.max_pool1d(conv_out, conv_out.size(2)).squeeze(2)  # [batch_size, num_filters]
            conv_outputs.append(pooled)
        
        concatenated = torch.cat(conv_outputs, dim=1)  # [batch_size, len(convs) * num_filters]
        output = self.dropout(concatenated)
        return self.classifier(output)

# --- Training function optimized for transformers ---
def train_model_transformer(model, dataset, device, epochs=10, batch_size=16, lr=2e-5):
    model.train()
    
    # Use different learning rates for backbone and classifier
    backbone_params = []
    classifier_params = []
    
    for name, param in model.named_parameters():
        if 'backbone' in name:
            backbone_params.append(param)
        else:
            classifier_params.append(param)
    
    optimizer = torch.optim.AdamW([
        {'params': backbone_params, 'lr': lr},
        {'params': classifier_params, 'lr': lr * 10}  # Higher LR for classifier
    ], weight_decay=1e-4)
    
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    loss_fn = nn.CrossEntropyLoss()
    
    # Create data loader
    train_dataset = TensorDataset(
        dataset['train_input_ids'],
        dataset['train_attention_mask'],
        dataset['train_label']
    )
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    
    best_val_acc = 0
    patience = 3
    patience_counter = 0
    
    for epoch in range(epochs):
        total_loss = 0
        model.train()
        
        for batch_input_ids, batch_attention_mask, batch_y in train_loader:
            optimizer.zero_grad()
            
            outputs = model(batch_input_ids, batch_attention_mask)
            loss = loss_fn(outputs, batch_y)
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            total_loss += loss.item()
        
        scheduler.step()
        
        # Quick validation
        if (epoch + 1) % 2 == 0:
            model.eval()
            with torch.no_grad():
                # Use subset for quick validation
                val_size = min(200, len(dataset['test_input_ids']))
                val_outputs = model(
                    dataset['test_input_ids'][:val_size],
                    dataset['test_attention_mask'][:val_size]
                )
                val_pred = val_outputs.argmax(dim=1)
                val_acc = (val_pred == dataset['test_label'][:val_size]).float().mean().item()
            
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

# --- Evaluation for transformer models ---
def evaluate_model_transformer(model, dataset, device, batch_size=32):
    model.eval()
    all_preds = []
    
    # Create data loader for evaluation
    test_dataset = TensorDataset(
        dataset['test_input_ids'],
        dataset['test_attention_mask']
    )
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    
    with torch.no_grad():
        for batch_input_ids, batch_attention_mask in test_loader:
            outputs = model(batch_input_ids, batch_attention_mask)
            preds = outputs.argmax(dim=1)
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

# --- Speed benchmark for transformers ---
def benchmark_speed_transformer(model, dataset, device, batch_size=16, reps=5):
    model.eval()
    forward_times, backward_times = [], []
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
    loss_fn = nn.CrossEntropyLoss()

    # Warm up
    for _ in range(2):
        idx_end = min(batch_size, len(dataset['train_input_ids']))
        input_ids = dataset['train_input_ids'][:idx_end]
        attention_mask = dataset['train_attention_mask'][:idx_end]
        labels = dataset['train_label'][:idx_end]
        
        pred = model(input_ids, attention_mask)
        loss = loss_fn(pred, labels)
        loss.backward()
        optimizer.zero_grad()

    for _ in range(reps):
        idx_end = min(batch_size, len(dataset['train_input_ids']))
        input_ids = dataset['train_input_ids'][:idx_end]
        attention_mask = dataset['train_attention_mask'][:idx_end]
        labels = dataset['train_label'][:idx_end]

        # Forward timing
        if device == 'cuda':
            torch.cuda.synchronize()
        t0 = time.time()
        pred = model(input_ids, attention_mask)
        if device == 'cuda':
            torch.cuda.synchronize()
        t1 = time.time()
        forward_times.append((t1 - t0) * 1000)

        # Backward timing
        optimizer.zero_grad()
        if device == 'cuda':
            torch.cuda.synchronize()
        t2 = time.time()
        loss = loss_fn(pred, labels)
        loss.backward()
        if device == 'cuda':
            torch.cuda.synchronize()
        t3 = time.time()
        backward_times.append((t3 - t2) * 1000)

    return {
        'forward_ms': np.mean(forward_times),
        'backward_ms': np.mean(backward_times)
    }

# --- Count parameters ---
def count_params(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable

# --- Run transformer models ---
def run_transformer_models(device='cuda', batch_size=16, epochs=8, max_length=256):
    
    # Available models (corrected the typo in electra)
    transformer_models = [
        # 'xlm-roberta-base',
        'bert-base-multilingual-cased',
        'google/electra-base-multilingual-cased',  # Fixed typo
        'distilbert-base-multilingual-cased',
        # 'multilingual-e5-large'  # This might be too large, uncomment if you have enough GPU memory
    ]
    
    results = {}
    
    for model_name in transformer_models:
        print(f"\n{'='*60}")
        print(f"PROCESSING MODEL: {model_name}")
        print(f"{'='*60}")
        
        try:
            # Load dataset with this tokenizer
            print("Loading dataset...")
            dataset, num_classes, tokenizer = load_myanmar_news_transformers(
                model_name=model_name, 
                device=device, 
                max_length=max_length
            )
            
            # Define model variants for this backbone
            model_variants = {
                f'{model_name.split("/")[-1]}_mlp': lambda: TransformerMLP(model_name, num_classes, freeze_backbone=False),
                # f'{model_name.split("/")[-1]}_cnn': lambda: TransformerCNN(model_name, num_classes, freeze_backbone=False),
                f'{model_name.split("/")[-1]}_fasterkan': lambda: TransformerKAN(model_name, num_classes, 'fasterkan', freeze_backbone=False),
                # Add more KAN variants if needed
                f'{model_name.split("/")[-1]}_efficientkan': lambda: TransformerKAN(model_name, num_classes, 'efficientkan', freeze_backbone=False),

            }
            
            for variant_name, constructor in model_variants.items():
                print(f"\n--- Training {variant_name.upper()} ---")
                
                try:
                    model = constructor().to(device)
                    total_params, trainable_params = count_params(model)
                    print(f"Parameters - Total: {total_params:,}, Trainable: {trainable_params:,}")
                    
                    # Train
                    train_start = time.time()
                    train_model_transformer(model, dataset, device, epochs=epochs, batch_size=batch_size)
                    train_time = time.time() - train_start
                    
                    # Evaluate
                    print("Evaluating...")
                    metrics = evaluate_model_transformer(model, dataset, device)
                    
                    # Speed benchmark
                    print("Speed benchmark...")
                    speed_metrics = benchmark_speed_transformer(model, dataset, device, batch_size=batch_size)
                    
                    result = {
                        'model_name': model_name,
                        'total_params': total_params,
                        'trainable_params': trainable_params,
                        'train_time_sec': train_time,
                        **metrics,
                        **speed_metrics
                    }
                    
                    results[variant_name] = result
                    print(f"Results: {result}")
                    
                    # Clean up GPU memory
                    del model
                    torch.cuda.empty_cache() if device == 'cuda' else None
                    
                except Exception as e:
                    print(f"Error with {variant_name}: {str(e)}")
                    results[variant_name] = {'error': str(e)}
            
            # Clean up dataset from GPU
            for key in dataset:
                del dataset[key]
            torch.cuda.empty_cache() if device == 'cuda' else None
                
        except Exception as e:
            print(f"Error loading {model_name}: {str(e)}")
    
    # Print final summary
    print(f"\n{'='*80}")
    print("FINAL SUMMARY - TRANSFORMER MODELS")
    print(f"{'='*80}")
    print(f"{'Model':<30} {'Accuracy':<8} {'F1':<6} {'Params':<10} {'Time(s)':<8} {'Fwd(ms)':<8}")
    print("-" * 80)
    
    for name, result in results.items():
        if 'error' not in result:
            print(f"{name:<30} {result['accuracy']:<8.3f} {result['f1']:<6.3f} "
                  f"{result['total_params']:<10,} {result['train_time_sec']:<8.1f} "
                  f"{result['forward_ms']:<8.2f}")
        else:
            print(f"{name:<30} ERROR: {result['error']}")

if __name__ == "__main__":
    # Recommended settings for transformers:
    # - Smaller batch size (16-32) due to memory constraints
    # - Lower learning rate (2e-5)
    # - Fewer epochs (5-10) as transformers converge faster
    # - Shorter max_length (256) for speed vs accuracy trade-off
    
    run_transformer_models(
        device='cuda', 
        batch_size=8, 
        epochs=5, 
        max_length=256
    )