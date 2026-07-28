import sys
import os

# Adds project root (baremetal-llm/) to Python's search path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
from core.tensor import Tensor
from core.ops import gelu, cross_entropy_loss
from core.optimizers import AdamW

def build_layer(in_f, out_f):
    scale = np.sqrt(2 / in_f)
    w_data = np.random.randn(in_f, out_f) * scale
    b_data = np.zeros((1, out_f))

    return Tensor(w_data, _op="W"), Tensor(b_data, _op="b")

def main():
    np.random.seed(42)  # For reproducible verification

    # 1. Hyperparameters & Dummy Dataset Setup
    num_samples = 32
    input_dim = 8
    hidden_dim = 16
    num_classes = 4
    epochs = 100
    learning_rate = 1e-2

    # Synthetic Input Features (Batch, Input_Dim)
    X_raw = np.random.randn(num_samples, input_dim).astype(np.float32)
    # Synthetic Ground Truth Labels (Batch,) with class indices 0 to (num_classes - 1)
    y_raw = np.random.randint(0, num_classes, size=(num_samples,))

    X = Tensor(X_raw, _op="X")

    # 2. Initialize Model Parameters
    W1, b1 = build_layer(input_dim, hidden_dim)
    W2, b2 = build_layer(hidden_dim, num_classes)

    parameters = [W1, b1, W2, b2]

    # 3. Initialize AdamW Optimizer
    optimizer = AdamW(parameters, lr=learning_rate, weight_decay=0.001)

    print("Starting Training Verification Loop...")
    print("-" * 50)

    # 4. Training Loop
    for epoch in range(1, epochs + 1):
        # A. Zero Out Gradients from Previous Step
        optimizer.reset_grad()

        # B. Forward Pass
        # Layer 1: Linear + GELU
        h1 = (X @ W1) + b1
        a1 = gelu(h1)

        # Layer 2: Linear (Logits)
        logits = (a1 @ W2) + b2

        # Compute Loss
        loss = cross_entropy_loss(logits, y_raw)

        ## (Debug) Checking if the weights actaully update 
        before = W2.data.copy()

        # C. Backward Pass
        loss.backward_prop()
 
        # D. Optimizer Step
        optimizer.step()



        # Log Progress Every 10 Epochs
        if epoch == 1 or epoch % 10 == 0:
            # Calculate classification accuracy
            preds = np.argmax(logits.data, axis=-1)
            accuracy = np.mean(preds == y_raw) * 100.0
            print(f"Epoch {epoch:3d}/{epochs} | Loss: {loss.data:.4f} | Accuracy: {accuracy:.1f}%")

    print("-" * 50)
    print("Verification Complete!")

if __name__ == "__main__":
    main()
