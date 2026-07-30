import sys
import os
import numpy as np

# Ensure project root is on Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from model.transformer import BareMetalTransformer
from core.ops import cross_entropy_loss
from core.optimizers import AdamW

def run_sanity_check():
    # Set seed for reproducible initialization
    np.random.seed(42)

    # 1. Hyperparameters for mini model
    vocab_size = 64
    d_model = 32
    num_heads = 4
    hidden_dim = 64
    n_layers = 2
    
    print("Initializing BareMetalTransformer model...")
    model = BareMetalTransformer(
        vocab_size=vocab_size,
        d_model=d_model,
        num_heads=num_heads,
        hidden_dim=hidden_dim,
        n_layers=n_layers
    )

    optimizer = AdamW(model.parameters(), lr=1e-2, weight_decay=0.0)

    # 2. Mock sequence data (Batch=1, Seq_Len=8)
    # Token sequence: [1, 5, 12, 8, 20, 3, 14, 9]
    input_ids = np.array([[1, 5, 12, 8, 20, 3, 14, 9]], dtype=np.int32)
    # Target sequence shifted by 1: [5, 12, 8, 20, 3, 14, 9, 1]
    targets = np.array([[5, 12, 8, 20, 3, 14, 9, 1]], dtype=np.int32)

    initial_expected_loss = np.log(vocab_size)
    print(f"Expected initial loss (random guess over vocab {vocab_size}): ~{initial_expected_loss:.4f}\n")
    print("--- Starting Overfitting Sanity Check ---")

    steps = 100
    for step in range(steps):
        # Clear accumulated gradients in autograd DAG
        optimizer.reset_grad()

        # Forward pass: (B, S) -> (B, S, V)
        logits = model.forward(input_ids)

        # Loss calculation
        loss = cross_entropy_loss(logits, targets)

        # Backward pass: compute analytical gradients through all blocks
        loss.backward_prop()

        # Optimizer update step
        optimizer.step()

        # Print loss progression
        if step % 10 == 0 or step == steps - 1:
            print(f"Step {step:02d} | Loss: {loss.data:.6f}")

    print("-----------------------------------------")
    if loss.data < 0.05:
        print("SUCCESS: Model successfully overfit on toy sequence and loss converged near zero!")
    else:
        print("WARNING: Loss did not decay as expected. Check learning rate or gradient flow.")

if __name__ == "__main__":
    run_sanity_check()
