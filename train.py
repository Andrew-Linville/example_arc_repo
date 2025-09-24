import time
import numpy as np
from tqdm import trange

def fake_training(epochs=10, sleep_time=2):
    for epoch in trange(1, epochs + 1, desc="Training", unit="epoch"):
        # simulate some "loss"
        loss = np.exp(-epoch / 5) + 0.05 * np.random.rand()
        print(f"Epoch {epoch}/{epochs} - loss: {loss:.4f}")
        time.sleep(sleep_time)

if __name__ == "__main__":
    print("Starting fake training run...")
    fake_training(epochs=400, sleep_time=1)
    print("Training complete.")
