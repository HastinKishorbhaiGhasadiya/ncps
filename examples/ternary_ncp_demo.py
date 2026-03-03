"""
Ternary NCP Demo: Benchmark comparing FP32 CfC vs Ternary CfC with NCP wiring.

Demonstrates combining three innovations:
1. CfC liquid dynamics (continuous-time ODE closed-form solution)
2. Ternary weights {-1, 0, +1} (16x memory compression)
3. NCP brain-like sparse wiring

No pytorch-lightning dependency required.
"""

import time
import numpy as np
import torch
import torch.nn as nn

from ncps.wirings import AutoNCP
from ncps.torch import CfC, TernaryCfC


def make_sinusoidal_data():
    """Generate sinusoidal data: sin+cos input, sin@2x target."""
    N = 48
    data_x = np.stack(
        [np.sin(np.linspace(0, 3 * np.pi, N)), np.cos(np.linspace(0, 3 * np.pi, N))],
        axis=1,
    )
    data_x = np.expand_dims(data_x, axis=0).astype(np.float32)
    data_y = np.sin(np.linspace(0, 6 * np.pi, N)).reshape([1, N, 1]).astype(np.float32)
    return torch.tensor(data_x), torch.tensor(data_y)


def count_params(model):
    return sum(p.numel() for p in model.parameters())


def get_memory_bytes(model):
    return sum(p.numel() * p.element_size() for p in model.parameters())


def train_model(model, data_x, data_y, epochs=400, lr=0.01, clip=1.0):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    model.train()
    t0 = time.time()
    final_loss = float("inf")
    for epoch in range(epochs):
        optimizer.zero_grad()
        output, _ = model(data_x)
        # output shape: (1, 48, output_size) — take last dim if > 1
        if output.shape[-1] > 1:
            pred = output[..., :1]
        else:
            pred = output
        loss = criterion(pred, data_y)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()
        final_loss = loss.item()
    elapsed = time.time() - t0
    return final_loss, elapsed


def run_inference_timing(model, data_x, runs=100):
    model.eval()
    with torch.no_grad():
        # Warm up
        for _ in range(5):
            model(data_x)
        t0 = time.time()
        for _ in range(runs):
            model(data_x)
        elapsed = time.time() - t0
    return elapsed / runs * 1000  # ms per run


def main():
    torch.manual_seed(42)
    np.random.seed(42)

    data_x, data_y = make_sinusoidal_data()
    print(f"Input shape: {data_x.shape}, Target shape: {data_y.shape}\n")

    # Create models
    model1 = CfC(2, AutoNCP(16, 1), batch_first=True)
    model2 = TernaryCfC(2, AutoNCP(16, 1), batch_first=True)
    model3 = TernaryCfC(2, 16, proj_size=1, batch_first=True)

    models = [
        ("FP32 CfC + NCP", model1),
        ("Ternary CfC + NCP", model2),
        ("Ternary CfC FC", model3),
    ]

    print("Training models for 400 epochs each...\n")
    results = []
    for name, model in models:
        loss, elapsed = train_model(model, data_x, data_y, epochs=400)
        params = count_params(model)
        mem = get_memory_bytes(model)
        results.append((name, params, mem, loss, elapsed))

    # Print comparison table
    print("-" * 75)
    print(f"{'Model':<22} {'Params':>8} {'Memory(B)':>12} {'FinalLoss':>12} {'Time(s)':>10}")
    print("-" * 75)
    for name, params, mem, loss, elapsed in results:
        print(f"{name:<22} {params:>8} {mem:>12} {loss:>12.6f} {elapsed:>10.2f}")
    print("-" * 75)

    # Print ternary stats for model2
    print("\nTernary weight distribution for 'Ternary CfC + NCP' (model2):")
    stats = model2.get_model_stats()
    print(f"  Total params:       {stats['total_params']}")
    print(f"  Ternary params:     {stats['ternary_params']}")
    print(f"  Compression ratio:  {stats['compression_ratio']:.2f}x")
    print(f"  FP32 memory (B):    {stats['fp32_memory_bytes']}")
    print(f"  Ternary memory (B): {stats['ternary_memory_bytes']}")
    print(f"  Memory saved (B):   {stats['memory_saved_bytes']}")
    print("\n  Per-layer ternary distribution:")
    for layer_name, dist in stats["ternary_layers"].items():
        print(
            f"    {layer_name}: neg=-1: {dist['neg_one']}, zero: {dist['zero']}, "
            f"pos=+1: {dist['pos_one']}, sparsity: {dist['sparsity']:.2%}"
        )

    # Inference timing
    print("\nInference timing (100 runs each):")
    print("-" * 40)
    for name, model in models:
        ms = run_inference_timing(model, data_x, runs=100)
        print(f"  {name:<22}: {ms:.3f} ms/run")
    print("-" * 40)


if __name__ == "__main__":
    main()
