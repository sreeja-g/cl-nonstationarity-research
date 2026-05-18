import torch
import numpy as np
from torch.utils.data import DataLoader

def evaluate_untrained_baseline_accs(
    model,
    benchmark,
    batch_size=256,
    device=None,
):
    """
    Evaluate the initialized, untrained model on each test experience.

    Returns:
        baseline_accs: list of baseline accuracies, one per test experience.
    """

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    model = model.to(device)
    model.eval()

    baseline_accs = []

    with torch.no_grad():
        for test_exp in benchmark.test_stream:
            loader = DataLoader(
                test_exp.dataset,
                batch_size=batch_size,
                shuffle=False,
            )

            correct = 0
            total = 0

            for batch in loader:
                x = batch[0].to(device)
                y = batch[1].to(device)

                logits = model(x)

                if isinstance(logits, (tuple, list)):
                    logits = logits[0]

                preds = torch.argmax(logits, dim=1)

                correct += (preds == y).sum().item()
                total += y.size(0)

            acc = correct / total if total > 0 else 0.0
            baseline_accs.append(acc)

    model.train()

    return baseline_accs