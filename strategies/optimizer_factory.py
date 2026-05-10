from torch.optim import SGD, Adam, AdamW


def build_optimizer(
    model,
    optimizer_name="sgd",
    lr=0.01,
    momentum=0.9,
    weight_decay=0.0,
    **kwargs,
):
    optimizer_name = optimizer_name.lower().strip()

    if optimizer_name == "sgd":
        return SGD(
            model.parameters(),
            lr=lr,
            momentum=momentum,
            weight_decay=weight_decay,
            **kwargs,
        )

    if optimizer_name == "adam":
        return Adam(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay,
            **kwargs,
        )

    if optimizer_name == "adamw":
        return AdamW(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay,
            **kwargs,
        )

    raise ValueError(
        f"Unknown optimizer_name: {optimizer_name}"
    )