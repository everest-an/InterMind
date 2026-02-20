import torch


def resolve_device(preference: str = "auto") -> torch.device:
    """Resolve the compute device based on user preference.

    Args:
        preference: One of "auto", "cpu", "cuda", "mps".

    Returns:
        A torch.device instance.
    """
    if preference == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(preference)
