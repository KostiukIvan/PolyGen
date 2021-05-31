import torch


def accuracy(y_pred, y_true, ignore_index=None, device=None):
    y_pred = y_pred.argmax(dim=1)
    if ignore_index:
        norm = torch.sum(y_true != ignore_index)
        mask = torch.where(
            y_true == ignore_index,
            torch.zeros_like(y_true),
            torch.ones_like(y_true),
        ).to(device, dtype=torch.float)
    else:
        norm = y_true.shape[0]
        mask = torch.ones_like(y_true).to(device, dtype=torch.float)

    acc = (y_pred.reshape(-1) == y_true.reshape(-1)).type(torch.float32)
    acc = torch.sum(acc * mask.reshape(-1))
    return acc / norm
