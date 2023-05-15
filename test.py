import torch

if __name__ == "__main__":
    data = [[1, 2, 3, 4, 5],
            [6, 7, 8, 9, 10],
            [11, 12, 13, 14, 0]]
    true_dist = torch.Tensor(data)
    padding_idx = [3, 2, 1]
    target = torch.tensor([1, 2, 3])

    true_dist.fill_(0.1 / (true_dist.size(1) - 2))
    true_dist.scatter_(1, target.data.unsqueeze(1).long(), 0.9)

    print(true_dist)
    print(target.data.unsqueeze(1).long())
    print(target.data)
    true_dist[:, padding_idx] = 0
    mask = torch.nonzero(target.data == torch.Tensor(padding_idx))

    print(true_dist)
    print(mask)
    print(target.data == torch.Tensor(padding_idx))

    if mask.dim() > 0:
        true_dist.index_fill_(0, mask.squeeze(), 0.0)

    print(true_dist)
