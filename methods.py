import torch
import logging
import random

def normalize(x, means, stds):
    mean = random.choice(means)
    std = random.choice(stds)

    mean = mean.to(x.device).view(1, 3, 1, 1)
    std = std.to(x.device).view(1, 3, 1, 1)

    return (x - mean) / (std + 1e-8)

def normalize_class_conditional(x, y, means_per_client, stds_per_client):

    B = x.size(0)
    device = x.device
    num_clients = len(means_per_client)

    idx = torch.randint(0, num_clients, (B,), device=device)

    all_means = torch.stack([means_per_client[i] for i in idx.tolist()]).to(device)  # [B, num_classes, C]
    all_stds = torch.stack([stds_per_client[i] for i in idx.tolist()]).to(device)    # [B, num_classes, C]

    chosen_means = all_means[torch.arange(B), y]  # [B, C]
    chosen_stds = all_stds[torch.arange(B), y]    # [B, C]

    chosen_stds = chosen_stds + 1e-8

    chosen_means = chosen_means.view(B, -1, 1, 1)
    chosen_stds = chosen_stds.view(B, -1, 1, 1)

    return (x - chosen_means) / chosen_stds


def train_local(net_id, net, train_dataloader, epochs, lr, optimizer_type, weight_decay, device,
                means_per_client=None, stds_per_client=None):

    net.train()

    params = filter(lambda p: p.requires_grad, net.parameters())
    if optimizer_type == 'sgd':
        optimizer = torch.optim.SGD(params, lr=lr, momentum=0.9, weight_decay=weight_decay)
    else:
        raise ValueError(f"Unsupported optimizer type: {optimizer_type}")

    criterion = torch.nn.CrossEntropyLoss().to(device)

    for epoch in range(1, epochs + 1):
        epoch_losses = []

        for batch_idx, (x, target) in enumerate(train_dataloader):
            x, target = x.to(device), target.to(device).long()

            x = normalize_class_conditional(x, target, means_per_client, stds_per_client)

            out = net(x)
            loss = criterion(out, target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_losses.append(loss.item())
        
        epoch_loss = sum(epoch_losses) / (len(epoch_losses) + 1e-14)
        logging.info(f"[Client {net_id}] Epoch {epoch}/{epochs} | Loss = {epoch_loss:.4f}")

    return sum(epoch_losses) / (len(epoch_losses) + 1e-14)

def test(model, data_loader, loss_fun, device, means=None, stds=None):
    model.eval()
    loss_all = 0.0
    total = 0
    correct = 0

    with torch.no_grad():
        for data, target in data_loader:
            data, target = data.to(device), target.to(device)

            if means is not None and stds is not None:
                data = normalize(data, means, stds)

            output = model(data)
            loss = loss_fun(output, target)
            loss_all += loss.item()

            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()
            total += target.size(0)

    avg_loss = loss_all / (len(data_loader) + 1e-14)
    acc = correct / total
    return avg_loss, acc

