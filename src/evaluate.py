import os.path as osp
import torch


def evaluate(model, data_loader, criterion, device, load_path=None):
    if load_path:
        model.load_state_dict(torch.load(load_path))
    
    model.eval()
    total_loss = 0.0
    correct = 0

    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            labels = labels.view(-1, 1)

            outputs = model(inputs)
            loss = criterion(outputs, labels.float())
            total_loss += loss.item()

            preds = torch.sigmoid(outputs) > 0.5
            correct += (preds.eq(labels.float())).sum().item()

    avg_loss = total_loss / len(data_loader)
    accuracy = correct / len(data_loader.dataset)
    
    return avg_loss, accuracy



def calculate_metrics(model, model_path, data_loader, device):
    if osp.isfile(model_path):
        model.load_state_dict(torch.load(model_path))
    
    model.eval()
    true_positive = 0
    false_positive = 0
    false_negative = 0

    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            labels = labels.view(-1, 1)

            outputs = model(inputs)
            preds = torch.sigmoid(outputs) > 0.5

            true_positive += ((preds.eq(1)) & (labels.eq(1))).sum().item()
            false_positive += ((preds.eq(1)) & (labels.eq(0))).sum().item()
            false_negative += ((preds.eq(0)) & (labels.eq(1))).sum().item()

    total = true_positive + false_positive + false_negative
    accuracy = (true_positive + false_positive) / total if total > 0 else 0
    precision = true_positive / (true_positive + false_positive) if (true_positive + false_positive) > 0 else 0
    recall = true_positive / (true_positive + false_negative) if (true_positive + false_negative) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    return accuracy, precision, recall, f1_score