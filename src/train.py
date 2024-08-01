import torch
# import torch.optim as optim
# import torch.nn as nn

def train(model, train_loader, val_loader, criterion, optimizer, num_epochs, device):
    model.train()
    for epoch in range(num_epochs):
        # Training phase
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            labels = labels.view(-1, 1)  # Ensure labels are shaped [batch_size, 1]

            optimizer.zero_grad()
            outputs = model(inputs)

            loss = criterion(outputs, labels.float())  # Ensure labels are float
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        # Calculate average loss for this epoch
        avg_loss = running_loss / len(train_loader)
        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {avg_loss:.4f}")

        # Validation phase
        validate(model, val_loader, criterion, device)

def validate(model, val_loader, criterion, device):
    model.eval()  # Set model to evaluation mode
    val_loss = 0.0
    correct = 0

    with torch.no_grad():  # No gradient calculation during validation
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            labels = labels.view(-1, 1)  # Ensure labels are shaped [batch_size, 1]

            outputs = model(inputs)
            loss = criterion(outputs, labels.float())
            val_loss += loss.item()

            # Get predictions
            preds = torch.sigmoid(outputs) > 0.5  # Apply sigmoid and threshold
            correct += (preds.eq(labels.float())).sum().item()

    avg_val_loss = val_loss / len(val_loader)
    accuracy = correct / len(val_loader.dataset)
    
    print(f"Validation Loss: {avg_val_loss:.4f}, Accuracy: {accuracy:.4f}")