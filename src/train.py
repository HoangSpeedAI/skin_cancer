from datetime import datetime
import os
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm

from evaluate import evaluate
from utils import *


def train(model, train_loader, val_loader, criterion, optimizer,
          num_epochs, device, architecture, base_dir, config):
    model.train()
    best_val_loss = float('inf')
    train_losses = []
    val_losses = []
    lrs = []
    best_epoch = 0
    best_accuracy = 0.0
    best_model_path = None

    session_dir = create_session_directory(base_dir, architecture, config)

    # learning rate scheduler
    scheduler = create_scheduler(optimizer, config)
    assert scheduler is not None, f'Expected valid scheduler'

    for epoch in range(num_epochs):
        running_loss = 0.0

        with tqdm(total=len(train_loader), desc=f"Epoch {epoch + 1}/{num_epochs}",
                  unit="batch") as pbar:
            for inputs, labels in train_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                labels = labels.view(-1, 1)

                optimizer.zero_grad()
                outputs = model(inputs)

                loss = criterion(outputs, labels.float())
                loss.backward()

                # Gradient Clipping
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

                optimizer.step()

                running_loss += loss.item()
                current_lr = optimizer.param_groups[0]['lr']  # Get current learning rate
                lrs.append(current_lr)
                pbar.set_postfix(loss=running_loss / (pbar.n + 1), lr=current_lr)
                pbar.update(1)

        avg_train_loss = running_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        val_loss, accuracy = validate(model, val_loader, criterion, device)
        val_losses.append(val_loss)

        print(f"Epoch [{epoch + 1}/{num_epochs}], "
              f"Train Loss: {avg_train_loss:.4f}, "
              f"Validation Loss: {val_loss:.4f}, "
              f"Accuracy: {accuracy:.4f}")
        
        # Step the scheduler
        scheduler.step()

        if val_loss < best_val_loss:
            # Remove the previous best model if it exists
            if best_model_path is not None and os.path.exists(best_model_path):
                os.remove(best_model_path)
                print(f"--> Removed old best model at {best_model_path}")

            best_val_loss = val_loss
            best_epoch = epoch + 1
            best_accuracy = accuracy
            model_name = f"{architecture}_ep{best_epoch}_{best_accuracy:.4f}.pth"
            best_model_path = os.path.join(session_dir, model_name)
            torch.save(model.state_dict(), best_model_path)
            print(f"--> Saved best model to {best_model_path}")

    plot_losses(train_losses, val_losses, session_dir)
    plot_learning_rates(lrs, session_dir)

    return best_model_path


def validate(model, val_loader, criterion, device):
    return evaluate(model, val_loader, criterion, device)
