import torch
import os
from tqdm import tqdm
from fastprogress import master_bar, progress_bar 

def train_one_epoch(model, train_dataloader, device, loss_criteria, optimizer, mb):
    model.train()
    training_loss = 0
    exact_matches = 0
    total_samples = 0

    for batch_idx, (images, labels) in enumerate(progress_bar(train_dataloader, parent=mb)):
        images, labels = images.to(device), labels.to(device)

        # Zero the gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(images)

        # Compute loss
        loss = loss_criteria(outputs, labels)

        # Backward pass and optimizer step
        loss.backward()
        optimizer.step()

        # Accumulate training loss
        training_loss += loss.item()

        # Compute exact match accuracy
        probabilities = torch.sigmoid(outputs)  # Sigmoid for multi-label classification
        predictions = (probabilities > 0.5).float()
        exact_matches += torch.all(predictions == labels, dim=1).sum().item()
        total_samples += labels.size(0)

    # Compute average loss and exact match ratio
    avg_loss = training_loss / len(train_dataloader)
    exact_match_accuracy = exact_matches / total_samples * 100

    return avg_loss, exact_match_accuracy
    
def validate(model, val_dataloader, device, loss_criteria, mb):
    model.eval()
    val_loss = 0
    exact_matches = 0
    total_samples = 0

    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(progress_bar(val_dataloader, parent=mb)):
            images, labels = images.to(device), labels.to(device)

            # Forward pass
            outputs = model(images)

            # Compute loss
            loss = loss_criteria(outputs, labels)
            val_loss += loss.item()

            # Compute exact match accuracy
            probabilities = torch.sigmoid(outputs)  # Sigmoid for multi-label classification
            predictions = (probabilities > 0.5).float()
            exact_matches += torch.all(predictions == labels, dim=1).sum().item()
            total_samples += labels.size(0)

    # Compute average loss and exact match ratio
    avg_loss = val_loss / len(val_dataloader)
    exact_match_accuracy = exact_matches / total_samples * 100

    return avg_loss, exact_match_accuracy 
import time  # Import time module for timestamp

def train(model, train_dataloader, val_dataloader, device, loss_criteria, optimizer, epochs, checkpoint_dir=None):
    """
    Train the model and save the best checkpoint with a timestamp.
    """
    mb = master_bar(range(epochs))
    best_model_weights = None
    best_val_accuracy = 0.0

    # Ensure checkpoint directory exists
    if checkpoint_dir and not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    for epoch in mb:
        mb.main_bar.comment = f"Epoch {epoch + 1}"

        # Train for one epoch
        train_loss, train_accuracy = train_one_epoch(model, train_dataloader, device, loss_criteria, optimizer, mb)
        print(f"Epoch {epoch + 1}/{epochs}, Training Loss: {train_loss:.4f}, Training Accuracy: {train_accuracy:.2f}%")
        # Validation
        if val_dataloader is not None:
            val_loss, val_accuracy = validate(model, val_dataloader, device, loss_criteria, mb)
            print(f"Epoch {epoch + 1}/{epochs}, Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.2f}%")

        # Save the last model at the end of each epoch
    if checkpoint_dir:
        # Add timestamp to the checkpoint filename
        timestamp = time.strftime("%Y-%m-%d_%H-%M")
        checkpoint_path = os.path.join(checkpoint_dir, f"model_last_{timestamp}.pth")
        torch.save(model.state_dict(), checkpoint_path)
        print(f"Last model saved to {checkpoint_path}") 
            
        # # Validation
        # if val_dataloader is not None:
        #     val_loss, val_accuracy = validate(model, val_dataloader, device, loss_criteria, mb)
        #     print(f"Epoch {epoch + 1}/{epochs}, Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.2f}%")

        #     # Save the best model
        #     if val_accuracy > best_val_accuracy:
        #         best_val_accuracy = val_accuracy
        #         best_model_weights = model.state_dict()

        #         if checkpoint_dir:
        #             # Add timestamp to the checkpoint filename
        #             timestamp = time.strftime("%Y-%m-%d_%H-%M")
        #             checkpoint_path = os.path.join(checkpoint_dir, f"model_best_{timestamp}.pth")
        #             torch.save(best_model_weights, checkpoint_path)
        #             print(f"Best model saved to {checkpoint_path}")

    # Load the best model weights before returning
    # if best_model_weights:
    #     model.load_state_dict(best_model_weights)
    #     print("Best model weights loaded.")

    return model  # Return the trained model
 
    

def load_model_weights(model, weight_dict):
    model.load_state_dict(weight_dict)
    print("Model weights loaded successfully.")
    return model

 

 