import torch

def evaluate(model, dataset: str, dataloader, criterion, print_output=False):
    # Set model to evaluation mode
    model.eval()

    # Initialize evaluation metrics
    total_loss = 0.0
    num_samples = 0

    # Loop over the data
    for inputs, labels in dataloader:
        if dataset == 'yahoo':
            labels = labels.long()
        elif dataset == 'imdb':
            labels = labels.to(dtype=torch.float64)
        elif dataset == 'agnews':
            pass
        else:
            raise ValueError
            # Forward pass
        with torch.no_grad():
            outputs = model(inputs)
            loss = criterion(outputs, labels)

        # Accumulate evaluation metrics
        total_loss += loss.item() * inputs.size(0)
        num_samples += inputs.size(0)

    # Compute average loss
    avg_loss = total_loss / num_samples

    # Print evaluation metrics
    if print_output:
        print("Loss: {:.4f}".format(avg_loss))
    return avg_loss
