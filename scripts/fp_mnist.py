import torch
import torchvision
import os
import numpy as np
import argparse
import pandas as pd
import pickle 

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 64
NUM_WORKERS = 23
SEED = 2024
PATH_DATASET = os.path.join("/scratch/msc24h18/msc_project/data", "DATASET")

def get_dataset(data_split=(0.8, 0.2)):
    trainval_set = torchvision.datasets.MNIST(PATH_DATASET, train=True, download=True, transform=torchvision.transforms.ToTensor())
    test_set = torchvision.datasets.MNIST(PATH_DATASET, train=False, download=True, transform=torchvision.transforms.ToTensor())
    
    train_set, valid_set = torch.utils.data.random_split(trainval_set, data_split, generator=torch.Generator().manual_seed(SEED))
    
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
    valid_loader = torch.utils.data.DataLoader(valid_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

    return train_loader, valid_loader, test_loader

def create_model(input_size, hidden_sizes, output_size):
    model = torch.nn.Sequential(
        torch.nn.Linear(input_size, hidden_sizes[0], True),
        torch.nn.Sigmoid(),
        torch.nn.Linear(hidden_sizes[0], hidden_sizes[1], True),
        torch.nn.Sigmoid(),
        torch.nn.Linear(hidden_sizes[1], output_size, True),
        torch.nn.LogSoftmax(dim=1)
    )
    return model

def train(model, optimizer, scheduler, train_set, valid_set, epochs=3, save_weights=False):
        metrics = np.zeros((epochs, 4))
        weights = [[parameter.data.cpu().numpy() for parameter in model.parameters()]]

        classifier = torch.nn.NLLLoss()

        for epoch_number in range(epochs):
            print(f"Epoch {epoch_number}:")
            total_loss = 0
            for i, (images, labels) in enumerate(train_set):
                images = images.to(DEVICE)
                labels = labels.to(DEVICE)
                
                images = images.view(images.shape[0], -1)

                optimizer.zero_grad()
                # Add training Tensor to the model (input).
                output = model(images)
                loss = classifier(output, labels)

                # Run training (backward propagation).
                loss.backward()

                # Optimize weights.
                optimizer.step()

                total_loss += loss.item()

            print("\t- Training loss: {:.16f}".format(total_loss / len(train_set)))

            # Save weights.
            if save_weights:
                weights.append([parameter.data.cpu().numpy() for parameter in model.parameters()])

            # Evaluate the model.
            predicted_ok = 0
            total_images = 0
            val_loss = 0
            with torch.no_grad():
                for images, labels in valid_set:
                    # Predict image.
                    images = images.to(DEVICE)
                    labels = labels.to(DEVICE)

                    images = images.view(images.shape[0], -1)
                    pred = model(images)
                    loss = classifier(pred, labels)
                    val_loss += loss.item()

                    _, predicted = torch.max(pred.data, 1)
                    total_images += labels.size(0)
                    predicted_ok += (predicted == labels).sum().item()

                print(f"\t- Validation loss: {val_loss / len(valid_set):.16f}")
                print(f"\t- Validation accuracy: {predicted_ok / total_images:.4f}")

            # Decay learning rate if needed.
            scheduler.step()

            # Update metrics.
            metrics[epoch_number, 0] = epoch_number
            metrics[epoch_number, 1] = total_loss / len(train_set) # train_loss
            metrics[epoch_number, 2] = val_loss / len(valid_set) # valid_loss
            metrics[epoch_number, 3] = predicted_ok / total_images # valid_accuracy

        return metrics, weights

def test(model, test_set):
        """Test trained network

        Args:
            model (nn.Model): Trained model to be evaluated
            test_set (DataLoader): Test set to perform the evaluation
        """
        # Setup counter of images predicted to 0.
        predicted_ok = 0
        total_images = 0

        model.eval()

        for images, labels in test_set:
            # Predict image.
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            images = images.view(images.shape[0], -1)
            pred = model(images)

            _, predicted = torch.max(pred.data, 1)
            total_images += labels.size(0)
            predicted_ok += (predicted == labels).sum().item()

        print("\nNumber Of Images Tested = {}".format(total_images))
        print("Model Accuracy = {}".format(predicted_ok / total_images))
        test_accuracy = predicted_ok / total_images
        return test_accuracy

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a Feedforward Neural Network on MNIST dataset")
    parser.add_argument("--output_dir", type=str, help="Output directory")
    parser.add_argument("--epochs", type=int, default=25, help="Number of epochs")
    parser.add_argument("--lr", type=float, default=0.01, help="Learning rate")
    parser.add_argument("--save_weights", action='store_true', help="Save weights")
    args = parser.parse_args()

    # Define the model
    input_size = 784
    hidden_sizes = [256, 128]
    output_size = 10

    model = create_model(input_size, hidden_sizes, output_size)
    model.to(DEVICE)

    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

    train_loader, valid_loader, test_loader = get_dataset()

    metrics, weights = train(model, optimizer, scheduler, train_loader, valid_loader, epochs=args.epochs, save_weights=True)
    test_accuracy = test(model, test_loader)

    test_acc_vec = np.zeros(shape=(metrics.shape[0], 1))
    test_acc_vec[-1] = test_accuracy

    metrics = np.concatenate((metrics, test_acc_vec), axis=1)

    if args.output_dir is not None:
        if not os.path.exists(args.output_dir):
            os.makedirs(args.output_dir)
        df = pd.DataFrame(metrics, columns=["epoch", "train_loss", "val_loss", "val_acc", "test_acc"])
        df.to_csv(os.path.join(args.output_dir, 'metrics.csv'), index=False)
        if args.save_weights:
            with open(os.path.join(args.output_dir, 'weights.pkl'), 'wb') as f:
                pickle.dump(weights, f)



    