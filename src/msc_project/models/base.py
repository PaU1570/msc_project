"""Base classes to implement models."""

import torch
import torchvision
import aihwkit

import os
import numpy as np

BATCH_SIZE = 64
NUM_WORKERS = 23
PATH_DATASET = os.path.join("/scratch/msc24h18/msc_project/data", "DATASET")


class BaseModel:
    def __init__(self, model, seed=2024):
        self.model = model
        self.seed = seed
        self.optimizer = aihwkit.optim.AnalogSGD(model.parameters())
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=10)
        self.use_cuda = 0
        if aihwkit.simulator.rpu_base.cuda.is_compiled():
            self.use_cuda = 1
            model.cuda()
        self.device = torch.device("cuda" if self.use_cuda else "cpu")

    def get_dataset(self):
        raise NotImplementedError
    
    def get_analog_weights(self):
        try:
            # when apply_weight_scaling = False, the read weights are the raw conductance values (not scaled)
            return [tile.read_weights(apply_weight_scaling=False) for tile in self.model.analog_tiles()]
        except:
            print("Model does not have analog weights.")
            return None
        
    def preprocess_input(self, input):
        return input
    
    def train(self, train_set, valid_set, epochs=3, save_weights=False):
        metrics = np.zeros((epochs, 4))
        weights = [self.model.get_weights()]
        analog_weights = [self.get_analog_weights()]

        classifier = self.classifier

        for epoch_number in range(epochs):
            print(f"Epoch {epoch_number}:")
            total_loss = 0
            for i, (images, labels) in enumerate(train_set):
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                images = self.preprocess_input(images)

                self.optimizer.zero_grad()
                # Add training Tensor to the model (input).
                output = self.model(images)
                loss = classifier(output, labels)

                # Run training (backward propagation).
                loss.backward()

                # Optimize weights.
                self.optimizer.step()

                total_loss += loss.item()

            print("\t- Training loss: {:.16f}".format(total_loss / len(train_set)))

            # Save weights.
            if save_weights:
                weights.append(self.model.get_weights())
                analog_weights.append(self.get_analog_weights())

            # Evaluate the model.
            predicted_ok = 0
            total_images = 0
            val_loss = 0
            with torch.no_grad():
                for images, labels in valid_set:
                    # Predict image.
                    images = images.to(self.device)
                    labels = labels.to(self.device)

                    images = self.preprocess_input(images)
                    pred = self.model(images)
                    loss = classifier(pred, labels)
                    val_loss += loss.item()

                    _, predicted = torch.max(pred.data, 1)
                    total_images += labels.size(0)
                    predicted_ok += (predicted == labels).sum().item()

                print(f"\t- Validation loss: {val_loss / len(valid_set):.16f}")
                print(f"\t- Validation accuracy: {predicted_ok / total_images:.4f}")

            # Decay learning rate if needed.
            self.scheduler.step()

            # Update metrics.
            metrics[epoch_number, 0] = epoch_number
            metrics[epoch_number, 1] = total_loss / len(train_set) # train_loss
            metrics[epoch_number, 2] = val_loss / len(valid_set) # valid_loss
            metrics[epoch_number, 3] = predicted_ok / total_images # valid_accuracy

        self.metrics = metrics
        self.weights = weights
        self.analog_weights = analog_weights
        return metrics, weights, analog_weights
    
    def test(self, test_set):
        """Test trained network

        Args:
            model (nn.Model): Trained model to be evaluated
            test_set (DataLoader): Test set to perform the evaluation
        """
        # Setup counter of images predicted to 0.
        predicted_ok = 0
        total_images = 0

        self.model.eval()

        for images, labels in test_set:
            # Predict image.
            images = images.to(self.device)
            labels = labels.to(self.device)

            images = self.preprocess_input(images)
            pred = self.model(images)

            _, predicted = torch.max(pred.data, 1)
            total_images += labels.size(0)
            predicted_ok += (predicted == labels).sum().item()

        print("\nNumber Of Images Tested = {}".format(total_images))
        print("Model Accuracy = {}".format(predicted_ok / total_images))
        self.test_accuracy = predicted_ok / total_images


class BaseMNIST(BaseModel):
    def __init__(self, model, seed=2024, data_split=[0.8, 0.2], batch_size=BATCH_SIZE, num_workers=NUM_WORKERS):
        super().__init__(model, seed)
        trainval_set = torchvision.datasets.MNIST(PATH_DATASET, train=True, download=True, transform=torchvision.transforms.ToTensor())
        self.test_set = torchvision.datasets.MNIST(PATH_DATASET, train=False, download=True, transform=torchvision.transforms.ToTensor())
        
        self.train_set, self.valid_set = torch.utils.data.random_split(trainval_set, data_split, generator=torch.Generator().manual_seed(self.seed))
        
        self.train_loader = torch.utils.data.DataLoader(self.train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers)
        self.valid_loader = torch.utils.data.DataLoader(self.valid_set, batch_size=batch_size, shuffle=False, num_workers=num_workers)
        self.test_loader = torch.utils.data.DataLoader(self.test_set, batch_size=batch_size, shuffle=False, num_workers=num_workers)

        self.classifier = torch.nn.NLLLoss()

    def get_dataset(self):
        return self.train_loader, self.valid_loader, self.test_loader
    
    def preprocess_input(self, images):
        # Flatten MNIST images into a 784 vector.
        return images.view(images.shape[0], -1)
    

class BaseCIFAR10(BaseModel):
    def __init__(self, model, seed=2024, data_split=[0.8, 0.2], batch_size=BATCH_SIZE, num_workers=NUM_WORKERS):
        super().__init__(model, seed)

        mean = torch.Tensor([0.4914, 0.4822, 0.4465])
        std = torch.Tensor([0.2470, 0.2435, 0.2616])
        transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor(), torchvision.transforms.Normalize(mean, std)])
        
        trainval_set = torchvision.datasets.CIFAR10(PATH_DATASET, train=True, download=True, transform=transform)
        self.test_set = torchvision.datasets.CIFAR10(PATH_DATASET, train=False, download=True, transform=transform)
        
        self.train_set, self.valid_set = torch.utils.data.random_split(trainval_set, data_split, generator=torch.Generator().manual_seed(self.seed))
        
        self.train_loader = torch.utils.data.DataLoader(self.train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers)
        self.valid_loader = torch.utils.data.DataLoader(self.valid_set, batch_size=batch_size, shuffle=False, num_workers=num_workers)
        self.test_loader = torch.utils.data.DataLoader(self.test_set, batch_size=batch_size, shuffle=False, num_workers=num_workers)

        self.classifier = torch.nn.CrossEntropyLoss()

    def get_dataset(self):
        return self.train_loader, self.valid_loader, self.test_loader