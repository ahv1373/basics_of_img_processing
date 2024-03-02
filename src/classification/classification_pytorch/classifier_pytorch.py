import os
from typing import Tuple

import cv2
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from sklearn.metrics import confusion_matrix
from torch import nn
from torch import optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms


class CustomDataLoaderTorch(Dataset):  # Need to inherit from torch.utils.data.Dataset to use DataLoader
    def __init__(self, images_dir: str = os.path.join("..", "..", "dataset", "cat_and_dog", "train"),
                 transform: transforms.Compose = transforms.Compose(
                     [transforms.ToPILImage(), transforms.Resize((224, 224)), transforms.ToTensor()])):
        self.total_images = [os.path.join(images_dir, file) for file in os.listdir(images_dir)]
        self.transform = transform

    def __len__(self):  # This method is called when you do len(instance_of_the_class)
        return len(self.total_images)

    def __getitem__(self, idx):  # This method is called when you iterate over the dataset
        img_ = cv2.imread(self.total_images[idx])
        img_ = cv2.cvtColor(img_, cv2.COLOR_BGR2RGB)
        label = 0 if "cat" in os.path.basename(self.total_images[idx]).lower() else 1
        if self.transform:
            img_ = self.transform(img_)
        return img_, label


class CustomCNNClassifierTorchModel(nn.Module):  # Need to inherit from torch.nn.Module to create a model
    def __init__(self, num_classes: int = 2, input_shape: Tuple[int, int, int] = (224, 224, 3)):
        super(CustomCNNClassifierTorchModel, self).__init__()
        self.num_classes = num_classes
        self.input_shape = input_shape
        # use batch normalization to speed up the training process

        self.conv1 = nn.Conv2d(in_channels=self.input_shape[2], out_channels=32, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.relu3 = nn.ReLU()

        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(64 * (input_shape[0] // 4) * (input_shape[1] // 4), 64)  # Adjusted for maxpooling
        self.bn4 = nn.BatchNorm1d(64)
        self.relu4 = nn.ReLU()
        self.fc2 = nn.Linear(64, num_classes)
        # # since we are using BCELoss, we don't need to use the softmax activation function in the last layer
        # self.sigmoid = nn.Sigmoid()

    def forward(self, x):  # This method is called when you pass the input data to the model and is required
        x = self.pool1(self.relu1(self.bn1(self.conv1(x))))
        x = self.pool2(self.relu2(self.bn2(self.conv2(x))))
        x = self.relu3(self.bn3(self.conv3(x)))
        x = self.flatten(x)
        x = self.relu4(self.bn4(self.fc1(x)))
        x = self.fc2(x)
        return x


if __name__ == "__main__":
    # Set hyperparameters
    LEARNING_RATE = 1e-3
    NUMBERS_OF_EPOCHS = 10
    BATCH_SIZE = 32
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {DEVICE}")

    PERFORM_TRAINING = False
    PERFORM_EVALUATION = False
    PERFORM_INFERENCE = True

    # Load the data
    train_dataset = CustomDataLoaderTorch(images_dir=os.path.join("..", "..", "dataset", "cat_and_dog", "train"))
    test_dataset = CustomDataLoaderTorch(images_dir=os.path.join("..", "..", "dataset", "cat_and_dog", "test"))

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    if PERFORM_TRAINING is True:
        # Create the model
        model = CustomCNNClassifierTorchModel().to(DEVICE)  # move the model to the device
        # set model to training mode
        model.train()

        # Define the loss and the optimizer
        criterion = nn.CrossEntropyLoss()  # This is used to calculate the loss
        optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

        # Train the model
        for epoch in range(NUMBERS_OF_EPOCHS):
            for index, (input_data, input_label) in enumerate(train_loader):
                input_data, input_label = input_data.to(DEVICE), input_label.to(DEVICE)  # move the data to the device
                optimizer.zero_grad()  # This is used to reset the gradients to zero before backpropagation
                output = model(input_data)
                loss = criterion(output, input_label)
                loss.backward()  # This is used to perform backpropagation
                optimizer.step()  # This is used to update the weights
                if index % 10 == 0:  # Print the loss every 10 batches
                    print(f"Epoch {epoch}, batch {index}, loss: {loss.item()}")

        print("Finished Training")
        # save the model
        torch.save(model.state_dict(), "custom_cnn_torch.pth")  # *.pth is the extension for PyTorch models

    if PERFORM_EVALUATION:
        # load the model
        model = CustomCNNClassifierTorchModel().to(DEVICE)  # to load the model, you need to create an instance of the model
        model.load_state_dict(torch.load("custom_cnn_torch.pth"))  # load the weights of the model using the load_state_dict method
        model.eval()  # set the model to evaluation mode. This switch is important because some layers like dropout behave differently in training and evaluation mode

        # Evaluate the model and determine confusion matrix
        tp, tn, fp, fn = 0, 0, 0, 0
        with torch.no_grad():  # This is used to disable the gradient calculation to speed up the process
            for input_data, input_label in test_loader:
                input_data, input_label = input_data.to(DEVICE), input_label.to(DEVICE)
                output = model(input_data)
                _, predicted = torch.max(output, 1)
                for i in range(len(predicted)):
                    if input_label[i] == 1 and predicted[i] == 1:
                        tp += 1
                    elif input_label[i] == 0 and predicted[i] == 0:
                        tn += 1
                    elif input_label[i] == 0 and predicted[i] == 1:
                        fp += 1
                    elif input_label[i] == 1 and predicted[i] == 0:
                        fn += 1

        accuracy = (tp + tn) / (tp + tn + fp + fn)
        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        f1_score = 2 * (precision * recall) / (precision + recall)
        # use sklearn to calculate the confusion matrix

        y_true = [1] * (tp + fn) + [0] * (tn + fp)
        y_pred = [1] * (tp + fp) + [0] * (tn + fn)
        confusion_matrix_ = confusion_matrix(y_true, y_pred)
        # visualize the confusion matrix

        sns.heatmap(confusion_matrix_, annot=True, fmt="d")
        # save and show
        print(f"True Positives: {tp}, True Negatives: {tn}, False Positives: {fp}, False Negatives: {fn}")
        print(f"Accuracy: {accuracy}, Precision: {precision}, Recall: {recall}, F1 Score: {f1_score}")
        plt.savefig("confusion_matrix.png", dpi=300)
        plt.show()

    if PERFORM_INFERENCE:
        # load the model
        model = CustomCNNClassifierTorchModel().to(DEVICE)
        model.load_state_dict(torch.load("custom_cnn_torch.pth"))
        model.eval()
        # load some images and perform inference
        img_dir = os.path.join("..", "..", "dataset", "cat_and_dog", "train")
        for img_name in os.listdir(img_dir):
            img = cv2.imread(os.path.join(img_dir, img_name))
            img_copy = img.copy()
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = transforms.Compose([transforms.ToPILImage(), transforms.Resize((224, 224)), transforms.ToTensor()])(
                img)
            img = img.unsqueeze(0).to(DEVICE)
            output = model(img)
            _, predicted = torch.max(output, 1)
            predicted_label = "cat" if predicted.item() == 0 else "dog"
            # visualize the image
            cv2.putText(img_copy, f"Predicted: {predicted_label}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            cv2.imshow("Inference", img_copy)
            key = cv2.waitKey(0)
            if key == ord("q"):
                break
