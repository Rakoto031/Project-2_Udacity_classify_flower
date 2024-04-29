
import os.path
import argparse
import torch
from torch import nn
from torch import optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from collections import OrderedDict
from PIL import Image
from torch import Tensor
import os
import time


def input_args():
    ''' Created command line arguments and defined the argparse module. '''
    args = argparse.ArgumentParser()
    args.add_argument('--data_dir', default='flowers', type=str, help='directory to image folders')
    args.add_argument('--save_dir', default='.', help='directory to checkpoint file')
    args.add_argument('--arch', default='vgg16', type=str, choices=['vgg16', 'resnet50', 'alexnet'], help='model architecture from torchvision: choice of vgg16, alexnet, resnet50')
    args.add_argument('--hidden_units', default=4096, type=int, help='params of hidden units')
    args.add_argument('--learning_rate', default=0.001, type=float, help='learning rate')
    args.add_argument('--epochs', default=3, type=int, help='number of iteration over each training dataset')
    args.add_argument('--gpu', default=True, action='store_true', help='use GPU for training')

    return args.parse_args()

def load_dataset(data_dir):
    """
    Loads the dataset into a dataloader.
    Arguments:
        data_dir: Path to the folder where data stored. There are 3 sub folders named "train", "valid", "test".
    Returns:
        traindataloader: Train dataloader iterator.
        valdataloader: Validation dataloader iterator.
        testdataloader: test dataloader iterator
        class_to_idx:  mapping of flower class values to the flower indices on the image_train_datasets         
    """
    # Separate the data into training, validation, and test sets
    train_dir = os.path.join(data_dir, 'train')
    valid_dir = os.path.join(data_dir, 'valid')
    test_dir = os.path.join(data_dir, 'test')

    # Define the transforms for the training, validation, and testing sets
    train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                           transforms.RandomResizedCrop(224),
                                           transforms.RandomHorizontalFlip(),
                                           transforms.ToTensor(),
                                           transforms.Normalize([0.485, 0.456, 0.406],
                                                                [0.229, 0.224, 0.225])])

    valtest_transforms = transforms.Compose([transforms.Resize(256),
                                          transforms.CenterCrop(224),
                                          transforms.ToTensor(),
                                          transforms.Normalize([0.485, 0.456, 0.406],
                                                               [0.229, 0.224, 0.225])])

    # Load the datasets with ImageFolder
    image_train_datasets = datasets.ImageFolder(train_dir, transform=train_transforms)
    image_valid_datasets = datasets.ImageFolder(valid_dir, transform=valtest_transforms)
    image_test_datasets = datasets.ImageFolder(test_dir, transform=valtest_transforms)

    # Using the image datasets and the trainforms, define the dataloaders
    traindataloader = torch.utils.data.DataLoader(image_train_datasets, batch_size=64, shuffle=True)
    valdataloader = torch.utils.data.DataLoader(image_valid_datasets, batch_size=64, shuffle=True)
    testdataloader = torch.utils.data.DataLoader(image_test_datasets, batch_size=64, shuffle=True)

    # Load the mapping of classes to indices from the training dataset
    class_to_idx = image_train_datasets.class_to_idx

    return traindataloader, valdataloader, testdataloader, class_to_idx

def set_model(arch):
    ''' Retrieve a pre-trained model architecture and match it with its respective input size.
        Argument:
            arch: choose the ImageNet
        Returns:
            model: pre trained model
            input_size: the in_features at the first layer of the pre trained model'''
    if arch == 'vgg16':
        model = models.vgg16(pretrained=True)
        input_size = model.classifier[0].in_features        
    elif arch == 'alexnet':
        model = models.alexnet(pretrained=True)
        input_size = model.classifier[1].in_features         
    elif arch == 'resnet50':
        model = models.resnet50(pretrained=True)
        input_size = model.fc.in_features        
    else:
        raise ValueError('{} Unknown model, please choose vgg16 or alexnet or resnet50.'.format(arch))
    return model, input_size

def set_classifier(arch, model, input_size, hidden_units, learning_rate):
    ''' Build pretrained model by specifying, 
        Arguments:
            arch: the ImageNet
            model: pretrained model
            input size: input at the 1rst layer
            hidden units: parameter at the hidden layer
            learning rate: learning rate for the optimizer 
        Returns:
            model: model build
            criterion: to calculate the model error
            optimizer: to make regularisation to the model
      '''
    # Build a feed-forward classifier with ReLU, dropout and Softmax on the output
    output_size = 102    
    # choose fc for resnet classifier attribute
    if arch == 'resnet50':
        # Freeze parameters so the program doesn't backprop through them
        if hidden_units != None:
            print('Please choose "None" for resnet50 hidden layer. So set "--hidden_units None"')
        elif hidden_units == 4096:
            hidden_units = None            
        else:
            hidden_units = None
        
        for param in model.parameters():
            param.requires_grad = False
        classifier = nn.Sequential(nn.Linear(input_size, output_size),
                                          nn.LogSoftmax(dim=1))
        model.fc = classifier
        optimizer = optim.Adam(model.fc.parameters(), lr=learning_rate)
    else:
        for param in model.parameters():
            param.requires_grad = False
        classifier = nn.Sequential(OrderedDict([
                          ('fc1', nn.Linear(input_size, hidden_units)),
                          ('relu1', nn.ReLU()),
                          ('dropout1', nn.Dropout(p=0.5)),
                          ('fc2', nn.Linear(hidden_units, output_size)),
                          ('output', nn.LogSoftmax(dim=1))
                          ]))
        model.classifier = classifier
        optimizer = optim.Adam(model.classifier.parameters(), lr=learning_rate)
    # Define the criterion    
    criterion = nn.NLLLoss()        
    return model, criterion, optimizer

def test_validation(model, valdataloader, criterion, device):
    '''Step to validate the training process
    Arguments:
        model: after the model is trained , then pass on validation process 
        valdataloader: model uses the validation dataloader during the validation phase
        criterion: to identify the loss function
        device: choose between gpu or cpu
    Returns:
        test loss: to know the loss function
        accuracy: to know the perfomance of our model
        '''
    test_loss = 0
    accuracy = 0
    model.to(device)
                
    for inputs, labels in valdataloader:
        inputs, labels = inputs.to(device), labels.to(device)
        output=model.forward(inputs)
        loss=criterion(output, labels)
        test_loss += loss.item()                    
        #accuracy
        ps=torch.exp(output)        
        equality = (labels.data == ps.max(dim=1)[1])
        accuracy +=torch.mean(equality.type(torch.FloatTensor))       
    return test_loss, accuracy

def training_model(model, traindataloader, valdataloader, criterion, optimizer, epochs, device,patience=10, print_every=40 ):
    '''
    Train the network with a cross-validation pass.
    Arguments:
        model: model to train
        traindataloader: data uses the train dataloader in the training phase
        valdataloader: model uses the validation dataloader during the validation phase
        criterion: to identify the loss function
        optimizer: model regularization
        epochs: number of iteration over each training dataset
        device: choose between gpu or cpu
        patience: waiting time according iteration
        print_every: number of iteration needed to print
    Return:
        model: model trained and validated
    '''
    starttime = time.time()  
    step =0
    model.to(device)
    model.train()        
    for e in range(epochs):            
        running_loss =0            
        for inputs, labels in traindataloader:
            step +=1
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model.forward(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss +=loss.item()
            #Backward process
            if step % print_every == 0:
                model.eval()
                with torch.no_grad():
                    validtest_loss, accuracy = test_validation(model, valdataloader, criterion, device)                       
                print(f"Epoch {e+1}/{epochs}.. "
                          f"Training loss: {running_loss/print_every:.3f}... "
                          f"Validation loss: {validtest_loss/len(valdataloader):.3f}... "                
                          "Validation Accuracy: {:.3f}%".format((accuracy/len(valdataloader))*100))
                         
                model.train()
                running_loss=0
       
    time_elapsed = time.time() - starttime
    print('Training complete in {:.0f}m {:.0f}s'.format(
            time_elapsed // 60, time_elapsed % 60))
    return model

def test_accuracy(model, testdataloader, device):
    '''
     Implement the test pass for the trained model to test the accuracy of the model.
     Arguments:
        model: model trained need to test
        testdataloader: data uses for the test accuracy
        device: choose between gpu or cpu
    '''
    test_accuracy = 0
    model.eval()
    with torch.no_grad():
            
        for inputs, labels in testdataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            output=model.forward(inputs)
            #accuracy
            ps=torch.exp(output)            
            equality = (labels.data == ps.max(dim=1)[1])           
            test_accuracy +=torch.mean(equality.type(torch.FloatTensor)) 
    print("Test Accuracy: {:.3f}%".format((test_accuracy/len(testdataloader))*100))

def save_checkpoint(arch, model, input_size, output_size, hidden_units, class_to_idx, criterion, optimizer, learning_rate, epochs, save_dir, data_dir):
    ''' Save the trained and tested network for further use.
    Arguments:
        - model: model we save
        - model parameters: 
                        arch: the Imagenet
                        input_size: in features of our model input layer
                        output_size: output of our model output layer
                        hidden_units: parameter of our model hidden layer
                        class_to_idx: mapping of flower class values to the flower indices
                        criterion: loss function 
                        optimizer: model regularization 
                        learning_rate: step we want our optimizer make during the model regularization
                        epochs: number of iteration over each training dataset
                        save_dir: location of our pretrained model that we want to store
                        data_dir: data location
    Return:
       model: pretrained model saved 
    '''
    model.to('cpu')    
    _,_,_,model.class_to_idx = load_dataset(data_dir)
    if arch =='resnet50':
        model.fc = model.classifier
    else:
        model.classifier
    checkpoint = {'arch': arch,
                  'input_size': input_size,
                  'output_size': output_size,
                  'hidden_units': hidden_units,
                  'Learning_rate': learning_rate,
                  'criterion': criterion,
                  'optimizer': optimizer,
                  'epochs': epochs,                  
                  'classifier': model.classifier,
                  'class_to_idx':  model.class_to_idx,             
                  'state_dict': model.state_dict()
                 }
    torch.save(checkpoint, os.path.join(save_dir, 'checkpoint.pth'))    
    return model
    
def main():
    ''' Command Line arguments. '''
    in_args = input_args()
    output_size = 102
    # Enable GPU functionality, when available
    device = torch.device('cuda' if torch.cuda.is_available() and in_args.gpu else 'cpu')
    data_dir = in_args.data_dir
    print_every = 40
    
    model, input_size = set_model(in_args.arch)
    model, criterion, optimizer = set_classifier(in_args.arch, model, input_size, in_args.hidden_units, in_args.learning_rate)
    traindataloader, valdataloader, testdataloader, class_to_idx = load_dataset(data_dir)
    trained_model = training_model(model, traindataloader, valdataloader, criterion, optimizer, in_args.epochs , device, print_every)
    test_accuracy(model, testdataloader, device)
    create_model = save_checkpoint(in_args.arch, trained_model, input_size, output_size, in_args.hidden_units, class_to_idx, criterion, optimizer, in_args.learning_rate, in_args.epochs, in_args.save_dir, in_args.data_dir)

if __name__ == '__main__':
    main()