import argparse
import torch
from torchvision import datasets, transforms, models
import numpy as np
import matplotlib.pyplot as plt
import json
from PIL import Image

# from train import model_arch
def input_args():
    ''' Created command line arguments and defined the argparse module. '''
    args = argparse.ArgumentParser()
    args.add_argument('--image_path', type=str, help='Directory of the image for predictions', default = 'flowers/test/58/image_02738.jpg')
    args.add_argument('--model_checkpoint', type=str, help='checkpoint to load the classifier', default = 'checkpoint.pth')
    args.add_argument('--top_k', type=int, help='Choose number K to liste the top most likely classes', default=5)
    args.add_argument('--category_names', type=str, help='Map category to real name', default='cat_to_name.json')
    args.add_argument('--gpu', action='store_true', help='use GPU for inference', default=True)    
    return args.parse_args()

def loading_checkpoint(filepath, device):
    """
    Loads the model from a checkpoint file.
    Arguments: 
        filepath: Path to checkpoint file    
    Returns: 
        pre_trained_model: loaded model.
        pre_trained_model.class_to_idx: Index to class mapping for further evaluation.
    """
    # Enable choice of CPU or GPU functioning    
    if device == 'cuda' and torch.cuda.is_available():
        checkpoint = torch.load(filepath)
    else:
        checkpoint = torch.load(filepath, map_location=lambda storage, loc: storage)
    # Load pretrained model    
    if checkpoint['arch'] == 'vgg16':
        pre_trained_model = models.vgg16(pretrained=True)                
    elif checkpoint['arch'] == 'alexnet':
        pre_trained_model = models.alexnet(pretrained=True)                 
    elif checkpoint['arch'] == 'resnet50':
        pre_trained_model = models.resnet50(pretrained=True)
    else:
        raise ValueError('{} Unknown model, please choose vgg16 or alexnet or resnet50.'.format(checkpoint['arch']))
    
    pre_trained_model.arch=checkpoint['arch']
    pre_trained_model.classifier = checkpoint['classifier']
    pre_trained_model.class_to_idx = checkpoint['class_to_idx']   
     # Loading the parameters
    pre_trained_model.input_size: checkpoint['input_size']
    pre_trained_model.output_size: checkpoint['output_size']
    pre_trained_model.hidden_units: checkpoint['hidden_units']
    pre_trained_model.Learning_rate: checkpoint['learning_rate']
    pre_trained_model.epochs = checkpoint['epochs']    
    #pre_trained_model optimizer
    pre_trained_model.criterion=checkpoint['criterion']
    pre_trained_model.optimizer=checkpoint['optimizer']
    # load model
    pre_trained_model.load_state_dict(checkpoint['state_dict'])  
    return pre_trained_model, pre_trained_model.class_to_idx

def process_image(image):
    ''' Function that Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns image transformed
    '''    
    # TODO: Process a PIL image for use in a PyTorch model
    PIL_image = Image.open(image)
   
    image_transforms = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])    
    img = image_transforms(PIL_image)
    return img

def predict(image_path, model, class_to_idx, top_k, device='cpu'):
    ''' 
    Predict the class (or classes) of an image using a trained deep learning model.    
    Arguments:
        image_path: Path to the image to predict
        model: Trained model
        class_to_idx: mapping of flower class values to the flower indices
        top_k: number of selection at the top
        device: choose gpu or cpu
    Returns:
        probs: the probability preticted to those classes
        top_classes: Top k class numbers.
    '''
    model.eval()
  
    # Implement the code to predict the class from an image file    
    image_tensor = process_image(image_path)
    model_input = image_tensor.unsqueeze(0)
    #model = model.double()
    model_input = model_input.float()
    
    with torch.no_grad():
        output = model.forward(model_input)
        prob_output=torch.exp(output)
    
        probs, classes = prob_output.topk(top_k)  
        probs = probs.detach().numpy()  
        probs = probs.tolist()[0] #converting both to list        
   # Swap the dictionary keys and values to obtain the numerical classes of the flower names
    mapping = {val: key for key, val in model.class_to_idx.items()}    
    top_classes = [mapping[each] for each in classes.cpu().numpy()[0]]    
    return probs, top_classes

def predict_names(classes, category_names):
    ''' Map the category (or categories) of an image to the category names. 
    Arguments:
        -classes: class number of the flower
        -category_names: category corresponding of the class number
    Returns:
        - names: the class name
      '''
    # Load a mapping from category label to category name    
    with open(category_names, 'r') as f:
        cat_to_name = json.load(f)    
    names = []
    for c in classes:
        names.append(cat_to_name[c])    
    return names

def main():
    ''' Create & retrieve Command Line arguments. '''
    in_args = input_args()    
    device = torch.device('cuda:0' if torch.cuda.is_available() and in_args.gpu else 'cpu')    
    model, class_idx = loading_checkpoint(in_args.model_checkpoint, device)
    probs, classes = predict(in_args.image_path, model, class_idx, in_args.top_k, device='cpu')
    names = predict_names(classes, in_args.category_names)    
  
    for i in range(in_args.top_k):
        print("Top: {}/{}.. ".format(i+1, in_args.top_k),
            "Class number: {}..".format(classes[i]),
            "Class name: {}----- ".format(names[i]),
            "Probability: {:.3f}% ".format(probs[i]*100))

if __name__ == '__main__':
	main()