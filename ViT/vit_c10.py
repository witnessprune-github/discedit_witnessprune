import argparse

import comet_ml
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import pytorch_lightning as pl
import warmup_scheduler
import numpy as np

# from utils import get_model, get_dataset, get_experiment_name, get_criterion
from da import CutMix, MixUp
from vit import ViT

from collections import OrderedDict

import torchsummary as summary

import torch
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets, transforms
from torch.nn.utils import prune as prune
import torch
import torch.nn as nn
# import cvxpy as cp # type: ignore
import numpy as np

import numpy as np
import matplotlib.pyplot as plt

class SubsetDataset(Dataset):

    def __init__(self, dataset, indices):

        self.dataset = dataset
        self.indices = indices

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        # Fetch the dataset item via the stored index
        return self.dataset[self.indices[idx]]

def create_class_conditional_datasets(root, train=True, transform=None):
    # Load the CIFAR10 dataset with transformations
    full_dataset = datasets.CIFAR10(root=root, train=train, download=True, transform=transform)
    
    # Initialize containers for class-conditional datasets
    class_datasets = {}
    non_class_datasets = {}
    
    # Filter data by class and package into new datasets
    for class_id in range(10):  # CIFAR10 has 10 classes
        # Collecting indices for all samples belonging to the current class_id
        class_indices = [i for i, y in enumerate(full_dataset.targets) if y == class_id]
        non_class_indices = [i for i, y in enumerate(full_dataset.targets) if y != class_id]
        
        # Create a subset for each class using the indices
        class_datasets[class_id] = SubsetDataset(full_dataset, class_indices)
        non_class_datasets[class_id] = SubsetDataset(full_dataset, non_class_indices)
    
    return class_datasets, non_class_datasets

transform = transforms.Compose([
        transforms.ToTensor(),  # Convert images to PyTorch tensors
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Normalize the images
    ])
# class_datasets, non_class_datasets = create_class_conditional_datasets('./data', train=True, transform=transform)

# # DataLoader for class 0 and not class 0
# class_loader = DataLoader(class_datasets[0], batch_size=4, shuffle=True)
# non_class_loader = DataLoader(non_class_datasets[0], batch_size=4, shuffle=True)


def get_layer_activation(model, layer, data):

    with torch.no_grad():
        # Dictionary to store the activations from the forward hook
        activations = {}

        # Define the hook function
        def hook_function(module, input, output):
            activations['output'] = output.detach()
#             return activations['output']

        # Attach the hook to the layer
        handle = layer.register_forward_hook(hook_function)


        # Run the layer on the input data
        with torch.no_grad(): 
            out = model(data)  

        # Remove the hook after use
        
        act = activations['output']
#         t = torch.sum(torch.exp(torch.sum(activation, [2,3])),0)
    #     t1 = torch.exp(t)
        
        # Retrieve and return the captured activation
        
        handle.remove()
        return act

import torch

def get_layer_activation2(model, layer, data):
    with torch.no_grad():
        # Dictionary to store the activations from the forward hook
        activations = {}

        # Define the hook function
        def hook_function(module, input, output):
            activations['output'] = output.detach()
            print("Hook called. Output shape:", output.shape)

        # Attach the hook to the layer
        handle = layer.register_forward_hook(hook_function)

        # Set the model to evaluation mode
        model.eval()

        # Run the layer on the input data
        out = model(data)

        # Check if activations dictionary has been populated
        if 'output' not in activations:
            raise RuntimeError("The hook function was not called. Check the layer and input data.")

        # Retrieve and return the captured activation
        act = activations['output']

        # Remove the hook after use
        handle.remove()

        return act
    
def get_wit_moments(model, layer, data, batch_size):

    with torch.no_grad():
        # Dictionary to store the activations from the forward hook
        activations = {}

        # Define the hook function
        def hook_function(module, input, output):
            activations['output'] = output.detach()

        # Attach the hook to the layer
        handle = layer.register_forward_hook(hook_function)

        # Run the layer on the input data
        with torch.no_grad():  
            output = model(data)  

        # Remove the hook after use
        activation = activations['output']
        handle.remove()
        
        t1 = torch.sum(torch.exp(torch.sum(activation, [2,3])),0) 
        t2 = torch.sum(torch.exp(2*torch.sum(activation, [2,3])),0)
    #     t1 = torch.exp(t)
        t1.detach().numpy()
        t2.detach().numpy()
        t1 = t1/batch_size
        t2 = t2/batch_size
        
        return t1,t2


class WitnessPrune:
    def __init__(self, model, layer, budget):
        if not 0 <= budget <= 1:
            raise ValueError("Budget must be a real number between 0 and 1.")

        self.model = model
        self.layer = layer
        self.budget = budget
     

        # Validate if the specified layer is a convolutional layer
        # if not isinstance(self.layer, nn.Conv2d):
        #     raise ValueError("The specified layer is not a convolutional layer.")

    def get_basic_masks(self, saliencies):
        # Get the number of filters in the layer
        num_filters = self.layer.weight.shape[0]

        # Ensure the number of saliencies matches the number of filters
        if len(saliencies) != num_filters:
            raise ValueError("Number of saliencies must match the number of filters in the layer.")

        # Compute the number of filters to prune based on the budget
        num_pruned = int(torch.floor(torch.tensor(self.budget) * num_filters))

        # Sort saliencies and get the indices of the M_f smallest elements
        sorted_indices = torch.argsort(saliencies)

        # Define a basic mask where M_f smallest saliencies are set to 0 and the rest to 1
        basic_mask = torch.ones_like(saliencies)
        basic_mask[sorted_indices[-num_pruned:]] = 0

        return basic_mask

    def build_pruning_mask(self, basic_mask):
        # Get the number of filters in the layer
        num_filters = self.layer.weight.shape[0]

        # Ensure the length of the basic mask matches the number of filters
        if len(basic_mask) != num_filters:
            raise ValueError("Length of basic mask must match the number of filters in the layer.")

        # Create the pruning mask with the same shape as the layer's weight tensor
        pruning_mask = torch.ones_like(self.layer.weight)

        # Set elements corresponding to pruned filters to 0
        for i, mask_value in enumerate(basic_mask):
            if mask_value == 0:
                pruning_mask[i] = 0

        return pruning_mask
    
    def Prune2(self, basic_mask, pruning_mask):
        # Prune the bias of the convolutional layer using basic_mask
        if isinstance(self.layer, nn.Conv2d):
            torch.pruning_utils.prune.custom_from_mask(self.layer, name='bias', mask=basic_mask)

        # Prune the weights of the batch normalization layer associated with the current convolutional layer
        if hasattr(self.layer, 'bn'):
            prune.custom_from_mask(self.layer.bn, name='weight', mask=basic_mask)

        # Prune the biases of the batch normalization layer associated with the current convolutional layer
        if hasattr(self.layer, 'bn'):
            prune.custom_from_mask(self.layer.bn, name='bias', mask=basic_mask)

        # Prune the convolutional filters using pruning_mask
        prune.custom_from_mask(self.layer, name='weight', mask=pruning_mask)

    
    def Prune(self, pruning_mask, basic_mask, model, lnum):
        conv_layer = model.features[lnum]
        bn_layer = model.features[lnum + 1]

        # Prune the bias of the convolutional layer using basic_mask
        # if isinstance(conv_layer, nn.Conv2d):
        m1 = prune.custom_from_mask(conv_layer, name='bias', mask=basic_mask)
        m1 = prune.remove(m1, name='bias')
        # Prune the weights of the batch normalization layer associated with the current convolutional layer
        # if isinstance(bn_layer, nn.BatchNorm2d):
        m2 = prune.custom_from_mask(bn_layer, name='weight', mask=basic_mask)
        m2 = prune.remove(m2, name='weight')
        # Prune the biases of the batch normalization layer associated with the current convolutional layer
        # if isinstance(bn_layer, nn.BatchNorm2d):
        m3 = prune.custom_from_mask(bn_layer, name='bias', mask=basic_mask)
        m3 = prune.remove(m3, name='bias')
        # Prune the convolutional filters using pruning_mask
        m4 = prune.custom_from_mask(conv_layer, name='weight', mask=pruning_mask)
        m4 = prune.remove(m4, name='weight')
        return 
    
    def build_kernel_mask(self, basic_mask, clnum, nclnum, model):
        # Get the current and next convolutional layers from the model
        conv_layer = model.features[clnum]
        next_conv_layer = model.features[nclnum]

        # Get the number of input and output channels in the current and next convolutional layers
        num_input_channels = conv_layer.weight.shape[1]
        num_output_channels = conv_layer.weight.shape[0]

        # Ensure the length of basic_mask matches the number of output channels in the current layer
        if len(basic_mask) != num_output_channels:
            raise ValueError("Length of basic mask must match the number of output channels in the current layer.")

        # Initialize kernel mask with all 1s
        kernel_mask = torch.ones(next_conv_layer.weight.shape)

        # Iterate over the indices of basic_mask
        for i in range(len(basic_mask)):
            if basic_mask[i] == 0:
                # If the output channel is pruned, set the corresponding slice of the kernel mask tensor to 0
                kernel_mask[:, i, :, :] = 0
        return kernel_mask
    



def adjust_labels(dataset):
    # Adjust labels to ensure they are contiguous
    label_map = {old_label: new_label for new_label, old_label in enumerate(set(dataset.targets))}
    adjusted_labels = [label_map[label] for label in dataset.targets]
    dataset.targets = adjusted_labels
    return dataset

def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res
    
def get_accuracy2(model, dataloader, device):
    model.eval()
    correct = 0
    total = 0
    topk=(1,)
    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = correct / total
    return accuracy

def get_accuracy(model, dataloader, device):
    model.eval()
    correct = 0
    total = 0
    topk=(1,)
    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            maxk = max(topk)
            batch_size = labels.size(0)

            _, pred = outputs.topk(maxk, 1, True, True)
            pred = pred.t()
            correct = pred.eq(labels.view(1, -1).expand_as(pred))

            res = []
            for k in topk:
                correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
                correct_k.to(device='cpu').detach().numpy()
                res.append(correct_k.mul_(100.0 / batch_size))
        return res

    accuracy = correct / total
    return accuracy

def get_gauss_moments(model, layer, data, batch_size,sc): 
    
    def get_activation(name):
            def hook(model, input, output):
                activation[name] = output.detach().clone()
#         activation[name] = input[0].detach().clone()
            return hook
    
    with torch.no_grad():
        
        data.to(device='cuda')
        model.to(device='cuda')
        handle = layer.register_forward_hook(get_activation('output'))
#         print(next(model.parameters()).device)
#         print(data.is_cuda)
        # Run the layer on the input data
        with torch.no_grad():  
            activation = {}
            out = model(data)  
            outs = activation['output']
        

        
        
        
        
        t1 = torch.sum(torch.exp(-sc*torch.norm(outs, dim=[1])),0) 
        t2 = torch.sum(torch.exp(-2*sc*torch.norm(outs, dim=[1])),0)
    #     t1 = torch.exp(t)

        t1 = t1/batch_size
        t2 = t2/batch_size
        # Retrieve and return the captured activation
        
        handle.remove()
        
        return t1,t2
    
    
def get_gauss_moments2(model, layer, data, batch_size,sc): 
    
    def get_activation(name):
            def hook(model, input, output):
                activation[name] = output.detach().clone()
#         activation[name] = input[0].detach().clone()
            return hook
    
    with torch.no_grad():
        
        data.to(device='cuda')
        model.to(device='cuda')
        handle = layer.register_forward_hook(get_activation('output'))
#         print(next(model.parameters()).device)
#         print(data.is_cuda)
        # Run the layer on the input data
        with torch.no_grad():  
            activation = {}
            out = model(data)  
            outs = activation['output']
        
        
        
        t1 = torch.sum(torch.exp(-sc*(outs**2)) ,0) 
        t2 = torch.sum(torch.exp(-2*sc*(outs**2)),0)
    #     t1 = torch.exp(t)

        t1 = t1/batch_size
        t2 = t2/batch_size
        # Retrieve and return the captured activation
        
        handle.remove()
        
        return t1,t2
    

def main():
    transform = transforms.Compose([
        transforms.ToTensor(),  # Convert images to PyTorch tensors
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Normalize the images
        ])

    class_datasets, non_class_datasets = create_class_conditional_datasets('./data', train=True, transform=transform)
    test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    test_class_datasets, test_non_class_datasets = create_class_conditional_datasets('./data', train=False, transform=transform) 

    device='cuda'
    f = open('pruning_accs_vitc10.txt', 'a')
    for i in range(10):
        
        b,c,h,w = 4, 3, 32, 32

        model = ViT(in_c=c, num_classes= 10, img_size=h, patch=8, dropout=0.1, num_layers=7, hidden=384, head=12, mlp_hidden=384, is_cls_token=True)
        state_dict = torch.load('./ViT-CIFAR/weights/vit_c10_aa_ls.pth')
        new_state_dict = OrderedDict()
        for key in state_dict.keys():
            new_key = key.replace('model.', '')
            new_state_dict[new_key] = state_dict[key]
        
        # load params
        model.load_state_dict(new_state_dict, strict=False)
        model.eval()
        model.to(device)
        
        class_id = i
        tot_pruned = 0
        # model = torch.hub.load("chenyaofo/pytorch-cifar-models", "cifar10_vgg16_bn", pretrained=True)
        print(len(test_non_class_datasets[class_id]))
        class_loader = DataLoader(class_datasets[class_id], batch_size=64, shuffle=True)
        non_class_loader = DataLoader(non_class_datasets[class_id], batch_size=576, shuffle=True)
        test_class_loader = DataLoader(test_class_datasets[class_id], batch_size=980, shuffle=True)
        test_non_class_loader = DataLoader(test_non_class_datasets[class_id], batch_size=8192, shuffle=True)
        dataiter_class = iter(class_loader)
        dataiter_nonclass = iter(non_class_loader)
        test_dataiter_class = iter(test_class_loader)
        test_dataiter_nonclass = iter(test_non_class_loader)
        model.eval()
        model = model.to(device)
        for j in range(5):
            jj = j+2
            print(i,jj)
            
            with torch.no_grad():
                image,label = next(dataiter_class)
                image = image.to(device)
                label = label.to(device)
    #             print(image.is_cuda)
            #     output = model(image)
            #     output = output.to(device)
            #     print(output.shape)
                t1,t2 = get_gauss_moments(model, model.enc[jj].la1, image, 64, .05)

                image2,label2 = next(dataiter_nonclass)
                image2 = image2.to(device)
                label2 = label2.to(device)
    #             print(image2.is_cuda)
            #     output = model(image)
            #     output = output.to(device)
            #     print(output.shape)
                f1,f2 = get_gauss_moments(model, model.enc[jj].la1, image2, 576, .05)
                rr = (t1-f1)**2/(2*(t2 + f2))
                mu = torch.mean(rr)
                std = torch.std(rr)

                thresh = mu+1.375*std
                large = rr > thresh
                num_pruned = large.sum().item()
                tot_pruned = tot_pruned + num_pruned
                sorted_indices = torch.argsort(rr)
                basic_mask = torch.ones_like(rr)
                basic_mask[sorted_indices[-num_pruned:]] = 0
                
                w1 = model.enc[jj].la1.weight
                w1 = w1 * basic_mask
                b1 = model.enc[jj].la1.bias
                b1 = b1 * basic_mask

                with torch.no_grad():
                    model.enc[jj].la1.weight.copy_(w1)
                    model.enc[jj].la1.bias.copy_(w1)
                    
            with torch.no_grad():
                image,label = next(dataiter_class)
                image = image.to(device)
                label = label.to(device)
    #             print(image.is_cuda)
            #     output = model(image)
            #     output = output.to(device)
            #     print(output.shape)
                t1,t2 = get_gauss_moments(model, model.enc[jj].la2, image, 64, .05)

                image2,label2 = next(dataiter_nonclass)
                image2 = image2.to(device)
                label2 = label2.to(device)
    #             print(image2.is_cuda)
            #     output = model(image)
            #     output = output.to(device)
            #     print(output.shape)
                f1,f2 = get_gauss_moments(model, model.enc[jj].la2, image2, 576, .05)
                rr = (t1-f1)**2/(2*(t2 + f2))
                mu = torch.mean(rr)
                std = torch.std(rr)

                thresh = mu+1.25*std
                large = rr > thresh
                num_pruned = large.sum().item()
                tot_pruned = tot_pruned + num_pruned
                sorted_indices = torch.argsort(rr)
                basic_mask = torch.ones_like(rr)
                basic_mask[sorted_indices[-num_pruned:]] = 0
                
                w2 = model.enc[jj].la2.weight
                w2 = w2 * basic_mask
                b2 = model.enc[jj].la2.bias
                b2 = b2 * basic_mask

                with torch.no_grad():
                    model.enc[jj].la2.weight.copy_(w2)
                    model.enc[jj].la2.bias.copy_(w2)
            
        with torch.no_grad():
            image,label = next(dataiter_class)
            image = image.to(device)
            label = label.to(device)
    #             print(image.is_cuda)
        #     output = model(image)
        #     output = output.to(device)
        #     print(output.shape)
            t1,t2 = get_gauss_moments2(model, model.fc[0], image, 64, .05)

            image2,label2 = next(dataiter_nonclass)
            image2 = image2.to(device)
            label2 = label2.to(device)
    #             print(image2.is_cuda)
        #     output = model(image)
        #     output = output.to(device)
        #     print(output.shape)
            f1,f2 = get_gauss_moments2(model, model.fc[0], image2, 576, .05)

    #         print(f1, f2, t1, t2)
            rr = (t1-f1)**2/(2*(t2 + f2))
    #         print(rr.shape)
            mu = torch.mean(rr)
            std = torch.std(rr)


            thresh = mu+1.25*std
            large = rr > thresh
            num_pruned = large.sum().item()
            tot_pruned = tot_pruned + num_pruned
            sorted_indices = torch.argsort(rr)
            basic_mask = torch.ones_like(rr)
    #         print(basic_mask.shape, rr.shape, basic_mask)
            basic_mask[sorted_indices[-num_pruned:]] = 0

            w2 = model.fc[0].weight
            w2 = w2 * basic_mask
            b2 = model.fc[0].bias
            b2 = b2 * basic_mask
            with torch.no_grad():
                model.fc[0].weight.copy_(w2)
                model.fc[0].bias.copy_(b2)
                    
            
        model = model.to(device)
        test_dataiter_class = iter(test_class_loader)
        test_dataiter_nonclass = iter(test_non_class_loader)
        with torch.no_grad():
            image_tc,label_tc = next(test_dataiter_class)
            image_tc = image_tc.to(device)
            label_tc = label_tc.to(device)
    #             print(image.is_cuda)
            output_tc = model(image_tc)
            output_tc = output_tc.to(device)
    #             print(output.shape)

            acc_class = accuracy(output_tc, label_tc, topk=(1,))


            image_tn,label_tn = next(test_dataiter_nonclass)
            image_tn = image_tn.to(device)
            label_tn = label_tn.to(device)
    #             print(image.is_cuda)
            output_tn = model(image_tn)
            output_tn = output_tn.to(device)
    #             print(output.shape)

            acc_nonclass = accuracy(output_tn, label_tn, topk=(1,))

            f.write(f"c_id: {class_id}, cacc_old = {acc_class}, ncacc_old = {acc_nonclass}, pnum = {tot_pruned} \n")
            
if __name__ == "__main__":
    main()