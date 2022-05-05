import torch
import numpy as np
import cv2
from torchvision.transforms import Compose, Normalize

# device selected: GPU or CPU
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
# Preprocess->Normalize
image_net_preprocessing = Compose([
    Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# # inverse the normalization operation to obtain the original image
# class NormalizeInverse(Normalize):

#     def __init__(self, mean, std):
#         mean = torch.Tensor(mean)
#         std = torch.Tensor(std)
#         # inverse the std value
#         std_inv = 1 / std
#         # inverse the mean value
#         mean_inv = -mean * std_inv
#         # Use the inversed std and mean values to perform the Normalize operation on the normalized image data
#         super().__init__(mean=mean_inv, std=std_inv)

#     def __call__(self, tensor):
#         # Return a copy of the inversed normalized image data (original input image)
#         return super().__call__(tensor.clone())

# image_net_postprocessing = Compose([
#     NormalizeInverse(
#         mean=[0.485, 0.456, 0.406],
#         std=[0.229, 0.224, 0.225])
# ])

# Postprocess->inverse the normalization operation
image_net_postprocessing = Compose([Normalize(mean = [ 0., 0., 0. ], std = [ 1/0.229, 1/0.224, 1/0.225 ]),
                                Normalize(mean = [ -0.485, -0.456, -0.406 ], std = [ 1., 1., 1. ]),
                               ]) # let's try a clever way

# transform tensor to cam 
def tensor2cam(image, cam):

    image_with_heatmap = image2cam(image.squeeze().permute(1,2,0).cpu().numpy(),# permute(): change the order of channels from (c,h,w) to (h,w,c)
              cam.detach().cpu().numpy())
    # transform the numpy array to tensor
    return torch.from_numpy(image_with_heatmap).permute(2,0,1) # change the order of channels from (h,w,c) back to (c,h,w)

# transform image to cam
def image2cam(image, cam):
    # Get the shape of the input image
    h, w, c = image.shape
    # Normalize between 0-1
    cam -= np.min(cam)
    cam /= np.max(cam)
    # Resize the cam to the size of the input image
    cam = cv2.resize(cam, (w,h))
    # modify the data-type to uint8
    cam = np.uint8(cam * 255.0)
    # apply the colormap JET
    img_with_cam = cv2.applyColorMap(cam, cv2.COLORMAP_JET)
    # convert the BGR (used by opencv) image to RGB image
    img_with_cam = cv2.cvtColor(img_with_cam, cv2.COLOR_BGR2RGB)
    # overlaying
    img_with_cam = img_with_cam + (image * 255)
    # Normalize between 0-1
    img_with_cam /= np.max(img_with_cam)
    
    # return the overlayed image
    return img_with_cam


# a function to trace the CNN model for getting a list of all the layers
def module2traced(module, inputs):
    # module: CNN model; inputs: specfic input image
    # Creat empty list
    modules = []
    
    def trace(module, inputs, outputs):
        # Store the module (layer) in the list
        modules.append(module)
        
    def traverse(module):
        # Returns an iterator over immediate children modules.
        for m in module.children():
            # recursion: trace each sub-module
            traverse(m)  
        # Define the layer without further nested layers as leaf layer
        is_leaf = len(list(module.children())) == 0
        # Register the hook on the leaf layers
        if is_leaf: module.register_forward_hook(trace)
    
    # Search each layers of the module
    traverse(module)
    # Forward pass trigger
    _ = module(inputs)
    # Return the flatted modules
    return modules
