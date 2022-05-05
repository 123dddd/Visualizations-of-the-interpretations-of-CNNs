import torch
from torch.autograd import Variable
from .Base import Base
from torch.nn import  Conv2d
import torch.nn.functional as F

from .utils import tensor2cam, module2traced

class GradCam(Base):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # variable to store the gradients 
        self.gradients = None
        # variable to store the outputs of the last covoltuional layer
        self.conv_outputs = None
        
    # Store the outputs and gradients of the convolutional layer
    def store_outputs_and_grad(self, layer):
        # Layer: a layer in the model
        # Store gradients
        def store_grads(module, grad_in, grad_out):
            # grad_in: Gradients of the target class wrt. the input of the current layer
            # grad_out: Gradients of the target class wrt. the output of the current layer
            #           = grad_in*(gradient of layer output wrt. layer input)
            # store the gradients of the target class wrt. the outputs of this layer in self.gradients
            # [0] to get rid of the first channel
            self.gradients = grad_out[0]
        # store the outputs (feature maps) of this layer
        def store_outputs(module, input, outputs):
            self.conv_outputs = outputs 
                
        # Register the hooks in the forward pass to get the outputs
        layer.register_forward_hook(store_outputs)
        # Register the hooks in the backward pass to get the gradients
        layer.register_backward_hook(store_grads)


    def __call__(self, input_image, target_class, postprocessing = lambda x: x):
        # Zero gradients
        self.module.zero_grad()

        # Using the module2traced function to get a list of all the layers in the model
        modules = module2traced(self.module, input_image)
        # Find the last convolutional layer by enumerating
        for i, module in enumerate(modules):
            if isinstance(module, Conv2d): # Return whether an object is an instance of a class or of a subclass thereof.
                # Here the layer is the last convolutional layer
                layer = module
        
        # Register the hooks on this layer to obtain the outputs and the gradients of this layer 
        self.store_outputs_and_grad(layer)
        
        # Convert to Pytorch variable. By setting requires_grad=True, the tensors created by the torch.autograd.Variable class can form 
        # a backward graph that tracks every operations applied on them to calculate the gradients.
        input_var = Variable(input_image, requires_grad=True).to(self.device)
        # Forward pass 
        predictions = self.module(input_var)
        # Creat a empty tenor with the same size as predictions
        one_hot_output = torch.zeros(predictions.size()).to(self.device)
        # Clear the target class for backpropagation
        one_hot_output[0][target_class] = 1
        # Call backward on predictions
        # Becaues the predictions is non-scacle, so the 'one_hot_output' is used to transform it to a scalar for backpropagating
        predictions.backward(gradient = one_hot_output) 

        # After backward pass, no_grad() is used to disable gradient calculation
        with torch.no_grad():
            # Get the shape of the outputs of the last convolutional layer
            _, c, h, w = self.conv_outputs.shape
            # Pass the gradients of the last convolutional layer through a global average pooling operation
            avg_channel_grad = F.adaptive_avg_pool2d(self.gradients.data, 1)
            # avg_channel_grad.reshape((1, c)): Gives a new shape '(1, c)' to this array without changing its data.
            # self.conv_outputs.reshape((c, h * w)): Gives a new shape '(c, h * w)' to this array without changing its data.
            # @: matrix multiplication to obtain the cam heatmap
            cam = avg_channel_grad.reshape((1, c)) @ self.conv_outputs.reshape((c, h * w))
            # Pass the cam with size (h,w) through a ReLU function
            self.cam = F.relu(cam.reshape(h, w))
            # Overlaying the cam heatmap and original image
            image_with_heatmap = tensor2cam(postprocessing(input_image.squeeze().cpu()), self.cam)

        # Return the overlaying result
        return  image_with_heatmap, { 'prediction': target_class}


