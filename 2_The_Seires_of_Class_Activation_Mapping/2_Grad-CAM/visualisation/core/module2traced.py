import torch
from torchvision import models

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
        if is_leaf: 
            module.register_forward_hook(trace)
    
    # Search each layers of the module
    traverse(module)
    # Forward pass trigger
    _ = module(inputs)
    # Return the flatted modules
    return modules

if __name__ == "__main__":
    net = models.squeezenet1_0(pretrained=True)
    # Put model in evaluation mode
    net.eval()

    fake_input = torch.randn((1,3,224,224))

    modules_traced = module2traced(net, fake_input)

    print(modules_traced[:5])