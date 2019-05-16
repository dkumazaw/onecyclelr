# One cycle policy learning rate scheduler
A PyTorch implementation of one cycle policy proposed in [Super-Convergence: Very Fast Training of Neural Networks Using Large Learning Rates](https://arxiv.org/abs/1708.07120).

## Usage
The implementation has an interface similar to other common learning rate schedulers. 
```python
from onecyclelr import OneCycleLR

optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
scheduler = OneCycleLR(optimizer, num_steps=num_steps, lr_range=(0.1, 1.))
for epoch in range(epochs):
    for step, X in enumerate(train_dataloader):
        train(...) 
        scheduler.step()
```
