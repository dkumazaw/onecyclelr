# One cycle policy learning rate scheduler
A PyTorch implementation of one cycle policy proposed in [Super-Convergence: Very Fast Training of Neural Networks Using Large Learning Rates](https://arxiv.org/abs/1708.07120).

## Introduction
This scheduler consists of three learning rate regimes: 1) upscale period, 2) downscale period, and 3) annihilation period.


The authors of the paper claim that by using this learning rate schedule, the total number of iterations needed until convergence can be reduced significantly; in their experiments, the total numbers of epochs needed to train a ResNet50 on ImageNet can be reduced from **100** to **20**.

![](paper_imagenet.png)
(Image Credit: https://arxiv.org/abs/1708.07120)

## How to use
The implementation has an interface similar to other common learning rate schedulers. 
```python
optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
scheduler = OneCycleLR(optimizer, num_steps=num_steps, lr_range=(0.1, 1.))
for epoch in range(epochs):
    for step, X in enumerate(train_dataloader):
        train(...) 
        scheduler.step()
```
