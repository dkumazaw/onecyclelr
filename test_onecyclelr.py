import torch
from torchvision import models
import math
import unittest

from onecyclelr import OneCycleLR


class TestOneCycleLR(unittest.TestCase):
    def setUp(self):
        self.model = models.resnet18()
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=0.1, momentum=0.1)
        self.scheduler = OneCycleLR(
            self.optimizer,
            num_steps=1000,
            lr_range=(0.1, 1.),
            momentum_range=(0.85, 0.95),
            annihilation_frac=0.1,
            reduce_factor=0.01,
            last_step=-1
        )

    def test_internals(self):
        assert self.scheduler.num_cycle_steps == 900
        assert math.isclose(self.scheduler.final_lr, 0.1 * 0.01)
        assert math.isclose(self.scheduler.get_lr(), 0.1)
        assert math.isclose(self.scheduler.get_momentum(), 0.95)

    def test_step(self):
        # Scale up
        for i in range(450):
            self.scheduler.step()
        assert self.scheduler.last_step == 450
        assert math.isclose(self.scheduler.get_lr(), 1.)
        assert math.isclose(self.scheduler.get_momentum(), 0.85)

        # Scale down
        for i in range(450):
            self.scheduler.step()
        assert self.scheduler.last_step == 900
        assert math.isclose(self.scheduler.get_lr(), 0.1)
        assert math.isclose(self.scheduler.get_momentum(), 0.95)

        for i in range(100):
            self.scheduler.step()
        assert self.scheduler.last_step == 1000
        assert math.isclose(self.scheduler.get_lr(), 0.001)
        assert math.isclose(self.scheduler.get_momentum(), 0.95)

        # Go beyond the given num of steps: check if it works okay
        for i in range(50):
            self.scheduler.step()
        assert math.isclose(self.scheduler.get_lr(), 0.001)
        assert math.isclose(self.scheduler.get_momentum(), 0.95)
