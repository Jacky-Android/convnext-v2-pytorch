from model import convnextv2_base
from torchinfo import summary

model = convnextv2_base(num_classes=2)
out = summary(model, (1, 3, 384,384))