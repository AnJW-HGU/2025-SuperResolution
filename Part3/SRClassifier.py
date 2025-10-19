from torch import nn, optim

class SRClassifier(nn.Module):
    def __init__(self, generator, classifier):
        super().__init__()
        self.generator = generator
        self.classifier = classifier

    def forward(self, x):
        sr = self.generator(x)
        pred = self.classifier(sr)
        return pred, sr