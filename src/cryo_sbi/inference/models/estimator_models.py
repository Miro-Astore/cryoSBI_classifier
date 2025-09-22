import math
import os
from typing import Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class Classifier(nn.Module):
    def __init__(self, input_dim, out_dim, num_layers, nodes_per_layer, activation=nn.ReLU, dropout=0.0):
        super().__init__()
        self.classifier = nn.ModuleList()

        for i in range(num_layers):
            if i == 0:
                self.classifier.append(nn.Linear(input_dim, nodes_per_layer))
            else:
                self.classifier.append(nn.Linear(nodes_per_layer, nodes_per_layer))
                if dropout > 0.0:
                    self.classifier.append(nn.Dropout(dropout))
            self.classifier.append(activation())
        
        self.classifier.append(nn.Linear(nodes_per_layer, out_dim))
        self.classifier = nn.Sequential(*self.classifier)

    def forward(self, x):
        return self.classifier(x)


class PrototypeClassifier(nn.Module):
    def __init__(self, input_dim, out_dim, num_layers, nodes_per_layer,
                 activation=nn.ReLU, dropout=0.0, use_cosine=False):
        """
        input_dim: input feature dimension
        out_dim: number of classes
        num_layers: number of hidden layers
        nodes_per_layer: hidden layer size
        feat_dim: final embedding size (defaults to nodes_per_layer)
        use_cosine: if True, use cosine similarity; else negative squared distance
        """
        super().__init__()
        
        self.use_cosine = use_cosine
        self.input_dim = input_dim
        self.out_dim = out_dim

        # Learnable prototypes, one per class
        self.prototypes = nn.Parameter(torch.randn(self.out_dim, self.input_dim))
        print(f"Initialized PrototypeClassifier with {self.out_dim} classes, "
              f"embedding dim {self.input_dim}, use_cosine={use_cosine}")

    def forward(self, z, tau=1.0):

        if self.use_cosine:
            z = F.normalize(z, dim=-1)
            protos = F.normalize(self.prototypes, dim=-1)
            logits = z @ protos.T / tau  # cosine similarity scaled
        else:
            # Negative squared Euclidean distance
            # logits_{i,c} = -||z_i - proto_c||^2
            z2 = (z ** 2).sum(dim=1, keepdim=True)          # [B,1]
            p2 = (self.prototypes ** 2).sum(dim=1).unsqueeze(0)  # [1,C]
            logits = -(z2 + p2 - 2 * z @ self.prototypes.T) / tau

        return logits 


class ClassifierWithEmbedding(nn.Module):
    """Classification with embedding net

    Attributes:
        classifier: Classification model
        embedding (nn.Module): embedding net
    """

    def __init__(
        self,
        embedding_net: nn.Module,
        output_embedding_dim: int,
        num_classes: int = 2,
        num_layers: int = 5,
        nodes_per_layer: int = 128,
        prototype: bool = False,
        **kwargs,
    ) -> None:
        """
        Neural Posterior Estimation with embedding net.

        Args:
            embedding_net (nn.Module): embedding net
            output_embedding_dim (int): output embedding dimension
            num_transforms (int, optional): number of transforms. Defaults to 4.
            num_hidden_flow (int, optional): number of hidden layers in flow. Defaults to 2.
            hidden_flow_dim (int, optional): hidden dimension in flow. Defaults to 128.
            flow (nn.Module, optional): flow. Defaults to zuko.flows.MAF.
            theta_shift (float, optional): Shift of the theta for standardization. Defaults to 0.0.
            theta_scale (float, optional): Scale of the theta for standardization. Defaults to 1.0.
            kwargs: additional arguments for the flow

        Returns:
            None
        """

        super().__init__()
        if prototype:
            self.classifier = PrototypeClassifier(
                output_embedding_dim,
                num_classes,
                num_layers,
                nodes_per_layer,
                **kwargs,
            )
        else:   
            self.classifier = Classifier(
                output_embedding_dim,
                num_classes,
                num_layers,
                nodes_per_layer,
                **kwargs,
            )
        self.embedding = embedding_net()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the classifier model
        Args:
            theta (torch.Tensor): Conformational parameters.
            x (torch.Tensor): Image to condition the posterior on.
        Returns:
            torch.Tensor: unnormalized class probabilities.
        """

        return self.classifier(self.embedding(x))
    
    def prob(self, x: torch.Tensor) -> torch.Tensor:
        """
        Predict the class probabilities for the input data.
        
        Args:
            x (torch.Tensor): Input data.
        
        Returns:
            torch.Tensor: Class probabilities.
        """
        return torch.nn.functional.softmax(self.forward(x))
    
    def logits_embedding(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get the logits from the classifier and the embedding from the embedding net.

        Args:
            x (torch.Tensor): Input data.

        Returns:
            torch.Tensor: Logits from the classifier.
            torch.Tensor: Embedding from the embedding net.
        """
        embeddings = self.embedding(x)
        logits = self.classifier(embeddings)
        return logits, embeddings
