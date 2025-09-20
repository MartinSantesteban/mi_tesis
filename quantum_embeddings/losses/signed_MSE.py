from ..quantum_knowledge_graph_embeddings.overlap_estimator import (SwapTestFidelityEstimator,
                                                                  QuantumForkingInnerProductRealPartEstimator,
                                                                  ComputeUncomputeFidelityEstimator)
from pykeen.losses import MSELoss

import torch

# BoolTensor: TypeAlias = torch.Tensor  
# FloatTensor: TypeAlias = torch.Tensor 
# LongTensor: TypeAlias = torch.Tensor  

class signed_MSELoss(MSELoss):
    def __init__(self, overlap_estimator=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.overlap_estimator = overlap_estimator

    def process_slcwa_scores(
        self,
        positive_scores: torch.Tensor,
        negative_scores: torch.Tensor,

        label_smoothing: float | None = None,
        batch_filter: torch.Tensor | None = None,
        num_entities: int | None = None,
        pos_weights: torch.Tensor | None = None,
        neg_weights: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # note: batch_filter
        #  - negative scores have already been pre-filtered
        #  - positive scores do not need to be filtered here
        # flatten and stack
        positive_scores = positive_scores.view(-1)
        negative_scores = negative_scores.view(-1)
        predictions = torch.cat([positive_scores, negative_scores], dim=0)

        # Fix
        if self.overlap_estimator == QuantumForkingInnerProductRealPartEstimator:
            labels = torch.cat([torch.ones_like(positive_scores), -torch.ones_like(negative_scores)])
        else: 
            labels = torch.cat([torch.ones_like(positive_scores), torch.zeros_like(negative_scores)])
            
        if pos_weights is None and neg_weights is None:
            weights = None
        else:
            # TODO: broadcasting?
            weights = torch.ones_like(predictions)
            if pos_weights is not None:
                weights[: len(positive_scores)] = pos_weights.view(-1)
            if neg_weights is not None:
                weights[len(positive_scores) :] = neg_weights.view(-1)
        return self.process_lcwa_scores(
            predictions=predictions,
            labels=labels,
            label_smoothing=label_smoothing,
            num_entities=num_entities,
            weights=weights,
        )