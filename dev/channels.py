import torch
import torch.nn as nn
from typing import List, Tuple, Union

DISTINCT_VALUE = -100
loss_function = nn.MSELoss(reduction='none')

def compute_abstracted_loss(predictions: torch.Tensor,
                            ground_truth: torch.Tensor,
                            config: List[Tuple[List[int], int]]) -> torch.Tensor:
    """
    Compute the abstracted loss based on the provided configuration.

    Args:
    - predictions: Predicted tensor values of shape (batch_size, num_channels, height, width).
    - ground_truth: Ground truth tensor values.
    - config: Configuration for abstracted channels.

    Returns:
    - Torch tensor representing the abstracted loss.
    """
    final_size = predictions.size(1) + len(config)
    # Step 1: Initialize the mask with predictions' channels unmasked and aggregates masked.
    batch_size, _, height, width = predictions.size()
    include_mask = torch.ones((batch_size, final_size), dtype=torch.bool)
    include_mask[:, predictions.size(1):] = False

    # sort config by last element in each tuple
    config.sort(key=lambda x: x[1])

    for sources, target in config:
        # Perform concatenation
        combined_pred = predictions[:, sources, :, :].sum(dim=1, keepdim=True)
        predictions = torch.cat([predictions, combined_pred], dim=1)

        # Compute the channel mask for sources
        channel_mask = ~(ground_truth[:, sources, :, :] == DISTINCT_VALUE).any(dim=2).any(dim=2)

        # Update the aggregate mask for the target
        aggregate_mask_idx = predictions.size(1) - 1  # Last channel after concatenation
        include_mask[:, sources] = include_mask[:, sources] & channel_mask
        include_mask[:, aggregate_mask_idx] = include_mask[:, aggregate_mask_idx] | ~channel_mask.any(dim=1)

    # Apply masks and collapse losses
    individual_losses = loss_function(predictions, ground_truth)
    individual_losses = individual_losses.sum(dim=(2, 3))  # Collapse height and width dimensions
    masked_losses = individual_losses.masked_fill(~include_mask, 0)
    final_loss = masked_losses.sum() / batch_size

    return final_loss


# Generate new test cases and hardcoded calculations
# (Following the new logic and constraints)
# Note: This will also need an updated ground_truth tensor and configuration.


# Test cases
batch_size = 3
height, width = 1, 1
num_classes = 10
num_unique_classes = 7
predictions = torch.rand((batch_size, num_unique_classes, height, width))
ground_truth = torch.rand((batch_size, num_classes, height, width))

# Create some DISTINCT_VALUE (-100) entries
ground_truth[0, 0:3, :, :] = DISTINCT_VALUE
ground_truth[1, 4:6, :, :] = DISTINCT_VALUE

config = [((0, 1, 2), 8), ((2, 3), 9), ((4, 5), 7)]
loss = compute_abstracted_loss(predictions, ground_truth, config)

# Manually computing expected loss

# Step 1: Concatenate all combined channels to the predictions tensor
combined_ch_7 = predictions[:, 4:6, :, :].sum(dim=1, keepdim=True)  # Combining channels 4, 5 for entire batch
combined_ch_8 = predictions[:, 0:3, :, :].sum(dim=1, keepdim=True)  # Combining channels 0, 1, 2 for entire batch
combined_ch_9 = predictions[:, 2:4, :, :].sum(dim=1, keepdim=True)  # Combining channels 2, 3 for entire batch

# Append combined channels
predictions_with_combined = torch.cat([predictions, combined_ch_7, combined_ch_8, combined_ch_9], dim=1)

# Next, mask out channels based on the given rules for each item in the batch.
# Step 2: Mask out channels for each item in the batch based on the ground truth
include_mask = torch.ones_like(predictions_with_combined, dtype=torch.bool)
include_mask[:, num_unique_classes:] = False  # Mask out the combined channels

# Masking for item 0
include_mask[0, 0:3, :, :] = False
include_mask[0, 8, :, :] = True

# Masking for item 1
include_mask[1, 4:6, :, :] = False
include_mask[1, 7, :, :] = True

# Assuming the name of the combined predictions tensor is `predictions_with_combined`

# Compute the MSE loss
individual_losses = loss_function(predictions_with_combined, ground_truth)
individual_losses[~include_mask] = 0
final_loss = individual_losses.sum() / batch_size

print(f"The manual MSE loss is: {final_loss.item()}")
print(f"The other MSE loss is: {loss.item()}")