import torch

for i in range(10):
    # Mock environment setup
    torch.manual_seed(0)  # For reproducible results

    # Randomly generate mock data
    composite_image = torch.rand(3, 10, 10)  # Example image
    composite_image_before_new_layer = torch.rand(3, 10, 10)  # Example before layer image
    mask_for_current_dataset = torch.randint(0, 2, (10, 10))  # Random binary mask

    start_y, start_x = 2, 3  # Example start coordinates
    img_height, img_width = 5, 5  # Example dimensions
    self_how_much_darker = 0.8  # Example attribute value

    # Calculating slices
    slice_y = slice(start_y, min(start_y + img_height, composite_image.shape[1]))
    slice_x = slice(start_x, min(start_x + img_width, composite_image.shape[2]))

    # Original method (compact)
    mask = mask_for_current_dataset[slice_y, slice_x].bool()
    ignore_index_compact = torch.zeros_like(mask_for_current_dataset, dtype=torch.float32)
    ignore_index_compact[slice_y, slice_x] = torch.max(ignore_index_compact[slice_y, slice_x],
                                                       mask & (torch.mean(composite_image[:, slice_y, slice_x], dim=0)
                                                               > self_how_much_darker * torch.mean(composite_image_before_new_layer[:, slice_y, slice_x], dim=0))
                                                       )

    # Revised method (expanded)
    mask = mask_for_current_dataset[slice_y, slice_x].bool()
    mean_img_after = torch.mean(composite_image[:, slice_y, slice_x], dim=0)
    adjusted_mean_threshold = self_how_much_darker * torch.mean(composite_image_before_new_layer[:, slice_y, slice_x], dim=0)
    new_mask_after_pasting = mask & (mean_img_after > adjusted_mean_threshold -5 )
    ignore_index_expanded = torch.zeros_like(mask_for_current_dataset, dtype=torch.float32)
    ignore_index_expanded[slice_y, slice_x] = torch.max(ignore_index_expanded[slice_y, slice_x], new_mask_after_pasting)

    # Comparing results
    comparison = torch.equal(ignore_index_compact, ignore_index_expanded)
    assert comparison
