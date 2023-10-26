from docgen.transforms.transforms import *



group_transform = A.OneOf([
    VerticalFlip(p=1),  # Vertical Flip
    HorizontalFlip(p=1),  # Horizontal Flip
    Rotate(limit=(180, 180), p=1),  # 180-degree rotation
    lambda x: x  # Identity
], p=1)

pipeline = A.Compose([
    RandomBottomLeftEdgeCropSquare(p=1),
    group_transform(),
])
