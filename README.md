# Coding-Prompts
A series of prompts for improving the performance of coding agent. These prompts emphasize readability and maintainability, as well as in soft engineering, aiming to generate better code.

- [Deep Learning](./deep_learning.md)

## Example Output Template
```python
# data/loader.py
from torch.utils.data import Dataset, DataLoader
from typing import Tuple, Dict


class CustomDataset(Dataset):
    """Data loader with dynamic augmentation support
    
    Args:
        config: Dictionary containing transform parameters
        mode: Runtime mode (train/val/test)
    """
    
    def __init__(self, config: Dict, mode: str = 'train'):
        self.transforms = build_augmentations(config[mode])
        
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        try:
            # Implement actual data loading
            return image, label
        except Exception as e:
            logger.error(f"Data loading failed: {str(e)}")
            raise CustomDataError("Data exception handling")


def create_loader(config: Dict) -> DataLoader:
    """Create distributed-ready DataLoader"""
    dataset = CustomDataset(config)
    sampler = DistributedSampler(dataset) if distributed else None
    return DataLoader(dataset, 
                     batch_size=config.batch_size,
                     sampler=sampler)
```