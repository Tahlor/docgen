import random
from typing import List, Any, Callable, Tuple, Union
from pathlib import Path
import numpy as np
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler())


class LayerSampler:
    def __init__(self, generators: List[Callable[[], Any]],
                 layer_weights: List[float],
                 default_min_layers: int = None,
                 default_max_layers: int = None,
                 number_of_layer_weights: List[int] = None):
        """
        Initialize the LayerSampler with word_image_generators and weights.

        Args:
            generators (List[Callable[[], Any]]): List of generator functions.
            layer_weights (List[float]): Non-normalized weights associated with each generator.
            default_min_layers (int, optional): Default minimum number of layers to be selected. Defaults to None.
            default_max_layers (int, optional): Default maximum number of layers to be selected. Defaults to None.
            number_of_layer_weights (List[int], optional): Weights associated with the number of layers. Defaults to None.
        """
        self.generators = generators
        self.weights = [w / sum(layer_weights) for w in layer_weights]  # Normalize the weights
        self.min_layers = default_min_layers if default_min_layers is not None else min(2, len(generators))
        self.max_layers = min(default_max_layers, len(generators)) if default_max_layers is not None else len(generators)

        if default_max_layers > len(generators):
            print(f"WARNING: default_max_layers ({default_max_layers}) is greater than the number of word_image_generators ({len(generators)}). Setting default_max_layers to {len(generators)}.")

        self.number_of_layer_weights = number_of_layer_weights


    def choose_num_layers(self) -> int:
        """
        Randomly choose the number of layers between min_layers and max_layers.

        Args:
            min_layers (int): Minimum number of layers to be selected.
            max_layers (int): Maximum number of layers to be selected.

        Returns:
            int: The chosen number of layers.
        """
        if self.number_of_layer_weights:
            if len(self.number_of_layer_weights) < self.max_layers - self.min_layers + 1:
                self.max_layers = len(self.number_of_layer_weights) + self.min_layers - 1
                logger.error("Length of number_of_layer_weights should be equal to max_layers - min_layers + 1.")
                logger.error(f"Setting max_layers to {self.max_layers}")

            return random.choices(range(self.min_layers, self.max_layers + 1)
                                  , weights=self.number_of_layer_weights[:self.max_layers-self.min_layers+1])[0]
        else:
            return random.randint(self.min_layers, self.max_layers)

    def _sample(self, num_layers: int=None, sample_with_replacement: bool=False) -> List[Callable[[], Any]]:
        """
        Sample layers based on their weights.

        Args:
            num_layers (int): Number of layers to be sampled.
            sample_with_replacement (bool): If True, sampling is done with replacement. Else, without replacement.

        Returns:
            List[Callable[[], Any]]: List of sampled generator functions.
        """
        if sample_with_replacement:
            return random.choices(self.generators, weights=self.weights, k=num_layers)
        else:
            #return list(set(random.choices(self.generators, weights=self.weights, k=min(num_layers, len(self.generators)))))
            # sample using weights, but without replacement
            choices = np.random.choice(len(self.generators), size=min(num_layers, len(self.generators)),
                                    replace=False, p=np.array(self.weights)/sum(self.weights))
            return [self.generators[i] for i in choices]


    def sample(self, replacement: bool=False) -> List[Callable[[], Any]]:
        layers = self.choose_num_layers()
        return self._sample(num_layers=layers, sample_with_replacement=replacement)


def main(min_layers: int, max_layers: int, with_replacement: bool):
    # Example word_image_generators (they could be any functions you wish)
    def gen1(): return "Layer1"

    def gen2(): return "Layer2"

    def gen3(): return "Layer3"

    generators = [gen1, gen2, gen3]
    weights = [2, 3, 1]

    sampler = LayerSampler(generators, weights)

    num_layers = sampler.choose_num_layers(min_layers, max_layers)
    chosen_generators = sampler.sample(num_layers, with_replacement)

    for gen in chosen_generators:
        print(gen())


if __name__ == "__main__":
    # Example usage
    main(min_layers=6, max_layers=9, with_replacement=True)