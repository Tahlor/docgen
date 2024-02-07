from matplotlib import pyplot as plt
from typing import List, Dict, Union, Tuple
from pathlib import Path
import torch
from typing import List, Union

def _list(x: Union[str, List[str], Tuple[str]]) -> List[str]:
    if isinstance(x, (tuple, list)):
        return list(x)
    return [x]

def _tuple(x: Union[str, List[str], Tuple[str]]) -> Tuple[str]:
    if isinstance(x, (tuple, list)):
        return tuple(x)
    return (x,)

def _str(x: Union[str, List[str], Tuple[str]]) -> str:
    return [el[0] if isinstance(el, (tuple, list)) else el for el in x]

def unravel(a_list):
    return [x if isinstance(y,list) else y for y in a_list for x in y ]

class NaiveChannelMapper:
    def __init__(self, output_channel_names: List[List[str]]=None,
                 input_channel_names: List[Union[str, List[str]]]=None,
                 data_fill_value=0):
        self.output_channel_names = [_list(x) for x in output_channel_names]
        self.input_channel_names = [_list(x) for x in input_channel_names]
        self.flat_output_channel_names = unravel(self.output_channel_names)
        self.flat_input_channel_names = unravel(self.input_channel_names)
        self.data_fill_value = data_fill_value

    def create_mapping(self):
        raise NotImplementedError

    def represent_gt(self, gt_tensor):
        return gt_tensor

    def __call__(self, gt_tensor):
        return self.represent_gt(gt_tensor)

    def convert_idx_config(self, idx_config):
        return idx_config


class SimpleChannelMapper(NaiveChannelMapper):
    """ This is a simple channel mapper that just aligns the channels in the GT with the model output channels
        It does not handle combined channels
        It does not adjust model outputs
        It merely aligns the GT channels with the model output channels

        CHANNELS IS THE FIRST DIMENSION
    """
    def __init__(self, output_channel_names: List[List[str]],
                 input_channel_names: List[Union[str, List[str]]],
                 data_fill_value=0):
        super().__init__(output_channel_names, input_channel_names, data_fill_value)

        self.mapping_from_input_to_output = self.create_mapping(
            list_of_channels_keys=self.input_channel_names,
            list_of_channels_values=self.flat_output_channel_names,
            match_any=False
        )
        self.mapping_from_output_to_input = self.create_mapping(
            list_of_channels_keys=self.output_channel_names,
            list_of_channels_values=self.flat_input_channel_names,
            match_any=True
        )

    def create_mapping_naive(self, list_of_channels_keys, list_of_channels_values):
        """Create a mapping from model output channels to GT channels."""
        mapping = {}
        for i, channel in enumerate(list_of_channels_keys):
            if channel in list_of_channels_values:
                gt_index = list_of_channels_values.index(channel)
                mapping[i] = gt_index
            else:
                mapping[i] = None
        return mapping

    def create_mapping(self, list_of_channels_keys, list_of_channels_values, match_any=False):
        """Create a mapping from model output channels to GT channels.

        list_of_channels_values: SHOULD BE FLATTENED

        1:m : allowable for 1 model key : many-GTs

        model_key : list of gt keys


        """
        _any = any if match_any else all
        mapping = {}
        for i, channel in enumerate(list_of_channels_keys):
            if _any([c in list_of_channels_values for c in _list(channel)]):
                gt_index = [list_of_channels_values.index(c) for c in channel if c in list_of_channels_values]
                mapping[i] = gt_index
            else:
                mapping[i] = None
        return mapping

    def represent_gt(self, gt_tensor):
        """Align the GT tensor with the model output dimensions."""
        output_tensor = torch.full([len(self.output_channel_names), *gt_tensor.shape[1:]], self.data_fill_value,
                                   dtype=gt_tensor.dtype, device=gt_tensor.device)
        for model_idx, gt_idx in self.mapping_from_output_to_input.items():
            if gt_idx is not None:
                if len(gt_idx) > 1:
                    output_tensor[model_idx] = torch.max(*[gt_tensor[_gt_idx] for _gt_idx in gt_idx])
                else:
                    output_tensor[model_idx] = gt_tensor[gt_idx]
        return output_tensor

    def convert_idx_config(self, idx_config):
        """An idx config is of the form of (1,2,(4,5)) which designates active channels 1,2,(4,5)
           We may remap these indices
        """
        return [self.mapping_from_input_to_output[x] for x in idx_config if self.mapping_from_input_to_output.get(x) is not None]

