from typing import Type

import numpy as np
import xarray as xr

import torch


class BaseChunkSpec:
    PREFIX = ''

    def __init__(self):
        self.__tag = None
        self.__names = None
        self.__range = None
        self.__dtype = None

    @property
    def tag(self) -> str:
        if self.__tag is None:
            raise NotImplementedError(
                f'Define {self.__class__.__name__}.tag'
            )

        return self.__tag

    @tag.setter
    def tag(self, value: str):
        if not isinstance(value, str):
            raise TypeError(
                f'Invalid type for "tag": {type(value)}'
            )

        if not value.startswith(self.PREFIX):
            value = f'{self.PREFIX}.{value}'

        self.__tag = value

    @property
    def names(self) -> list[str]:
        if self.__names is None:
            raise NotImplementedError(
                f'Define {self.__class__.__name__}.names'
            )

        return self.__names

    @names.setter
    def names(self, value):
        if not isinstance(value, (list, tuple)):
            raise TypeError(
                f'Invalid type for "names"'
            )
        if not all((isinstance(name, str) for name in value)):
            raise TypeError(
                f'Invalid type for "names"'
            )

        self.__names = list(value)

    @property
    def range(self) -> tuple[int, int]:
        if self.__range is None:
            raise NotImplementedError(
                f'Define {self.__class__.__name__}.range'
            )

        return self.__range

    @range.setter
    def range(self, value: tuple[int, int]):
        if not isinstance(value, (tuple, list)):
            raise TypeError(
                f'Invalid type for "range": {type(value)}'
            )

        if not isinstance(value[0], int):
            raise TypeError(
                f'Invalid type for "range[0]": {type(value)}'
            )

        if not isinstance(value[1], int):
            raise TypeError(
                f'Invalid type for "range[1]": {type(value)}'
            )

        if value[0] >= value[1]:
            raise ValueError('range[0] >= range[1]')

        self.__range = tuple(value)

    @property
    def dtype(self) -> np.dtype:
        if self.__dtype is None:
            raise NotImplementedError(
                f'Define {self.__class__.__name__}.dtype'
            )

        return self.__dtype

    @dtype.setter
    def dtype(self, value: np.dtype):
        if not isinstance(np.dtype(value), np.dtype):
            raise TypeError(
                f'Invalid type for "dtype": {type(value)}'
            )

        self.__dtype = np.dtype(value)


class EncodingChunkSpec(BaseChunkSpec):
    PREFIX = 'encoding'

    def __init__(self, tag, names, range_, dtype):
        self.tag = tag
        self.names = names
        self.range = range_
        self.dtype = dtype


class DecodingChunkSpec(BaseChunkSpec):
    PREFIX = 'decoding'

    def __init__(self, tag, names, range_, dtype):
        self.tag = tag
        self.names = names
        self.range = range_
        self.dtype = dtype


class LabelChunkSpec(BaseChunkSpec):
    PREFIX = 'label'

    def __init__(self, tag, names, range_, dtype):
        self.tag = tag
        self.names = names
        self.range = range_
        self.dtype = dtype


class ChunkExtractor:
    def __init__(self, df, chunk_specs):
        # Check tag duplication.
        tags = [spec.tag for spec in chunk_specs]
        if len(tags) != len(set(tags)):
            raise ValueError(
                f'Tags are duplicated. {[s.tag for s in chunk_specs]}.'
            )

        self.chunk_specs = chunk_specs
        self.chunk_min_t = min(spec.range[0] for spec in chunk_specs)
        self.chunk_max_t = max(spec.range[1] for spec in chunk_specs)
        self.chunk_length = self.chunk_max_t - self.chunk_min_t

        self.time_index_values = np.arange(len(df))

        self._preprocess(df)

    def _preprocess(self, df):
        self.data = {}
        for spec in self.chunk_specs:
            values = df[spec.names].astype(spec.dtype).values
            if len(values.shape) == 1:
                values = values[:, np.newaxis]
            self.data[spec.tag] = values

    def extract(self, start_time_index, return_time_index=False):
        assert start_time_index + self.chunk_min_t >= 0

        times = self.time_index_values[
            start_time_index + self.chunk_min_t :
            start_time_index + self.chunk_max_t
        ]

        chunk_dict = {}
        for spec in self.chunk_specs:
            array = self.data[spec.tag][
                start_time_index + self.chunk_min_t :
                start_time_index + self.chunk_max_t
            ]

            range_ = (
                spec.range[0] - self.chunk_min_t,
                spec.range[1] - self.chunk_min_t,
            )

            chunk_dict[spec.tag] = array[slice(*range_)]
            # Time index information.
            if return_time_index:
                chunk_dict[f'{spec.tag}.time_index'] = times[slice(*range_)]

        return chunk_dict


class ChunkInverter:
    def __init__(self, chunk_specs: list[BaseChunkSpec]):
        """Class to convert tensor to DataFrame."""
        self.chunk_specs = chunk_specs

        # Core tag to names dict.
        self.core_tag_dict = {}
        for spec in chunk_specs:
            core_tag = spec.tag.split('.')[1]

            if core_tag not in self.core_tag_dict:
                self.core_tag_dict[core_tag] = spec.names
            else:
                a = np.array(self.core_tag_dict[core_tag])
                b = np.array(spec.names)
                assert np.all(a ==  b)

        self.tag_dict = {spec.tag: spec.names for spec in chunk_specs}

    def _convert_to_numpy(self, tensor: torch.Tensor | np.ndarray):
        if isinstance(tensor, np.ndarray):
            return tensor
        elif isinstance(tensor, torch.Tensor):
            return tensor.cpu().numpy()
        else:
            raise TypeError(f'Invalid type for tensor {type(tensor)}')

    def _infer_shape(self, tensor: np.ndarray):
        pass

    def invert(self, tag: str, tensor: torch.Tensor | np.ndarray):
        if tag in self.tag_dict:
            names = self.tag_dict[tag]
        elif tag in self.core_tag_dict:
            names = self.core_tag_dict[tag]
        elif tag.split('.')[1] in self.core_tag_dict:
            names = self.core_tag_dict[tag.split('.')[1]]
        else:
            raise ValueError(f'"{tag}" not in chunk_specs.')

        data = self._convert_to_numpy(tensor)

        da = xr.DataArray(
            data=data,
            dims=('batch_index', 'time_index', 'feature'),
            name='',
        )

        # TODO: support for multi-dimensional features (like video).

        df = da.to_dataframe().unstack(2)
        df.columns = names

        return df

    def invert_dict(
        self,
        data: dict[str, torch.Tensor | np.ndarray],
    ):
        outputs = {}
        for tag, tensor in data.items():
            print(tag)
            outputs[tag] = self.invert(tag, tensor)

        return outputs