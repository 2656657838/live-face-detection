# Copyright (c) Alibaba, Inc. and its affiliates.
from typing import TYPE_CHECKING

from modelscope.utils.import_utils import LazyImportModule
# from .base import Preprocessor

# else:
_import_structure = {
    'base': ['Preprocessor'],
    'builder': ['PREPROCESSORS', 'build_preprocessor'],
    'common': ['Compose', 'ToTensor', 'Filter'],
    'asr': ['WavToScp'],
    'image': [
        'LoadImage', 'load_image', 'ImageColorEnhanceFinetunePreprocessor',
        'ImageInstanceSegmentationPreprocessor',
        'ImageDenoisePreprocessor', 'ImageDeblurPreprocessor'
    ],
    'cv': [
        'ImageClassificationMmcvPreprocessor',
        'ImageRestorationPreprocessor',
        'ControllableImageGenerationPreprocessor'
    ],
    'kws': ['WavToLists'],
    'multi_modal': [
        'DiffusionImageGenerationPreprocessor', 'OfaPreprocessor',
        'MPlugPreprocessor', 'HiTeAPreprocessor', 'MplugOwlPreprocessor',
        'ImageCaptioningClipInterrogatorPreprocessor'
    ],
}

import sys

sys.modules[__name__] = LazyImportModule(
    __name__,
    globals()['__file__'],
    _import_structure,
    module_spec=__spec__,
    extra_objects={},
)
