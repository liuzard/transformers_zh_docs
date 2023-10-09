<!--Copyright 2022 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->

# Utilities for Image Processors

This page lists all the utility functions used by the image processors, mainly the functional
transformations used to process the images.

Most of those are only useful if you are studying the code of the image processors in the library.

## Image Transformations

[[autodoc]] image_transforms.center_crop

[[autodoc]] image_transforms.center_to_corners_format

[[autodoc]] image_transforms.corners_to_center_format

[[autodoc]] image_transforms.id_to_rgb

[[autodoc]] image_transforms.normalize

[[autodoc]] image_transforms.pad

[[autodoc]] image_transforms.rgb_to_id

[[autodoc]] image_transforms.rescale

[[autodoc]] image_transforms.resize

[[autodoc]] image_transforms.to_pil_image

## ImageProcessingMixin

[[autodoc]] image_processing_utils.ImageProcessingMixin
