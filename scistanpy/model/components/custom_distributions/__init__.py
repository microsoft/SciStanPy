# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Custom distribution implementations for SciStanPy models.

This submodule provides enhanced and custom distribution implementations that
extend the capabilities of standard PyTorch and SciPy distributions. These
distributions address specific modeling needs and limitations in the standard
libraries while maintaining compatibility with SciStanPy's multi-backend
architecture.

The distributions maintain the standard PyTorch and SciPy interfaces while
providing additional functionality needed for advanced probabilistic modeling
scenarios. They are designed to be drop-in replacements for standard
distributions in most cases, with enhanced capabilities for specific use cases.
"""
