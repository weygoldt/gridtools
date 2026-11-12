"""Curated library of real chirp templates for the grid simulator.

Build once with :func:`build_library`, then load with :func:`load_library`
and draw templates with :meth:`TemplateLibrary.sample`. :func:`write_report`
documents a built library (needs the ``plot`` extra).
"""

from gridtools.templates.amplitude import (
    DEFAULT_DIP_RANGE,
    amplitude_from_frequency,
    random_dips,
)
from gridtools.templates.build import build_library
from gridtools.templates.config import BuildConfig
from gridtools.templates.library import (
    DEFAULT_EXCLUDE,
    EQUALIZE_DURATION_COLUMN,
    EqualizationBins,
    TemplateCycler,
    TemplateLibrary,
    TemplateSample,
    load_library,
)
from gridtools.templates.report import write_report

__all__ = [
    "DEFAULT_DIP_RANGE",
    "DEFAULT_EXCLUDE",
    "EQUALIZE_DURATION_COLUMN",
    "BuildConfig",
    "EqualizationBins",
    "TemplateCycler",
    "TemplateLibrary",
    "TemplateSample",
    "amplitude_from_frequency",
    "build_library",
    "load_library",
    "random_dips",
    "write_report",
]
