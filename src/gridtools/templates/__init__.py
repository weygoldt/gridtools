"""Curated library of real chirp templates for the grid simulator.

Build once with :func:`build_library`, then load with :func:`load_library`
and draw templates with :meth:`TemplateLibrary.sample`. :func:`write_report`
documents a built library (needs the ``plot`` extra).
:func:`assign_template_splits` splits a library by fish into train, val and
test (:mod:`gridtools.templates.splits`).
"""

from gridtools.templates.amplitude import (
    DEFAULT_DIP_RANGE,
    amplitude_from_frequency,
    random_dips,
)
from gridtools.templates.build import build_library
from gridtools.templates.config import BuildConfig
from gridtools.templates.library import (
    BRIEF_CYCLES_MAX,
    DEFAULT_ALPHA,
    DEFAULT_BALANCE,
    DEFAULT_BINS,
    DEFAULT_EXCLUDE,
    DEFAULT_SPACING,
    EQUALIZE_DURATION_COLUMN,
    MIN_DURATION_S,
    EqualizationBins,
    TemplateCycler,
    TemplateLibrary,
    TemplateSample,
    load_library,
)
from gridtools.templates.report import write_report
from gridtools.templates.splits import (
    SPLIT_NAMES,
    TemplateSplits,
    assign_template_splits,
    fish_units,
    load_splits,
    splits_report,
    write_splits,
)

__all__ = [
    "BRIEF_CYCLES_MAX",
    "DEFAULT_ALPHA",
    "DEFAULT_BALANCE",
    "DEFAULT_BINS",
    "DEFAULT_DIP_RANGE",
    "DEFAULT_EXCLUDE",
    "DEFAULT_SPACING",
    "EQUALIZE_DURATION_COLUMN",
    "MIN_DURATION_S",
    "SPLIT_NAMES",
    "BuildConfig",
    "EqualizationBins",
    "TemplateCycler",
    "TemplateLibrary",
    "TemplateSample",
    "TemplateSplits",
    "amplitude_from_frequency",
    "assign_template_splits",
    "build_library",
    "fish_units",
    "load_library",
    "load_splits",
    "random_dips",
    "splits_report",
    "write_report",
    "write_splits",
]
