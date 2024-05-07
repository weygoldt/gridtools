"""Data classes to load electrode grid datasets."""

import pathlib
from typing import Dict, Optional, Self, Type, TypeVar, Union

import numpy as np
import numpy.typing as npt
from pydantic import (
    BaseModel,
    ConfigDict,
    ValidationError,
    field_validator,
    model_validator,
)
from rich.pretty import pprint as rpprint
from thunderlab.dataloader import DataLoader

from gridtools.utils.exceptions import GridDataMismatchError

# The order determines the priority of which the data of the detectors is used.
chirp_detectors = ["gt", "rcnn", "cnn", "None"]
rise_detectors = ["gt", "rcnn", "pd", "None"]

# Define a few types for type hinting.
GridType = TypeVar("GridType", npt.NDArray, DataLoader)


def _pprint(obj: BaseModel) -> None:
    """Recursively pretty-print the attributes of an object."""

    def collect_vars(obj: BaseModel) -> Optional[Dict[str, str]]:
        """Collect all variables of a BaseModel object."""
        if isinstance(obj, BaseModel):
            result = {}
            for key, value in obj.__dict__.items():
                if not key.startswith("_") and value is not None:
                    if isinstance(value, BaseModel):
                        result[key] = collect_vars(value)
                    else:
                        result[key] = str(type(value).__name__)
            return result
        return None

    return rpprint(collect_vars(obj), expand_all=True)


class WavetrackerData(BaseModel):
    """Contains data extracted by the wavetracker.

    All check functions are automatically run when the object is instantiated.

    Parameters
    ----------
    - `freqs` : `numpy.ndarray[float]`
        Array of frequencies.
    - `powers` : `numpy.ndarray[float]`
        Array of powers.
    - `idents` : `numpy.ndarray[float]`
        Array of idents.
    - `indices` : `numpy.ndarray[int]`
        Array of indices.
    - `ids` : `numpy.ndarray[int]`
        Array of ids.
    - `times` : `numpy.ndarray[float]`
        Array of times.
    - `xpos` : `numpy.ndarray[float]]`
        Array of x positions
    - `ypos` : `numpy.ndarray[float]`
        Array of y positions
    - `has_positions` : `bool`
        Whether or not the wavetracker data contains position estimates.
        If not, the xpos and ypos arrays are empty.

    Methods
    -------
    - `pprint()`
        Pretty-print the attributes of the object.

    Raises
    ------
    - `ValidationError`
        If any of the arrays is not a numpy array.
    - `GridDataMismatch`
        If the times are not monotonically increasing.
        If the number of times is smaller than the number of unique indices.
        If the wavetracker data is not of equal length.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True, frozen=False)
    freqs: npt.NDArray[np.float_]
    powers: npt.NDArray[np.float_]
    idents: npt.NDArray[np.float_]
    indices: npt.NDArray[np.int_]
    ids: npt.NDArray[np.float_]
    times: npt.NDArray[np.float_]
    xpos: npt.NDArray[np.float_]
    ypos: npt.NDArray[np.float_]
    has_positions: bool

    @field_validator(
        "freqs", "powers", "idents", "indices", "ids", "times", "xpos", "ypos"
    )
    @classmethod
    def _check_numpy_array(
        cls: type["WavetrackerData"], v: np.ndarray
    ) -> np.ndarray:
        """
        Check if all the arrays are numpy arrays.

        Parameters
        ----------
        - `v` : `np.ndarray`
            Array to be checked.

        Returns
        -------
        - `np.ndarray`
            The input array.

        Raises
        ------
        - `ValidationError`
            If the input is not a numpy array.
        """
        if not isinstance(v, np.ndarray):
            msg = "Value must be a numpy.ndarray"
            raise ValidationError(msg)
        return v

    @field_validator("times")
    @classmethod
    def _check_times_sorted(
        cls: type["WavetrackerData"], v: np.ndarray
    ) -> np.ndarray:
        """Check that the times are monotonically increasing.

        Parameters
        ----------
        - `v :` np`.ndarray`
            Array of times.

        Returns
        -------
        - `np`.ndarray
            `The input array.`

        Raises
        ------
        - `GridDataMismatch`
            If the times are not monotonically increasing.
        """
        if not np.all(np.diff(v) > 0):
            msg = "Wavetracker times are not monotonically increasing!"
            raise GridDataMismatchError(msg)
        return v

    @model_validator(mode="after")
    def _check_times_indices(self: Self) -> Self:
        """Check that index and time array match.

        Checks that the indices in the indices array cannot go out of bounds
        of the times array.

        Returns
        -------
        - `WavetrackerData`
            The current instance of the WavetrackerData class.

        Raises
        ------
        - `GridDataMismatch`
            If the number of times is smaller than the number of unique
            indices.
        """
        if self.times.shape[0] < len(set(self.indices)):
            msg = "Number of times is smaller than number of unique indices!"
            raise GridDataMismatchError(msg)
        return self

    @model_validator(mode="after")
    def _check_wavetracker_data(self: Self) -> Self:
        """Check if the wavetracker data is of correct length.

        Returns
        -------
        - `WavetrackerData`
            The current instance of the WavetrackerData class.

        Raises
        ------
        - `GridDataMismatch`
            If the wavetracker data is not of equal length.
        """
        lengths = [
            np.shape(x)[0]
            for x in [self.freqs, self.powers, self.idents, self.indices]
        ]
        if len(set(lengths)) > 1:
            msg = "Wavetracker data is not of equal length!"
            raise GridDataMismatchError(msg)
        return self

    def pprint(self: Self) -> None:
        """Pretty-print the attributes of the object."""
        _pprint(self)


class GridData(BaseModel):
    """Contains raw data from the electrode grid recordings.

    Parameters
    ----------
    - `raw` : `Union[np.ndarray, DataLoader]`
        The raw data from the electrode grid recordings.
    - `samplerate` : `float`
        The sampling rate of the raw data.
    - `shape`: tuple`[int, int]`
        The shape of the raw data.

    Methods
    -------
    - `pprint()`
        Pretty-print the attributes of the object.

    Raises
    ------
    - `ValidationError`
        If the raw data is not a numpy array or a DataLoader object.

    Examples
    --------
    ```python
    import pathlib
    from gridtools.datasets load_grid
    rec = load_grid(pathlib.Path("path/to/raw"))
    rec.rec
    rec.samplerate
    ```
    """

    model_config = ConfigDict(arbitrary_types_allowed=True, frozen=False)
    rec: Union[npt.NDArray, DataLoader]
    samplerate: float
    shape: tuple[int, int]

    @field_validator("rec")
    @classmethod
    def _check_raw(
        cls: type[Self], v: Union[npt.NDArray, DataLoader]
    ) -> Union[npt.NDArray, DataLoader]:
        """
        Check if the raw data is a numpy array or a DataLoader object.

        Parameters
        ----------
        - `v` : `Union`[np.ndarray, DataLoader]`
            The raw data to be validated.

        Returns
        -------
        - `Union[np.ndarray, DataLoader]`
            The validated raw data.

        Raises
        ------
        - `ValidationError`
            If the raw data is not a numpy array or a DataLoader object.
        """
        n_dimensions = 2  # Grid data must have two dimensions.
        if not isinstance(v, (np.ndarray, DataLoader)):
            msg = "Raw data must be a numpy array or a DataLoader."
            raise ValidationError(msg)
        if not isinstance(v.shape, tuple):
            msg = "Raw data must have a shape."
            raise ValidationError(msg)
        if len(v.shape) != n_dimensions:
            msg = "Raw data must have two dimensions."
            raise ValidationError(msg)
        if not isinstance(v.shape[0], int):
            msg = "Raw data must have at least one dimension."
            raise ValidationError(msg)
        if not isinstance(v.shape[1], int):
            msg = "Raw data must have at least one dimension."
            raise ValidationError(msg)
        return v

    def pprint(self: Self) -> None:
        """Pretty-print the attributes of the object."""
        _pprint(self)


class ChirpData(BaseModel):
    """Contains data about chirps produced by fish in the dataset.

    Parameters
    ----------
    - `times` : `np.ndarray`
        A numpy array containing the times at which chirps were detected.
    - `idents` : `np.ndarray`
        A numpy array containing the identities of the fish that produced the
        chirps.
    - `params` : `np.ndarray`
        A numpy array containing the parameters of the chirps.
        Is empty if no parameters are provided.
    - `are_detected` : `bool`
        `Whether or not the chirps are detected.
        If not, all arrays are empty.
    - `have_params` : `bool`
        Whether or not the chirps have parameters.
        If not, the params array is empty.
    - `detector` : `str`
        The type of detector used to detect the chirps. Must be either
        'gt' or 'cnn'. More detector types will probably follow.

    Methods
    -------
    - `pprint()`
        Pretty-print the attributes of the object.
    - `check_numpy_array(v)`
        Check if times and idents are numpy arrays.
    - `check_detector(v)`
        Check if detector is either 'gt' or 'cnn' and a string.

    Raises
    ------
    - `ValidationError`
        If the input values do not meet the specified requirements.

    Examples
    --------
    ```python
    times = np.array([0.1, 0.2, 0.3])
    idents = np.array([1, 2, 3])
    detector = 'gt'
    chirps = ChirpData(times=times, idents=idents, detector=detector)
    ```
    """

    model_config = ConfigDict(arbitrary_types_allowed=True, frozen=False)
    times: npt.NDArray[np.float_]
    idents: npt.NDArray[np.int_]
    params: npt.NDArray[np.float_]
    are_detected: bool
    have_params: bool
    detector: str

    @field_validator("times", "idents", "params")
    @classmethod
    def _check_numpy_array(
        cls: Type["ChirpData"], v: npt.NDArray
    ) -> npt.NDArray:
        """
        Check if times and idents are numpy arrays.

        Parameters
        ----------
        v : npt.NDArray
            The value to be checked.

        Returns
        -------
        npt.NDArray
            The input value if it is a numpy array.

        Raises
        ------
        ValidationError
            If the input value is not a numpy array.
        """
        if not isinstance(v, np.ndarray):
            msg = "Value must be a numpy.ndarray"
            raise ValidationError(msg)
        return v

    @field_validator("detector")
    @classmethod
    def _check_detector(cls: Type["ChirpData"], v: str) -> str:
        """
        Check if detector is either 'gt' or 'cnn' and a string.

        Parameters
        ----------
        v : ChirpData
            The value to be checked.

        Returns
        -------
        str
            The input value if it is a string and either 'gt' or 'cnn'.

        Raises
        ------
        ValidationError
            If the input value is not a string or is not 'gt' or 'cnn'.
        """
        if not isinstance(v, str):
            msg = "Detector must be a string."
            raise ValidationError(msg)
        if v not in chirp_detectors:
            msg = f"Detector must be in {chirp_detectors}."
            raise ValidationError(msg)
        return v

    def pprint(self: "ChirpData") -> None:
        """Pretty-print the attributes of the object."""
        _pprint(self)


class ChirpDataV2(BaseModel):
    """Contains data about chirps produced by fish in the dataset.

    Parameters
    ----------
    - `recording` : `str`
        The name of the recording.
    - `spec_batches` : `np.ndarray`
        A numpy array containing the batch numbers of the spectrogram.
    - `spec_batch_raw_idxs` : `np.ndarray`
        A numpy array containing index to the data from the raw recording
        that was used to compute the spectrogram.
    - `spec_windows` : `np.ndarray`
        A numpy array containing the window (of n windows in one batch)
        of the spectrogram.
    - `spec_freq_ranges` : `np.ndarray`
        A numpy array containing the frequency range of the spectrogram.
    - `spec_batch_idxs` : `np.ndarray`
        A numpy array containing the batch index of the spectrogram.
    - `bbox_ids` : `np.ndarray`
        A numpy array containing the unique identifier of the bounding box.
        This is mostly only used during detection.
    - `bboxes_xy`: `np.ndarray`
        A numpy array containing the bounding box coordinates in the
        x1y1x2y2 format.
    - `bboxes_ft` : `np.ndarray`
        A numpy array containing the bounding box coordinates in the
        frequency-time domain.
    - `spec_powers` : `np.ndarray`
        A numpy array containing the interpolated power of the spectrogram.
    - `spec_freqs` : `np.ndarray`
        A numpy array containing the frequency bins of the spectrogram.
    - `spec_times` : `np.ndarray`
        A numpy array containing the time bins of the spectrogram.
    - `detection_scores` : `np.ndarray`
        A numpy array containing the detection scores of the chirps.
    - `pred_eodfs` : `np.ndarray`
        A numpy array containing the predicted EODFs of the chirp emitters.
    - `pred_ids` : `np.ndarray`
        A numpy array containing the predicted identifiers of the chirp emitters.
    - `are_detected` : `bool`
        Whether or not the chirps are detected.
        If not, all arrays are empty.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True, frozen=False)
    recording: str
    spec_batches: npt.NDArray[np.int_]
    spec_batch_raw_idxs: npt.NDArray[np.int_]
    spec_windows: npt.NDArray[np.int_]
    spec_freq_ranges: npt.NDArray[np.int_]
    spec_batch_idxs: npt.NDArray[np.int_]
    bbox_ids: npt.NDArray[np.int_]
    bboxes_x1y1x2y2: npt.NDArray[np.float_]
    bboxes_f1t1f2t2: npt.NDArray[np.float_]
    spec_powers: npt.NDArray[np.float_]
    spec_freqs: npt.NDArray[np.float_]
    spec_times: npt.NDArray[np.float_]
    detection_scores: npt.NDArray[np.float_]
    pred_eodfs: npt.NDArray[np.float_]
    pred_ids: npt.NDArray[np.int_]
    are_detected: bool
    detector: str


class RiseData(BaseModel):
    """Contains data of rises produced by fish in the dataset.

    Parameters
    ----------
    - `times` : `numpy.ndarray`
        An array of times.
    - `idents` : `numpy.ndarray`
        An array of identifiers.
    - `params` : `numpy.ndarray`
        An array of parameters.
    - `are_detected` : `bool`
        Whether or not the rises are detected.
        If not, all arrays are empty.
    - `have_params` : `bool`
        Whether or not the rises have parameters.
        If not, the params array is empty.
    - `detector` : `str`
        The detector used to produce the data. Must be either 'gt' (ground
        truth) or 'pd' (peak detection).

    Methods
    -------
    - `pprint()`
        Pretty-print the attributes of the object.

    Examples
    --------
    ```python
    times = np.array([0.1, 0.2, 0.3])
    idents = np.array([1, 2, 3])
    detector = 'gt'
    data = RiseData(times=times, idents=idents, detector=detector)
    ```
    """

    model_config = ConfigDict(arbitrary_types_allowed=True, frozen=False)
    times: npt.NDArray[np.float_]
    idents: npt.NDArray[np.int_]
    params: npt.NDArray[np.float_]
    are_detected: bool
    have_params: bool
    detector: str

    @field_validator("times", "idents", "params")
    @classmethod
    def _check_numpy_array(cls: type[Self], v: npt.NDArray) -> npt.NDArray:
        """Check if times and idents are numpy arrays.

        Parameters
        ----------
        - `v` : `numpy`.ndarray`
            The value to be validated.

        Returns
        -------
        - `numpy.ndarray`
            The validated value.

        Raises
        ------
        - `ValidationError`
            If the value is not a numpy array.
        """
        if not isinstance(v, np.ndarray):
            msg = "Value must be a numpy.ndarray"
            raise ValidationError(msg)
        return v

    @field_validator("detector")
    @classmethod
    def _check_detector(cls: Type[Self], v: str) -> str:
        """Check if detector is a string and valid.

        Check if detector is one of the detectors
        specified in the global detector variable.

        Parameters
        ----------
        - `v` : `str`
            The value to be validated.

        Returns
        -------
        - `str`
            The validated value.

        Raises
        ------
        - `ValidationError`
            If the value is not a string or is not 'gt' or 'pd'.
        """
        if not isinstance(v, str):
            msg = "Detector must be a string."
            raise ValidationError(msg)
        if v not in rise_detectors:
            msg = f"Detector must be in {rise_detectors}."
            raise ValidationError(msg)
        return v

    def pprint(self: "RiseData") -> None:
        """Pretty-print the attributes of the object."""
        _pprint(self)


class CommunicationData(BaseModel):
    """Contains data for communication signals.

    Parameters
    ----------
    - `chirp` : `ChirpData, optional`
        Data for the chirp signal produced by the fish.
    - `rise` : `RiseData, optional`
        Data for the rise signal produced by the fish.
    - `are_detected` : `bool`
        Whether or not communication signals are detected.
        If not, all arrays are empty.

    Methods
    -------
    - `pprint()`
        Pretty-print the attributes of the object.

    Raises
    ------
    - `ValidationError`
        If chirp data is not a ChirpData object or if rise data is not a
        RiseData object, or if neither chirp nor rise data is provided.

    Returns
    -------
    - `CommunicationData`
        An instance of the CommunicationData class.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)
    chirp: ChirpData
    rise: RiseData
    are_detected: bool

    @field_validator("chirp")
    @classmethod
    def _typecheck_chirp(
        cls: Type[Self], v: Type[ChirpData]
    ) -> Type[ChirpData]:
        """
        Check if chirp data is a ChirpData object if it is not none.

        Parameters
        ----------
        v : Any
            The value to check.

        Raises
        ------
        ValidationError
            If chirp data is not a ChirpData object.

        Returns
        -------
        Any
            The value that was passed in.
        """
        if v is not None and not isinstance(v, ChirpData):
            msg = "Chirp data must be a ChirpData object."
            raise ValidationError(msg)
        return v

    @field_validator("rise")
    @classmethod
    def _typecheck_rise(cls: type[Self], v: type[RiseData]) -> type[RiseData]:
        """
        Check if rise data is a RiseData object if it is not none.

        Parameters
        ----------
        v : Any
            The value to check.

        Raises
        ------
        ValidationError
            If rise data is not a RiseData object.

        Returns
        -------
        Any
            The value that was passed in.
        """
        if v is not None and not isinstance(v, RiseData):
            msg = "Rise data must be a RiseData object."
            raise ValidationError(msg)
        return v

    def pprint(self: Self) -> None:
        """Pretty-print the attributes of the object."""
        _pprint(self)


class Dataset(BaseModel):
    """The main dataset class to load data extracted from electrode grids.

    This is the main interface class to interact with grid data.
    Every grid dataset must at least contain the raw recording
    as well as the data extracted by Till Raabs [wavetracker](
    https://github.com/tillraab/wavetracker) software.

    Parameters
    ----------
    - `path` : `pathlib.Path`
        The path to the wavetracker dataset.
    - `grid` : `Optional[GridData], optional`
        The raw data, by default None.
    - `track` : `WavetrackerData`
        The wavetracker data.
    - `com` : `Optional[CommunicationData], optional`
        The communication data, by default None.

    Methods
    -------
    - `pprint()`
        Pretty-print the attributes of the object.

    Examples
    --------
    ```python
    import pathlib
    from gridtools.datasets import load
    ds = load(pathlib.Path("path/to/dataset"))
    # look at the chirps
    ds.com.chirp.times
    # or do something with the wavetracker data
    ds.track.freqs
    ```
    """

    # TODO: Update the integrity checks to not pass with None values.

    model_config = ConfigDict(arbitrary_types_allowed=True, frozen=False)
    path: pathlib.Path
    grid: GridData
    track: WavetrackerData
    com: CommunicationData

    @field_validator("path")
    @classmethod
    def _check_path(cls: type[Self], v: pathlib.Path) -> pathlib.Path:
        """Check if path is a pathlib.Path object.

        Parameters
        ----------
        v : Any
            The value to be validated.

        Returns
        -------
        pathlib.Path
            The validated path.

        Raises
        ------
        ValidationError
            If the value is not a pathlib.Path object.
        """
        if not isinstance(v, pathlib.Path):
            msg = "Path must be a pathlib.Path object."
            raise ValidationError(msg)
        return v

    @field_validator("grid")
    @classmethod
    def _check_rec(cls: type[Self], v: GridData) -> GridData:
        """Check if raw data is a GridData object or none.

        Parameters
        ----------
        v : Any
            The value to be validated.

        Returns
        -------
        GridData or None
            The validated raw data.

        Raises
        ------
        ValidationError
            If the value is not a GridData object.
        """
        if v is not None and not isinstance(v, GridData):
            msg = "Raw data must be a GridData object."
            raise ValidationError(msg)
        return v

    @field_validator("track")
    @classmethod
    def _check_track(cls: type[Self], v: WavetrackerData) -> WavetrackerData:
        """Check if wavetracker data is a WavetrackerData object.

        Parameters
        ----------
        v : Any
            The value to be validated.

        Returns
        -------
        WavetrackerData
            The validated wavetracker data.

        Raises
        ------
        ValidationError
            If the value is not a WavetrackerData object.
        """
        if v is not None and not isinstance(v, WavetrackerData):
            msg = "Wavetracker data must be a WavetrackerData object."
            raise ValidationError(msg)
        return v

    @field_validator("com")
    @classmethod
    def _check_com(cls: type[Self], v: CommunicationData) -> CommunicationData:
        """Check if communication data is a CommunicationData object or none.

        Parameters
        ----------
        v : Any
            The value to be validated.

        Returns
        -------
        CommunicationData or None
            The validated communication data.

        Raises
        ------
        ValidationError
            If the value is not a CommunicationData object.
        """
        if v is not None and not isinstance(v, CommunicationData):
            msg = "Communication data must be a CommunicationData object."
            raise ValidationError(msg)
        return v

    def pprint(self: Self) -> None:
        """Pretty-print the attributes of the object."""
        _pprint(self)
