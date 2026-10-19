"""Reusable YAML setups and distance sweeps for external analysis scripts."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from collections.abc import Callable, Iterable
import math

from . import api
from .scenario_io import load_scenario_yaml, save_scenario_yaml, scenario_from_dict, scenario_to_dict
from .types import Scenario, ScenarioResult

__all__ = ['load_setup', 'save_setup', 'with_overrides', 'run_setup',
           'sweep_distance', 'DistanceSample', 'DistanceSweep']


def with_overrides(setup: Scenario, **overrides) -> Scenario:
    """Return an independently validated copy; null clears an optional value."""
    return scenario_from_dict({**scenario_to_dict(setup), **overrides})


def load_setup(path: str | Path, **overrides) -> Scenario:
    """Load existing CLI-compatible YAML and replace only named fields."""
    return with_overrides(load_scenario_yaml(path), **overrides)


def save_setup(setup: Scenario, path: str | Path) -> None:
    """Save all resolved inputs, including defaults, enums and local date/time."""
    save_scenario_yaml(with_overrides(setup), path)


def run_setup(path: str | Path, *, overrides: dict | None = None,
              include_radiance: bool = True, include_turbulence: bool = True) -> ScenarioResult:
    """Run a saved setup with explicit input overrides and calculation options."""
    return api.run_scenario(load_setup(path, **(overrides or {})),
                            include_radiance=include_radiance,
                            include_turbulence=include_turbulence)


@dataclass(frozen=True)
class DistanceSample:
    distance_km: float
    score: float


@dataclass(frozen=True)
class DistanceSweep:
    samples: tuple[DistanceSample, ...]
    threshold: float
    crossed_threshold: bool

    @property
    def last_passing_distance_km(self) -> float | None:
        passing = [s.distance_km for s in self.samples if s.score >= self.threshold]
        return passing[-1] if passing else None

    @property
    def first_failing_distance_km(self) -> float | None:
        return self.samples[-1].distance_km if self.crossed_threshold else None


def sweep_distance(
    setup: Scenario | str | Path,
    distances_km: Iterable[float],
    *,
    metric: Callable[[ScenarioResult], float],
    threshold: float,
    overrides: dict | None = None,
    include_radiance: bool = True,
    include_turbulence: bool = False,
    on_sample: Callable[[DistanceSample], None] | None = None,
) -> DistanceSweep:
    """Increase slant range until ``metric(result) < threshold`` or samples end.

    The metric can be an external detectability/SNR calculation. No detector
    physics is assumed. Equality passes. Results bracket the first sampled
    crossing, not an exact or global maximum range. Full spectra are passed
    to the callback but not retained, bounding memory across iterations.

    Source altitude is cleared: distance positions the target along the saved
    sensor direction. Other saved inputs are preserved unless overridden.
    Distances must be finite, positive and strictly increasing. A requested
    range beyond ground/top-of-atmosphere raises before that spectrum runs,
    rather than silently reporting a clipped path as the requested distance.
    """
    if not math.isfinite(threshold):
        raise ValueError('threshold must be finite')
    distances = tuple(float(d) for d in distances_km)
    if not distances or any(not math.isfinite(d) or d <= 0 for d in distances):
        raise ValueError('distances must be finite, positive and nonempty')
    if any(b <= a for a, b in zip(distances, distances[1:])):
        raise ValueError('distances must be strictly increasing')
    base = load_setup(setup) if isinstance(setup, (str, Path)) else setup
    base = with_overrides(base, **(overrides or {}))
    base = with_overrides(base, source_alt_km=None, source_range_km=None)
    profile = api.build_profile(base)
    samples = []
    for distance in distances:
        scenario = with_overrides(base, source_range_km=distance)
        path = api.build_scenario_path(scenario, profile)
        if not math.isclose(path.total_length_km, distance, rel_tol=1e-7, abs_tol=1e-7):
            raise ValueError(f'Requested {distance:g} km exceeds the available path '
                             f'({path.total_length_km:g} km); reduce the sweep range or change the sensor direction')
        result = api.run_scenario(scenario, include_radiance=include_radiance,
                                  include_turbulence=include_turbulence)
        score = float(metric(result))
        if not math.isfinite(score):
            raise ValueError(f'metric returned a non-finite value at {distance:g} km')
        sample = DistanceSample(distance, score)
        samples.append(sample)
        if on_sample is not None:
            on_sample(sample)
        if score < threshold:
            return DistanceSweep(tuple(samples), threshold, True)
    return DistanceSweep(tuple(samples), threshold, False)
