"""
training_aois.py — polygons that exist only to train the SAR-to-NDVI estimator.

Six farms of 0.6 to 1.4 hectares is far too small a sample to fit a regression
on. These are larger cropland blocks in the same three districts, chosen so the
radar-to-NDVI relation they produce is representative of the fields the model
will actually be applied to.

They are NOT farms. They never appear in the portfolio, never get a confidence
badge, and never reach a client. They are training data and nothing else.

Why this is legitimate rather than a shortcut: the model being fitted maps radar
backscatter to NDVI for a crop type in a region. It does not need to be fitted on
the exact plots it is applied to — it needs to be fitted on the same crop, in the
same agro-climatic zone, under the same sensor geometry. Validating it on the
actual farms is a separate step, and that is what the held-out RMSE is for.

Each block is roughly 1 km across, so about 10,000 pixels at 10 m rather than 81.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Tuple


@dataclass
class TrainingAoi:
    aoi_id: str
    label: str
    district: str
    crop: str
    lat: float
    lon: float
    side_m: int = 1000
    polygon: List[Tuple[float, float]] = field(default_factory=list)


def _square(lat: float, lon: float, side_m: int) -> List[Tuple[float, float]]:
    dlat = (side_m / 2) / 111_320
    dlon = (side_m / 2) / (111_320 * math.cos(math.radians(lat)))
    return [
        (lon - dlon, lat - dlat),
        (lon + dlon, lat - dlat),
        (lon + dlon, lat + dlat),
        (lon - dlon, lat + dlat),
        (lon - dlon, lat - dlat),
    ]


# Maize is dominant in Karimnagar and Jagtial; cotton in Warangal. Blocks are
# placed on cropland away from settlements and water bodies.
TRAINING_AOIS: List[TrainingAoi] = [
    TrainingAoi("TR-MZ-01", "Choppadandi block A", "Karimnagar", "maize", 18.5180, 79.0840),
    TrainingAoi("TR-MZ-02", "Choppadandi block B", "Karimnagar", "maize", 18.5310, 79.1020),
    TrainingAoi("TR-MZ-03", "Korutla block A", "Jagtial", "maize", 18.8260, 78.7050),
    TrainingAoi("TR-MZ-04", "Korutla block B", "Jagtial", "maize", 18.8390, 78.7240),
    TrainingAoi("TR-MZ-05", "Metpally block", "Jagtial", "maize", 18.8480, 78.6320),
    TrainingAoi("TR-CT-01", "Parkal block A", "Warangal", "cotton", 18.2090, 79.6790),
    TrainingAoi("TR-CT-02", "Parkal block B", "Warangal", "cotton", 18.1930, 79.7010),
    TrainingAoi("TR-CT-03", "Atmakur block", "Warangal", "cotton", 18.1520, 79.6210),
    TrainingAoi("TR-CT-04", "Shayampet block", "Warangal", "cotton", 18.1210, 79.5480),
    TrainingAoi("TR-CT-05", "Duggondi block", "Warangal", "cotton", 17.9540, 79.7860),
]

for _a in TRAINING_AOIS:
    _a.polygon = _square(_a.lat, _a.lon, _a.side_m)
