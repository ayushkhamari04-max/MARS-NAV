
import rasterio
import glob
import numpy as np
import plotly.graph_objects as go
from rasterio.enums import Resampling

# Find the USGS Jezero elevation file
files = glob.glob("data/M20_JezeroCrater_CTXDEM_20m.tif")

if not files:
    raise FileNotFoundError("USGS Jezero DEM not found in data folder")

# Open the original elevation data
with rasterio.open(files[0]) as src:

    # Create a smaller preview without changing the original file
    terrain = src.read(
        1,
        out_shape=(350, 500),
        masked=True,
        resampling=Resampling.average
    )

# Preserve missing-data areas
elevation = terrain.filled(np.nan)

# Build the 3D terrain

fig = go.Figure(data=[
    go.Surface(
        z=elevation,
        colorscale="Turbo",
        colorbar=dict(title="Elevation (m)"),
        connectgaps=False,
        hovertemplate=(
            "West-East: %{x}<br>"
            "South-North: %{y}<br>"
            "Elevation: %{z:.2f} m"
            "<extra></extra>"
        )
    )
])

fig.update_layout(
    title="MARS-NAV | HiRISE 3D Terrain Preview",
    scene={
        "xaxis_title": "West-East",
        "yaxis_title": "South-North",
        "zaxis_title": "Elevation (m)",
        "aspectratio": {"x": 1.4, "y": 1, "z": 0.08},
        "camera": {
            "eye": {"x": 1.7, "y": -1.8, "z": 1.4}
        }
    },
    height=900
)

fig.write_html("terrain_preview.html", auto_open=True)
print("3D terrain preview created successfully!")
