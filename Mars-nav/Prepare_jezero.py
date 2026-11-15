import rasterio
import numpy as np
import plotly.graph_objects as go
from pyproj import Transformer

# Load real USGS Jezero elevation data
dem_path = "data/M20_JezeroCrater_CTXDEM_20m.tif"

with rasterio.open(dem_path) as src:
    elevation = src.read(1, masked=True)

    # Reduce resolution for smooth 3D rendering
    elevation = elevation[::20, ::20]
# Calculate terrain slope from real elevation data
pixel_width = abs(src.transform.a) * 10
pixel_height = abs(src.transform.e) * 10

z = elevation.filled(np.nan).astype(float)

valid = np.isfinite(z)
z_filled = np.where(valid, z, np.nanmedian(z))

dz_dy, dz_dx = np.gradient(
    z_filled,
    pixel_height,
    pixel_width
)

slope_degrees = np.degrees(
    np.arctan(np.sqrt(dz_dx**2 + dz_dy**2))
)

slope_degrees[~valid] = np.nan

print("SUCCESS: Terrain slope calculated!")
print("Maximum slope:", np.nanmax(slope_degrees), "degrees")
    # Get actual pixel coordinates
transform = src.transform
rows, cols = elevation.shape

x = transform.c + np.arange(cols) * transform.a * 20
y = transform.f + np.arange(rows) * transform.e * 20

X, Y = np.meshgrid(x, y)
# Mars 2000 spherical equirectangular coordinates
mars_radius = 3396190
standard_lat = np.radians(18.4663)

transformer = Transformer.from_crs(
    src.crs.geodetic_crs,
    src.crs,
    always_xy=True
)

def mars_xy(lon, lat):
    px, py = transformer.transform(lon, lat)
    return px, py

# Actual Perseverance landing site
landing_x, landing_y = mars_xy(77.4508, 18.4447)

# Illustrative MARS-NAV proposed base location
base_x, base_y = mars_xy(77.65, 18.30)
# Find nearest DEM grid cells for both markers
landing_col = np.abs(X[0, :] - landing_x).argmin()
landing_row = np.abs(Y[:, 0] - landing_y).argmin()

base_col = np.abs(X[0, :] - base_x).argmin()
base_row = np.abs(Y[:, 0] - base_y).argmin()

# Read marker elevations from the DEM
landing_z = float(z_filled[landing_row, landing_col])
base_z = float(z_filled[base_row, base_col])

# Create 3D terrain

# Create 3D terrain
fig = go.Figure(
    data=[
        go.Surface(
            x=X.astype(np.float32),
            y=Y.astype(np.float32),
            z=elevation.filled(np.nan).astype(np.float32),
            colorscale="Turbo",
            colorbar=dict(title="Elevation (m)"),
            connectgaps=False,
            hoverinfo="skip",
            showscale=True
        ),

        go.Scatter3d(
            x=[landing_x],
            y=[landing_y],
            z=[landing_z + 250],
            mode="markers+text",
            marker=dict(size=7, color="white", symbol="diamond"),
            text=["PERSEVERANCE"],
            textposition="top left",
            textfont=dict(size=12, color="white"),
            name="Perseverance Landing Site"
        ),

        go.Scatter3d(
            x=[base_x],
            y=[base_y],
            z=[base_z + 400],
            mode="markers+text",
            marker=dict(size=9, color="red", symbol="circle"),
            text=["MARS-NAV BASE"],
            textposition="bottom center",
            textfont=dict(size=16, color="white"),
            name="Proposed Base"
        )
    ]
)


fig.update_layout(
    title="MARS-NAV | Jezero Crater - 3D Elevation Model",
    template="plotly_dark",
    scene=dict(
        dragmode="orbit",
        xaxis_title="East-West (m)",
        yaxis_title="North-South (m)",
        zaxis_title="Elevation (m)",
        aspectmode="manual",
        aspectratio=dict(x=1.3, y=1, z=0.25),
        camera=dict(
            eye=dict(x=1.6, y=-1.7, z=1.25)
        )
    ),
    height=750,
    width=1100,
    margin=dict(l=10, r=10, t=65, b=10)
)

fig.write_html(
    "jezero_3d.html",
    include_plotlyjs=True,
    full_html=True,
    config={
        "responsive": True,
        "scrollZoom": False,
        "displayModeBar": False
    }
)
 
print("SUCCESS: Real Jezero 3D terrain created!")
print("File: jezero_3d.html")