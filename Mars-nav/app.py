
import math
import heapq
from pathlib import Path
import streamlit as st
import streamlit.components.v1 as components
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import folium
import rasterio
from rasterio.io import MemoryFile
from rasterio.enums import Resampling
from affine import Affine
from streamlit_folium import st_folium


# ============================================================
# MARS-NAV | MARTIAN EXPLORATION AND NAVIGATION SYSTEM
# DEVELOPMENT VERSION 3.0
# ============================================================

st.set_page_config(
    page_title="MARS-NAV | Martian Mission Control",
    page_icon="🔴",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# DESIGN AND THEME
# ============================================================

st.markdown("""
<style>

.stApp {
    background: linear-gradient(135deg, #100c0b, #21100e);
    color: #f5e6d6;
}

h1, h2, h3 {
    color: #ff9b62 !important;
}

[data-testid="stSidebar"] {
    background: #100d0c;
    border-right: 1px solid #9b4932;
}

div[data-testid="stMetric"] {
    background: #271511;
    border: 1px solid #9b4932;
    padding: 14px;
    border-radius: 12px;
}

.stButton button {
    background: #a83d28;
    color: white;
    border-radius: 8px;
    border: 1px solid #ff9b62;
}

</style>
""", unsafe_allow_html=True)


# ============================================================
# CONSTANTS
# ============================================================

MARS_RADIUS_KM = 3389.5

JEZERO_LAT = 18.4000
JEZERO_LON = 77.5000

PERSEVERANCE_LANDING_LAT = 18.4446
PERSEVERANCE_LANDING_LON = 77.4509


# REAL JEZERO CRATER EXPLORATION LANDMARKS
LANDMARKS = {
    "Jezero Crater Centre": (18.4000, 77.5000),

    "Octavia E. Butler Landing": (18.4446, 77.4509),

    "Seítah Formation": (18.4339, 77.4430),

    "Three Forks Sample Depot": (18.4528, 77.4082),

    "Western Fan Delta Front": (18.4589, 77.4061),
}


DATA_DIR = Path(__file__).parent / "data"

APP_DIR = Path(__file__).parent


# ============================================================
# SESSION STATE
# ============================================================

if "mission_sol" not in st.session_state:
    st.session_state.mission_sol = 1

if "rover_speed" not in st.session_state:
    st.session_state.rover_speed = 1.0

if "show_jezero_3d" not in st.session_state:
    st.session_state.show_jezero_3d = False

if "last_rover_route" not in st.session_state:
    st.session_state.last_rover_route = None


# ============================================================
# MARS DISTANCE CALCULATION
# ============================================================

def mars_distance(lat1, lon1, lat2, lon2):

    lat1, lon1, lat2, lon2 = map(
        np.radians,
        [lat1, lon1, lat2, lon2]
    )

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = (
        np.sin(dlat / 2) ** 2
        +
        np.cos(lat1) * np.cos(lat2)
        *
        np.sin(dlon / 2) ** 2
    )

    return float(
        2 * MARS_RADIUS_KM *
        np.arcsin(np.sqrt(a))
    )


# ============================================================
# MARS SURFACE MAP
# ============================================================

def create_map():

    # CREATE MARS MAP
    mars_map = folium.Map(
        location=[JEZERO_LAT, JEZERO_LON],
        zoom_start=10,
        tiles=None,
        crs="EPSG4326",
        control_scale=True,
        max_zoom=20
    )

    # BASE LAYER: MARS MOLA + HRSC TERRAIN
    folium.TileLayer(
        tiles="https://astro.arcgis.com/arcgis/rest/services/OnMars/MColorDEM/MapServer/tile/{z}/{y}/{x}",
        attr="USGS Astrogeology / Esri",
        name="Mars Terrain | MOLA + HRSC",
        overlay=False,
        control=True,
        max_native_zoom=7,
        max_zoom=20,
        crs="EPSG4326"
    ).add_to(mars_map)

    # OVERLAY: NASA CTX SATELLITE IMAGERY
    folium.TileLayer(
        tiles="https://astro.arcgis.com/arcgis/rest/services/OnMars/CTX/MapServer/tile/{z}/{y}/{x}",
        attr="USGS Astrogeology / Caltech / Esri",
        name="NASA CTX High Resolution",
        overlay=True,
        control=True,
        opacity=1.0,
        max_native_zoom=12,
        max_zoom=20,
        crs="EPSG4326"
    ).add_to(mars_map)

    # JEZERO CRATER MISSION CENTRE
    folium.Marker(
        location=[JEZERO_LAT, JEZERO_LON],
        tooltip="Jezero Crater Mission Centre",
        popup="MARS-NAV | Jezero Crater",
        icon=folium.Icon(
            color="red",
            icon="info-sign"
        )
    ).add_to(mars_map)

# MISSION LANDMARKS
    for index, (name, coordinates) in enumerate(LANDMARKS.items(), start=1):

        print(f"ADDING LANDMARK {index}: {name} - {coordinates}")

        folium.Marker(
            location=coordinates,
            tooltip=name,
            popup=folium.Popup(
                f"""
                <b>{name}</b><br>
                Latitude: {coordinates[0]:.4f}° N<br>
                Longitude: {coordinates[1]:.4f}° E
                """,
                max_width=300
            ),
            icon=folium.DivIcon(
                html=f"""
                <div style="
                    background:#ffb347;
                    color:#171717;
                    border:2px solid white;
                    border-radius:50%;
                    width:28px;
                    height:28px;
                    text-align:center;
                    line-height:24px;
                    font-weight:bold;
                    font-size:13px;">
                    {index}
                </div>
                """,
                icon_size=(28, 28),
                icon_anchor=(14, 14)
            )
        ).add_to(mars_map)
    print("LANDMARK COUNT:", len(LANDMARKS), LANDMARKS)

    # AUTOMATICALLY FRAME ALL MISSION LANDMARKS
    landmark_bounds = list(LANDMARKS.values()) + [(JEZERO_LAT, JEZERO_LON)]
    mars_map.fit_bounds(
                    landmark_bounds,
                    padding=(40, 40)
                )

        # LAYER CONTROL
    folium.LayerControl(
            collapsed=False
        ).add_to(mars_map)

    return mars_map


# ============================================================
# GEO TIFF ELEVATION LOADER
# ============================================================

def load_elevation_data(uploaded_file):

    try:

        content = uploaded_file.getvalue()

        with MemoryFile(content) as memfile:

            with memfile.open() as dataset:

                if dataset.count < 1:

                    st.error(
                        "The GeoTIFF contains no elevation band."
                    )

                    return None

                original_height = dataset.height

                original_width = dataset.width

                scale = min(
                    1.0,
                    700 / max(
                        original_height,
                        original_width
                    )
                )

                new_height = max(
                    2,
                    int(original_height * scale)
                )

                new_width = max(
                    2,
                    int(original_width * scale)
                )

                raster = dataset.read(

                    1,

                    out_shape=(new_height, new_width),

                    masked=True,

                    resampling=Resampling.average

                )

                elevation = np.ma.filled(

                    raster.astype(np.float32),

                    np.nan

                )

                transform = (
                    dataset.transform *
                    dataset.transform.scale(

                        original_width / new_width,

                        original_height / new_height

                    )
                )

                metadata = {
    "crs": str(dataset.crs),

    "bounds": dataset.bounds,

    "pixel_size": (
        abs(float(transform.a)),
        abs(float(transform.e)),
    ),

    "original_shape": (
        original_height,
        original_width,
    ),

    "pixel_width": abs(float(transform.a)),
    "pixel_height": abs(float(transform.e)),

    "nodata": dataset.nodata,

    "vertical_units": (
    dataset.units[0]
    if dataset.units and dataset.units[0]
    else None
),

    "units_note": (
        "Check source metadata to confirm "
        "vertical and horizontal units."
    ),
}

        if not np.isfinite(elevation).any():

            st.error(
                "No valid elevation values were found."
            )

            return None

        return {
            "elevation": elevation,
            **metadata
        }

    except Exception as error:

        st.error(
            f"Could not read elevation GeoTIFF: {error}"
        )

        return None

# =====================================
# MARS TERRAIN SLOPE CALCULATOR
# =====================================

def calculate_slope_degrees(
    elevation,
    pixel_width,
    pixel_height
):
    z = np.asarray(elevation, dtype=np.float32)

    if z.ndim != 2 or min(z.shape) < 2:
        raise ValueError("Invalid elevation grid.")

    dx = abs(float(pixel_width))
    dy = abs(float(pixel_height))

    if not np.isfinite(dx) or not np.isfinite(dy):
        raise ValueError("Invalid pixel resolution.")

    if dx <= 0 or dy <= 0:
        raise ValueError("Pixel resolution must be positive.")

    valid = np.isfinite(z)

    if valid.sum() < 4:
        raise ValueError("Insufficient valid elevation data.")

    # Calculate elevation gradients
    dz_dy, dz_dx = np.gradient(z, dy, dx)

    # Convert gradient into slope degrees
    slope = np.degrees(
        np.arctan(
            np.hypot(dz_dx, dz_dy)
        )
    )

    slope[~valid] = np.nan

    return slope


# ============================================================
# ROVER TERRAIN LOADER
# ============================================================

@st.cache_data(show_spinner="Loading Jezero terrain...")
def load_rover_terrain(dem_path, stride=20):

    import rasterio
    from rasterio.enums import Resampling
    import numpy as np

    try:
        with rasterio.open(dem_path) as src:

            # Reduce raster size for faster route planning
            out_height = max(2, int(src.height / stride))
            out_width = max(2, int(src.width / stride))

            raster = src.read(
                1,
                out_shape=(out_height, out_width),
                masked=True,
                resampling=Resampling.average
            )

            # Update pixel spacing after resampling
            transform = src.transform * src.transform.scale(
                src.width / out_width,
                src.height / out_height
            )

            pixel_size_x = abs(float(transform.a))
            pixel_size_y = abs(float(transform.e))

            dem_crs = str(src.crs)

            # Convert masked raster to floating-point elevation array
            elevation = np.ma.filled(
                raster.astype("float64"),
                np.nan
            )

        # Check that elevation data exists
        if not np.isfinite(elevation).any():
            raise ValueError("The DEM contains no valid elevation data.")

        # Fill missing cells for gradient calculation
        median_elevation = float(np.nanmedian(elevation))

        filled = np.where(
            np.isfinite(elevation),
            elevation,
            median_elevation
        )

        # Calculate terrain gradients
        gradient_y, gradient_x = np.gradient(
            filled,
            pixel_size_y,
            pixel_size_x
        )

        # Calculate slope in degrees
        slope = np.degrees(
            np.arctan(
                np.sqrt(
                    gradient_x ** 2 + gradient_y ** 2
                )
            )
        )

        # Restore NoData locations as invalid
        slope[~np.isfinite(elevation)] = np.nan

        return (
            elevation,
            slope,
            pixel_size_x,
            pixel_size_y,
            dem_crs
        )

    except Exception as error:

        st.error(
            f"Could not load rover terrain: {error}"
        )

        raise


# ============================================================
# TERRAIN-AWARE A* PATHFINDING ALGORITHM
# ============================================================

def terrain_aware_astar(

    elevation,

    slope,

    valid,

    start,

    goal,

    pixel_size_x,

    pixel_size_y,

    max_allowed_slope

):

    rows, cols = elevation.shape

    if not (
        0 <= start[0] < rows
        and 0 <= start[1] < cols
    ):

        return None

    if not (
        0 <= goal[0] < rows
        and 0 <= goal[1] < cols
    ):

        return None

    if not valid[start] or not valid[goal]:

        return None

    if (
        not np.isfinite(slope[start])
        or slope[start] > max_allowed_slope
    ):

        return None

    if (
        not np.isfinite(slope[goal])
        or slope[goal] > max_allowed_slope
    ):

        return None

    directions = [

        (-1, 0),
        (1, 0),
        (0, -1),
        (0, 1),

        (-1, -1),
        (-1, 1),
        (1, -1),
        (1, 1)

    ]

    def heuristic(node):

        dr = (
            goal[0] - node[0]
        ) * pixel_size_y

        dc = (
            goal[1] - node[1]
        ) * pixel_size_x

        return math.hypot(dr, dc)

    open_set = [

        (
            heuristic(start),
            0.0,
            start
        )

    ]

    came_from = {}

    cost_so_far = {
        start: 0.0
    }

    found = False

    while open_set:

        _, current_cost, current = heapq.heappop(
            open_set
        )

        if current_cost > cost_so_far.get(
            current,
            float("inf")
        ):

            continue

        if current == goal:

            found = True

            break

        for dr, dc in directions:

            nr = current[0] + dr

            nc = current[1] + dc

            if not (
                0 <= nr < rows
                and 0 <= nc < cols
            ):

                continue

            neighbor = (nr, nc)

            if not valid[neighbor]:

                continue

            neighbor_slope = float(
                slope[neighbor]
            )

            if not np.isfinite(neighbor_slope):

                continue

            if neighbor_slope > max_allowed_slope:

                continue

            # Prevent diagonal corner cutting.

            if dr != 0 and dc != 0:

                side_a = (
                    current[0] + dr,
                    current[1]
                )

                side_b = (
                    current[0],
                    current[1] + dc
                )

                if (

                    not valid[side_a]

                    or not valid[side_b]

                    or not np.isfinite(slope[side_a])

                    or not np.isfinite(slope[side_b])

                    or slope[side_a] > max_allowed_slope

                    or slope[side_b] > max_allowed_slope

                ):

                    continue

            step_distance = math.hypot(

                dr * pixel_size_y,

                dc * pixel_size_x

            )

            slope_penalty = (
                1.0 + neighbor_slope / 15.0
            )

            new_cost = (

                current_cost

                + step_distance * slope_penalty

            )

            if new_cost < cost_so_far.get(

                neighbor,

                float("inf")

            ):

                cost_so_far[neighbor] = new_cost

                came_from[neighbor] = current

                priority = (
                    new_cost + heuristic(neighbor)
                )

                heapq.heappush(

                    open_set,

                    (
                        priority,
                        new_cost,
                        neighbor
                    )

                )

    if not found:

        return None

    # Reconstruct the route.

    path = [goal]

    while path[-1] != start:

        previous = came_from.get(
            path[-1]
        )

        if previous is None:

            return None

        path.append(previous)

    path.reverse()

    return path


# ============================================================
# 3D TERRAIN VISUALIZATION FUNCTION
# ============================================================


def display_3d_terrain(elevation_data):

    z = np.asarray(elevation_data["elevation"], dtype=float)
    # Preserve actual elevations and create exaggerated display elevations
    z_min = np.nanmin(z)

    z_visual = z

    rows, cols = z.shape

    st.write("CRS:", elevation_data["crs"])
    st.write("Raster bounds:", elevation_data["bounds"])
    st.write("Pixel width:", elevation_data["pixel_width"])
    st.write("Pixel height:", elevation_data["pixel_height"])
    st.write("Raster dimensions:", rows, cols)

    pixel_width = abs(elevation_data["pixel_width"])
    pixel_height = abs(elevation_data["pixel_height"])

    st.write("Terrain rows:", rows)
    st.write("Terrain columns:", cols)
    st.write("Pixel width:", pixel_width)
    st.write("Pixel height:", pixel_height)

    x = np.arange(cols) * pixel_width
    y = np.arange(rows) * pixel_height

    # Mars-inspired scientific elevation palette
    mars_colorscale = [
        [0.00, "#102B46"],
        [0.15, "#176B87"],
        [0.30, "#65A9B0"],
        [0.45, "#D5C6A0"],
        [0.60, "#E9B66D"],
        [0.75, "#C97842"],
        [0.90, "#91452F"],
        [1.00, "#51271F"]
    ]

   # Temporary DEM diagnostics
    st.write("DEM minimum:", float(np.nanmin(z)))
    st.write("DEM maximum:", float(np.nanmax(z)))
    st.write("DEM elevation range:", float(np.nanmax(z) - np.nanmin(z)))


    fig = go.Figure(
        data=[
            go.Surface(
                x=x,
                y=y,
                z=z_visual,
                surfacecolor=z,
                colorscale=mars_colorscale,
                showscale=True,
                connectgaps=False,

                colorbar=dict(
                    title="Elevation (m)",
                    thickness=18,
                    len=0.75,
                    tickfont=dict(color="white")
                ),

                lighting=dict(
                    ambient=0.65,
                    diffuse=0.85,
                    specular=0.25,
                    roughness=0.8,
                    fresnel=0.15
                ),

                lightposition=dict(
                    x=1000,
                    y=1000,
                    z=2000
                ),

                contours=dict(
                    z=dict(
                        show=True,
                        usecolormap=True,
                        highlightcolor="#FFFFFF",
                        project_z=False,
                        width=1
                    )
                ),

                hovertemplate=(
                    "East-West: %{x:.1f} m<br>"
                    "North-South: %{y:.1f} m<br>"
                    "Actual Elevation: %{z:.2f} m"
                    "<extra>Jezero Crater</extra>"
                )
            )
        ]
    )

    fig.update_layout(
        title=dict(
            text="JEZERO CRATER | INTERACTIVE 3D ELEVATION MODEL",
            x=0.5,
            xanchor="center",
            font=dict(size=19, color="#F2A66A")
        ),

        height=900,
        template="plotly_dark",
        paper_bgcolor="#17100e",
        plot_bgcolor="#17100e",

        font=dict(
            color="white",
            family="Arial",
            size=12
        ),

        margin=dict(
            l=10,
            r=30,
            t=85,
            b=20
        ),

        scene=dict(
            xaxis_title="East-West distance (m)",
            yaxis_title="North-South distance (m)",
            zaxis_title="Elevation (m)",

            aspectmode="manual",
            aspectratio=dict(
                x=0.714,
                y=1,
                z=0.14
            ),

            camera=dict(
                eye=dict(
                    x=1.65,
                    y=-1.8,
                    z=1.15
                )
            ),

            bgcolor="#17100e",

            xaxis=dict(
                showbackground=True,
                backgroundcolor="#251812",
                gridcolor="#594234",
                zerolinecolor="#80604B"
            ),

            yaxis=dict(
                showbackground=True,
                backgroundcolor="#251812",
                gridcolor="#594234",
                zerolinecolor="#80604B"
            ),

            zaxis=dict(
                showbackground=True,
                backgroundcolor="#251812",
                gridcolor="#594234",
                zerolinecolor="#80604B"
            )
        )
    )

    st.plotly_chart(
        fig,
        use_container_width=True,
        config={
            "scrollZoom": True,
            "displayModeBar": True,
            "responsive": True
        },
        key="jezero_3d_elevation_surface"
    )


# ============================================================
# MAIN HEADER
# ============================================================

st.markdown("""
# 🔴 MARS-NAV
### MARTIAN MISSION CONTROL SYSTEM

**PLANET:** MARS |
**TARGET REGION:** JEZERO CRATER |
**MODE:** EXPLORATION / RESEARCH PROTOTYPE
""")

st.divider()


# ============================================================
# SIDEBAR NAVIGATION
# ============================================================

st.sidebar.markdown("# 🔴 MARS-NAV")

st.sidebar.caption("MISSION CONTROL TERMINAL")

page = st.sidebar.radio(
    "SYSTEM NAVIGATION",
    [
        "Mission Overview",
        "Mars Surface Map",
        "3D Jezero Terrain",
        "Rover Route Planner",
        "Terrain Analysis",
        "Survival Planner",
        "Scientific Data"
    ]
)

st.sidebar.divider()

st.sidebar.markdown("### MISSION SETTINGS")

st.session_state.mission_sol = st.sidebar.number_input(
    "Mission Sol",
    min_value=1,
    max_value=1000,
    value=int(st.session_state.mission_sol)
)

st.session_state.rover_speed = st.sidebar.slider(
    "Rover speed (km/hour)",
    min_value=0.1,
    max_value=5.0,
    value=float(st.session_state.rover_speed),
    step=0.1
)

st.sidebar.divider()

st.sidebar.markdown("### SYSTEM STATUS")

st.sidebar.success("Navigation system ready")

st.sidebar.info("Target: Jezero Crater")

st.sidebar.caption(
    "MARS-NAV | DEVELOPMENT VERSION 3.0"
)


# ============================================================
# PAGE 1 | MISSION OVERVIEW
# ============================================================

if page == "Mission Overview":

    st.markdown("## 🚀 MISSION OVERVIEW")

    st.write(
        "MARS-NAV is an interactive Martian exploration "
        "prototype focused on the Jezero Crater region."
    )

    c1, c2, c3, c4 = st.columns(4)

    c1.metric(
        "Mission Sol",
        st.session_state.mission_sol
    )

    c2.metric(
        "Target Planet",
        "Mars"
    )

    c3.metric(
        "Rover Speed",
        f"{st.session_state.rover_speed:.1f} km/h"
    )

    c4.metric(
        "Target Region",
        "Jezero"
    )

    st.divider()

    st.markdown("### MISSION OBJECTIVES")

    col1, col2 = st.columns(2)

    with col1:

        st.markdown("""
        ### 01. Surface Exploration

        - Explore Martian surface maps.
        - Locate important geological features.
        - Understand elevation differences.

        ### 02. Rover Navigation

        - Plan rover routes.
        - Calculate traversable paths.
        - Estimate travel distances.
        """)

    with col2:

        st.markdown("""
        ### 03. Terrain Intelligence

        - Visualize elevation data.
        - Identify steep terrain.
        - Analyse potential hazards.

        ### 04. Survival Planning

        - Estimate mission resources.
        - Calculate water requirements.
        - Calculate oxygen requirements.
        """)

    st.divider()

    st.markdown("### MISSION AREA")

    st.write("DEBUG LANDMARKS:", len(LANDMARKS), list(LANDMARKS.keys()))

    mission_map = create_map()

    st_folium(
        mission_map,
        height=450,
        use_container_width=True,
        key="overview_mars_map"
    )

    st.info(
        "MARS-NAV is a research and educational prototype. "
        "It is not an operational spacecraft navigation system."
    )


# ============================================================
# PAGE 2 | MARS SURFACE MAP
# ============================================================

elif page == "Mars Surface Map":

    st.markdown("## 🌍 GLOBAL MARS EXPLORER")

    st.write(
        "Explore the Jezero Crater region using "
        "Mars MOLA colour-relief terrain tiles."
    )

    st.markdown("### Exploration Landmarks")

    landmark_data = pd.DataFrame(
        [
            {
                "Location": name,
                "Latitude": coordinates[0],
                "Longitude": coordinates[1]
            }
            for name, coordinates in LANDMARKS.items()
        ]
    )

    st.dataframe(
        landmark_data,
        use_container_width=True,
        hide_index=True
    )

    st.divider()

    st.markdown("### Interactive Mars Surface Map")

    mars_map = create_map()

    st_folium(
        mars_map,
        height=650,
        use_container_width=True,
        key="surface_mars_map"
    )

    st.caption(
        "Map source: Mars MOLA global colour-relief tile service. "
        "The map is a visual basemap, not a rover-scale "
        "high-resolution navigation dataset."
    )


# ============================================================
# PAGE 3 | 3D JEZERO TERRAIN
# ============================================================

elif page == "3D Jezero Terrain":

    st.markdown("## 🏔️ 3D JEZERO CRATER TERRAIN")

    st.write(
        "Visualize actual elevation measurements "
        "from a Mars Digital Terrain Model."
    )

    st.info(
        "Upload a single-band elevation GeoTIFF. "
        "Coloured images and screenshots cannot be used "
        "as numerical elevation data."
    )

    st.markdown("### Upload elevation dataset")

    uploaded_dem = st.file_uploader(
        "Select Mars elevation GeoTIFF",
        type=["tif", "tiff"],
        key="terrain_upload"
    )

    if uploaded_dem is not None:

        with st.spinner("Processing Jezero elevation data..."):

            elevation_data = load_elevation_data(
                uploaded_dem
            )

        if elevation_data is not None:

            st.session_state["mars_elevation_data"] = elevation_data

            st.success(
                "Elevation dataset successfully loaded."
            )

            z = elevation_data["elevation"]

            valid_values = z[
                np.isfinite(z)
            ]

            st.markdown("### Terrain Statistics")

            c1, c2, c3 = st.columns(3)

            c1.metric(
                "Minimum Elevation",
                f"{np.min(valid_values):.2f}"
            )

            c2.metric(
                "Maximum Elevation",
                f"{np.max(valid_values):.2f}"
            )

            c3.metric(
                "Elevation Range",
                f"{np.ptp(valid_values):.2f}"
            )

            st.write(
                "**Coordinate Reference System:**",
                elevation_data["crs"]
            )

            st.write(
                "**Raster Dimensions:**",
                f"{z.shape[0]} × {z.shape[1]}"
            )

            st.divider()

            st.markdown("### Interactive 3D Elevation Surface")

            display_3d_terrain(
                elevation_data
            )

                # ==================================
                # MARS TERRAIN SLOPE ANALYSIS
                # ==================================
            st.divider()

            st.markdown(
                    "### 🏔️ Terrain Slope Analysis"
                )

            pixel_width = elevation_data.get(
                "pixel_width"
            )

            pixel_height = elevation_data.get(
                "pixel_height"
            )

            vertical_units = elevation_data.get(
                "vertical_units"
            )

            try:
                import rasterio
                import plotly.graph_objects as go

                crs_obj = rasterio.crs.CRS.from_user_input(
                    elevation_data["crs"]
                )

                horizontal_units = (
                    crs_obj.linear_units or ""
                ).lower()

                vertical_units = (
                    str(vertical_units or "").lower()
                )

                metre_units = {
                    "m", "meter", "meters",
                    "metre", "metres"
                }

                if not crs_obj.is_projected:
                    st.warning(
                        "Slope calculation requires "
                        "projected horizontal coordinates."
                    )

                elif horizontal_units not in metre_units:
                    st.warning(
                        "Horizontal CRS units must be "
                        "verified as metres."
                    )

                elif vertical_units not in metre_units:
                    st.warning(
                        "Elevation units are not confirmed "
                        "as metres. Check the DEM source."
                    )

                elif pixel_width is None or pixel_height is None:
                    st.error(
                        "GeoTIFF pixel resolution is missing."
                    )

                else:
                    slope_degrees = calculate_slope_degrees(
                        elevation_data["elevation"],
                        pixel_width,
                        pixel_height
                    )

                    valid_slope = slope_degrees[
                        np.isfinite(slope_degrees)
                    ]

                    if valid_slope.size > 0:

                        col1, col2, col3 = st.columns(3)

                        col1.metric(
                            "Minimum Slope",
                            f"{np.min(valid_slope):.2f}°"
                        )

                        col2.metric(
                            "Maximum Slope",
                            f"{np.max(valid_slope):.2f}°"
                        )

                        col3.metric(
                            "Average Slope",
                            f"{np.mean(valid_slope):.2f}°"
                        )

                        fig = go.Figure(
                            data=go.Heatmap(
                                z=slope_degrees,
                                colorscale=[
                                    [0.0, "#16A34A"],
                                    [0.25, "#EAB308"],
                                    [0.60, "#F97316"],
                                    [1.0, "#DC2626"]
                                ],
                                zmin=0,
                                zmax=max(
                                    30,
                                    float(
                                        np.nanpercentile(
                                            valid_slope,
                                            98
                                        )
                                    )
                                ),
                                colorbar=dict(
                                    title="Slope (°)"
                                )
                            )
                        )

                        fig.update_layout(
                            title="Mars Terrain Slope Map",
                            height=550,
                            xaxis_title="Raster Column",
                            yaxis_title="Raster Row"
                        )

                        st.plotly_chart(
                            fig,
                            use_container_width=True
                        )

                    else:
                        st.error(
                            "No valid slope measurements found."
                        )

            except Exception as error:
                st.error(
                    f"Slope analysis failed: {error}"
                )
        st.caption(
            "Rotate, zoom and inspect the terrain surface. "
            "Confirm elevation units using the original "
            "dataset metadata."
        )


    else:

        st.warning(
            "No uploaded elevation GeoTIFF. "
            "Upload your DTM to display its 3D surface."
        )

    st.divider()

    # --------------------------------------------------------
    # CESIUM 3D MARS VIEWER
    # --------------------------------------------------------

    st.markdown("## 🌍 Interactive 3D Mars Explorer")

    viewer_path = APP_DIR / "mars_viewer.html"

    if viewer_path.exists():

        try:

            viewer_html = viewer_path.read_text(
                encoding="utf-8"
            )

            cesium_token = st.secrets.get(
                "CESIUM_ION_TOKEN",
                ""
            )

            if cesium_token:

                viewer_html = viewer_html.replace(
                    "TOKEN_PLACEHOLDER",
                    cesium_token
                )

                components.html(
                    viewer_html,
                    height=700,
                    scrolling=False
                )

            else:

                st.warning(
                    "CESIUM_ION_TOKEN is missing from "
                    ".streamlit/secrets.toml. "
                    "Configure your Cesium token to load this viewer."
                )

        except Exception as error:

            st.error(
                f"Could not load the Mars viewer: {error}"
            )

    else:

        st.info(
            "mars_viewer.html was not found. "
            "The uploaded GeoTIFF terrain viewer remains available."
        )

    st.divider()

    # --------------------------------------------------------
    # JEZERO INTERACTIVE TERRAIN HTML
    # --------------------------------------------------------

    st.markdown("## 🪐 Interactive 3D Jezero Terrain")

    terrain_html_path = APP_DIR / "jezero_3d.html"

    if st.button(
        "Load Interactive 3D Jezero Terrain",
        type="primary"
    ):

        st.session_state.show_jezero_3d = True

    if st.session_state.show_jezero_3d:

        if terrain_html_path.exists():

            try:

                terrain_html = terrain_html_path.read_text(
                    encoding="utf-8"
                )

                components.html(
                    terrain_html,
                    height=650,
                    scrolling=False
                )

            except Exception as error:

                st.error(
                    f"Could not load Jezero terrain: {error}"
                )

        else:

            st.warning(
                "jezero_3d.html was not found. "
                "Run prepare_jezero.py first."
            )

    else:

        st.caption(
            "Click the button above to load the prepared "
            "interactive Jezero terrain."
        )


# ============================================================
# PAGE 4 | TERRAIN-AWARE ROVER ROUTE PLANNER
# ============================================================

elif page == "Rover Route Planner":

    st.markdown(
        "## 🤖 MARS-NAV | ROVER ROUTE PLANNER"
    )

    st.caption(
        "Terrain-aware A* route planning using the "
        "Jezero Crater Digital Elevation Model."
    )

    st.divider()

    # --------------------------------------------------------
    # LOAD AVAILABLE DEM DATA
    # --------------------------------------------------------

    dem_candidates = sorted(
        DATA_DIR.glob("*CTXDEM*.tif")
    )

    dem_source_note = "CTX DEM"

    if not dem_candidates:

        dem_candidates = sorted(
            DATA_DIR.glob("*DTM*.tif")
        )

        dem_source_note = "DTM fallback"

    if not dem_candidates:

        st.error(
            "No CTXDEM or DTM GeoTIFF found."
        )

        st.info(
            "Place your Jezero elevation GeoTIFF "
            "inside the existing data folder."
        )

    else:

        dem_path = dem_candidates[0]

        st.caption(
            f"Terrain file: {dem_path.name}"
        )

        st.caption(
            f"Selected terrain source: {dem_source_note}"
        )

        try:

            with st.spinner(
                "Loading and analysing Martian terrain..."
            ):

                (
                    elevation,
                    slope,
                    pixel_size_x,
                    pixel_size_y,
                    dem_crs
                ) = load_rover_terrain(
                    str(dem_path),
                    stride=20
                )

            valid = (
                np.isfinite(elevation)
                &
                np.isfinite(slope)
            )

            rows, cols = elevation.shape

            # ------------------------------------------------
            # TERRAIN STATUS
            # ------------------------------------------------

            st.success(
                f"Terrain successfully loaded: "
                f"{rows:,} × {cols:,} cells"
            )

            st.write(
                f"**Coordinate reference system:** {dem_crs}"
            )

            st.write(
                f"**Pixel spacing:** "
                f"{pixel_size_x:.3f} × "
                f"{pixel_size_y:.3f} source units"
            )

            st.divider()

            # ------------------------------------------------
            # ROUTE CONFIGURATION
            # ------------------------------------------------

            st.markdown(
                "### 🎯 ROUTE CONFIGURATION"
            )

            col1, col2, col3 = st.columns(3)

            with col1:

                start_row = st.slider(
                    "Start row",
                    0,
                    rows - 1,
                    min(
                        rows - 1,
                        int(rows * 0.25)
                    ),
                    key="route_start_row"
                )

            with col2:

                end_row = st.slider(
                    "Destination row",
                    0,
                    rows - 1,
                    min(
                        rows - 1,
                        int(rows * 0.75)
                    ),
                    key="route_end_row"
                )

            with col3:

                max_allowed_slope = st.slider(
                    "Maximum permitted slope (degrees)",
                    5,
                    45,
                    25,
                    key="route_max_slope"
                )

            col4, col5 = st.columns(2)

            with col4:

                start_col = st.slider(
                    "Start column",
                    0,
                    cols - 1,
                    min(
                        cols - 1,
                        int(cols * 0.20)
                    ),
                    key="route_start_col"
                )

            with col5:

                end_col = st.slider(
                    "Destination column",
                    0,
                    cols - 1,
                    min(
                        cols - 1,
                        int(cols * 0.80)
                    ),
                    key="route_end_col"
                )

            start = (
                start_row,
                start_col
            )

            goal = (
                end_row,
                end_col
            )

            st.info(
                "The planner avoids NoData cells and terrain "
                "above the selected slope threshold. "
                "Steeper traversable cells receive an additional "
                "movement-cost penalty."
            )

            st.divider()

            # ------------------------------------------------
            # CALCULATE SAFEST ROUTE
            # ------------------------------------------------

            if st.button(
                "🚀 Calculate Safest Rover Route",
                type="primary",
                use_container_width=True
            ):

                if start == goal:

                    st.warning(
                        "Choose different start and destination cells."
                    )

                    st.session_state.last_rover_route = None

                elif not valid[start]:

                    st.error(
                        "The selected start cell contains "
                        "invalid elevation data."
                    )

                    st.session_state.last_rover_route = None

                elif not valid[goal]:

                    st.error(
                        "The selected destination cell contains "
                        "invalid elevation data."
                    )

                    st.session_state.last_rover_route = None

                else:

                    with st.spinner(
                        "Running A* terrain-aware pathfinding..."
                    ):

                        path = terrain_aware_astar(

                            elevation=elevation,

                            slope=slope,

                            valid=valid,

                            start=start,

                            goal=goal,

                            pixel_size_x=pixel_size_x,

                            pixel_size_y=pixel_size_y,

                            max_allowed_slope=max_allowed_slope

                        )

                    st.session_state.last_rover_route = (

                        path,

                        start,

                        goal,

                        max_allowed_slope,

                        str(dem_path)

                    )

            # ------------------------------------------------
            # DISPLAY CALCULATED ROUTE
            # ------------------------------------------------

            route_result = (
                st.session_state.last_rover_route
            )

            if route_result is not None:

                (
                    path,
                    saved_start,
                    saved_goal,
                    saved_limit,
                    saved_dem
                ) = route_result

                if saved_dem != str(dem_path):

                    st.session_state.last_rover_route = None

                    st.info(
                        "Terrain source changed. "
                        "Calculate the route again."
                    )

                elif path is None:

                    st.warning(
                        "No traversable route found. "
                        "Try different endpoints or increase "
                        "the permitted slope threshold."
                    )

                else:

                    # ----------------------------------------
                    # ROUTE COORDINATES
                    # ----------------------------------------

                    path_rows = [
                        point[0]
                        for point in path
                    ]

                    path_cols = [
                        point[1]
                        for point in path
                    ]

                    # ----------------------------------------
                    # DISTANCE CALCULATION
                    # ----------------------------------------

                    route_distance = sum(

                        math.hypot(

                            (
                                path[i + 1][0]
                                - path[i][0]
                            ) * pixel_size_y,

                            (
                                path[i + 1][1]
                                - path[i][1]
                            ) * pixel_size_x

                        )

                        for i in range(
                            len(path) - 1
                        )

                    )

                    route_elevations = np.array(

                        [
                            elevation[r, c]
                            for r, c in path
                        ],

                        dtype=float

                    )

                    # ----------------------------------------
                    # ELEVATION GAIN
                    # ----------------------------------------

                    elevation_gain = float(

                        np.maximum(

                            np.diff(route_elevations),

                            0

                        ).sum()

                    )

                    # ----------------------------------------
                    # SLOPE STATISTICS
                    # ----------------------------------------

                    route_slopes = [

                        float(slope[r, c])
                        for r, c in path

                    ]

                    maximum_route_slope = max(
                        route_slopes
                    )

                    mean_route_slope = float(
                        np.mean(route_slopes)
                    )

                    # ----------------------------------------
                    # DISTANCE AND TIME
                    # ----------------------------------------

                    # Assumes horizontal DEM units are metres.
                    # Confirm units from the original DEM metadata.

                    route_distance_km = (
                        route_distance / 1000.0
                    )

                    travel_hours = (

                        route_distance_km /

                        max(
                            float(
                                st.session_state.rover_speed
                            ),
                            0.1
                        )

                    )

                    # ----------------------------------------
                    # ROUTE MEASUREMENTS
                    # ----------------------------------------

                    st.divider()

                    st.markdown(
                        "### 📊 ROUTE MEASUREMENTS"
                    )

                    m1, m2, m3, m4 = st.columns(4)

                    m1.metric(
                        "Route Distance",
                        f"{route_distance_km:.2f} km"
                    )

                    m2.metric(
                        "Estimated Travel Time",
                        f"{travel_hours:.2f} hours"
                    )

                    m3.metric(
                        "Elevation Gain",
                        f"{elevation_gain:.2f}"
                    )

                    m4.metric(
                        "Maximum Slope",
                        f"{maximum_route_slope:.2f}°"
                    )

                    st.caption(
                        f"Mean slope: {mean_route_slope:.2f}° | "
                        f"Route points: {len(path):,} | "
                        f"Slope threshold: {saved_limit}°"
                    )

                    # ----------------------------------------
                    # ROUTE VISUALIZATION
                    # ----------------------------------------

                    st.markdown(
                        "### 🗺️ TERRAIN-AWARE ROVER ROUTE"
                    )

                    route_fig = go.Figure()

                    route_fig.add_trace(

                        go.Heatmap(

                            z=elevation,

                            colorscale="Earth",

                            colorbar=dict(
                                title="Elevation"
                            ),

                            hovertemplate=(
                                "Column: %{x}<br>"
                                "Row: %{y}<br>"
                                "Elevation: %{z:.2f}"
                                "<extra></extra>"
                            )

                        )

                    )

                    route_fig.add_trace(

                        go.Scatter(

                            x=path_cols,

                            y=path_rows,

                            mode="lines",

                            name="A* Rover Route",

                            line=dict(
                                color="#00E5FF",
                                width=4
                            )

                        )

                    )

                    route_fig.add_trace(

                        go.Scatter(

                            x=[
                                saved_start[1],
                                saved_goal[1]
                            ],

                            y=[
                                saved_start[0],
                                saved_goal[0]
                            ],

                            mode="markers+text",

                      text=["Start", "Destination"],

                            textposition="top center",

                            name="Route Endpoints",

                            marker=dict(

                                size=12,

                                color=[
                                    "#00FF88",
                                    "#FF5252"
                                ]

                            )

                        )

                    )

                    route_fig.update_layout(

                        title=(
                            "MARS-NAV | "
                            "A* TERRAIN-AWARE ROVER ROUTE"
                        ),

                        template="plotly_dark",

                        xaxis_title="DEM Column",

                        yaxis_title="DEM Row",

                        yaxis=dict(
                            autorange="reversed"
                        ),

                        height=600,

                        paper_bgcolor="#17100E",

                        plot_bgcolor="#241713",

                        font=dict(
                            color="white"
                        )

                    )

                    st.plotly_chart(
                    route_fig,
                    width="stretch"
                )
                    st.session_state.last_rover_route = (
                        path,
                        start,
                        goal,
                        max_allowed_slope,
                        str(dem_path)
                        )


                    # ----------------------------------------
                    # ELEVATION PROFILE
                    # ----------------------------------------

                    st.markdown(
                        "### ⛰️ ROVER ELEVATION PROFILE"
                    )

                    profile_fig = go.Figure()

                    profile_fig.add_trace(

                        go.Scatter(

                            x=list(
                                range(
                                    len(route_elevations)
                                )
                            ),

                            y=route_elevations,

                            mode="lines",

                            fill="none",

                            name="Route Elevation",

                            line=dict(
                                color="#FF9B62",
                                width=3
                            )

                        )

                    )

                    profile_fig.update_layout(

                        title="Rover Route Elevation Profile",

                        xaxis_title="Route Point",

                        yaxis_title="Elevation (source units)",

                        template="plotly_dark",

                        height=350

                    )
                    st.plotly_chart(
                        profile_fig,
                        width="stretch"
                    )
                    

                    # ----------------------------------------
                    # CSV EXPORT
                    # ----------------------------------------

                    st.markdown(
                        "### 📥 EXPORT MISSION ROUTE"
                    )

                    route_export = pd.DataFrame({

                        "route_order": range(
                            1,
                            len(path) + 1
                        ),

                        "dem_row": path_rows,

                        "dem_column": path_cols,

                        "elevation": route_elevations,

                        "slope_degrees": route_slopes

                    })

                    st.download_button(

                        label="Download Rover Route CSV",

                        data=route_export.to_csv(
                            index=False
                        ).encode("utf-8"),

                        file_name="mars_nav_rover_route.csv",

                        mime="text/csv"

                    )

                    st.success(
                        "Terrain-aware demonstration route calculated."
                    )

                    st.warning(
                        "Research prototype limitation: distance assumes "
                        "the DEM horizontal coordinate units are metres. "
                        "Verify the CRS and units. This planner does not "
                        "yet account for rock detection, wheel slip, "
                        "energy consumption or validated rover safety rules."
                    )

        except Exception as error:

            st.error(
                f"Terrain loading or route planning failed: {error}"
            )

            st.exception(error)



# ============================================================
# PAGE 5 | TERRAIN ANALYSIS
# ============================================================

elif page == "Terrain Analysis":

    st.markdown("## 🏔️ TERRAIN ANALYSIS")

    st.write(
        "Analyse Martian elevation, slope and surface "
        "characteristics using Digital Terrain Models."
    )

    st.divider()

    st.markdown("### Important Terrain Parameters")

    terrain_table = pd.DataFrame({

        "Parameter": [
            "Elevation",
            "Slope",
            "Surface Roughness",
            "Crater Proximity",
            "Obstacle Density"
        ],

        "Purpose": [
            "Understand height differences",
            "Identify steep terrain",
            "Estimate surface irregularity",
            "Identify nearby depressions",
            "Estimate route difficulty"
        ],

        "Importance": [
            "High",
            "High",
            "High",
            "Medium",
            "High"
        ]

    })

    st.dataframe(
        terrain_table,
        use_container_width=True,
        hide_index=True
    )

    st.divider()

    st.markdown("### Upload Elevation Dataset")

    terrain_file = st.file_uploader(

        "Upload Mars elevation GeoTIFF",

        type=["tif", "tiff"],

        key="analysis_upload"

    )

    if terrain_file is not None:

        with st.spinner(
            "Analysing elevation dataset..."
        ):

            data = load_elevation_data(
                terrain_file
            )

        if data is not None:

            z = data["elevation"]

            valid_values = z[
                np.isfinite(z)
            ]

            # --------------------------------------------
            # ELEVATION STATISTICS
            # --------------------------------------------

            st.markdown(
                "### Elevation Statistics"
            )

            a, b, c = st.columns(3)

            a.metric(
                "Minimum Elevation",
                f"{np.min(valid_values):.2f}"
            )

            b.metric(
                "Maximum Elevation",
                f"{np.max(valid_values):.2f}"
            )

            c.metric(
                "Mean Elevation",
                f"{np.mean(valid_values):.2f}"
            )

            st.divider()

            # --------------------------------------------
            # ELEVATION HEATMAP
            # --------------------------------------------

            
            st.markdown(
                "### Martian Elevation Distribution"
            )

            mars_colorscale = [
                [0.00, "#102B46"],
                [0.15, "#176B87"],
                [0.30, "#65A9B0"],
                [0.45, "#D5C6A0"],
                [0.60, "#E9B66D"],
                [0.75, "#C97842"],
                [0.90, "#91452F"],
                [1.00, "#51271F"]
            ]

            heatmap = go.Figure(
                data=[
                    go.Heatmap(
                        z=z,
                        colorscale=mars_colorscale,
                        zsmooth="best",
                        colorbar=dict(
                            title="Elevation (m)",
                            thickness=18,
                            len=0.85,
                            ticks="outside"
                        ),
                        hovertemplate=(
                            "Grid Column: %{x}<br>"
                            "Grid Row: %{y}<br>"
                            "Elevation: %{z:.2f} m"
                            "<extra>Jezero Crater</extra>"
                        )
                    )
                ]
            )

            heatmap.update_layout(
                title=dict(
                    text="JEZERO CRATER | MARTIAN ELEVATION MAP",
                    x=0.5,
                    xanchor="center"
                ),
                height=650,
                template="plotly_dark",
                xaxis_title="Terrain Grid Column",
                yaxis_title="Terrain Grid Row",
                font=dict(
                    family="Arial",
                    size=13
                ),
                margin=dict(
                    l=65,
                    r=75,
                    t=85,
                    b=65
                ),
                hoverlabel=dict(
                    bgcolor="#241710",
                    font_size=13
                )
            )

            heatmap.update_xaxes(
                showgrid=False,
                zeroline=False
            )

            heatmap.update_yaxes(
                showgrid=False,
                zeroline=False
            )

            st.plotly_chart(
                heatmap,
                use_container_width=True,
                key="terrain_elevation_heatmap"
            )

            st.divider()

            # --------------------------------------------
            # SLOPE ANALYSIS
            # --------------------------------------------

            st.markdown(
                "### Terrain Slope Analysis"
            )

            pixel_x = max(
                data["pixel_width"],
                1e-9
            )

            pixel_y = max(
                data["pixel_height"],
                1e-9
            )

            filled = np.where(
                np.isfinite(z),
                z,
                float(np.nanmedian(z))
            )

            dz_dy, dz_dx = np.gradient(
                filled,
                pixel_y,
                pixel_x
            )

            slope_map = np.degrees(
                np.arctan(
                    np.sqrt(
                        dz_dx ** 2 + dz_dy ** 2
                    )
                )
            )

            slope_map[
                ~np.isfinite(z)
            ] = np.nan

            valid_slopes = slope_map[
                np.isfinite(slope_map)
            ]

            if len(valid_slopes) > 0:

                slope1, slope2, slope3 = st.columns(3)

                slope1.metric(
                    "Mean Slope",
                    f"{np.mean(valid_slopes):.2f}°"
                )

                slope2.metric(
                    "Maximum Slope",
                    f"{np.max(valid_slopes):.2f}°"
                )

                slope3.metric(
                    "Steep Terrain Cells",
                    f"{np.sum(valid_slopes > 25):,}"
                )

                slope_fig = go.Figure(

                    data=[

                        go.Heatmap(

                            z=slope_map,

                            colorscale="Turbo",

                            colorbar=dict(
                                title="Slope (degrees)"
                            )

                        )

                    ]

                )

                slope_fig.update_layout(

                    title="JEZERO CRATER | TERRAIN SLOPE MAP",

                    height=550,

                    template="plotly_dark"

                )

                st.plotly_chart(
                    slope_fig,
                    use_container_width=True
                )

                st.caption(
                    "Slope is calculated from elevation gradients. "
                    "Interpretation depends on the DEM's horizontal "
                    "and vertical units and resolution."
                )

    else:

        st.info(
            "Upload a GeoTIFF to begin terrain analysis."
        )


# ============================================================
# PAGE 6 | SURVIVAL PLANNER
# ============================================================

elif page == "Survival Planner":

    st.markdown(
        "## 🧑‍🚀 MISSION SURVIVAL PLANNER"
    )

    st.write(
        "Estimate essential resources for a hypothetical "
        "Martian surface exploration mission."
    )

    st.divider()

    # --------------------------------------------
    # MISSION INPUTS
    # --------------------------------------------

    col1, col2 = st.columns(2)

    with col1:

        crew = st.number_input(

            "Crew Members",

            min_value=1,

            max_value=12,

            value=2,

            step=1

        )

        duration = st.number_input(

            "Mission Duration (days)",

            min_value=1,

            max_value=1000,

            value=30,

            step=1

        )

    with col2:

        water_per_person = st.number_input(

            "Water Requirement per Person (litres/day)",

            min_value=1.0,

            max_value=20.0,

            value=3.0,

            step=0.5

        )

        oxygen_per_person = st.number_input(

            "Oxygen Requirement per Person (kg/day)",

            min_value=0.1,

            max_value=5.0,

            value=0.84,

            step=0.01

        )

    st.divider()

    # --------------------------------------------
    # RESOURCE CALCULATIONS
    # --------------------------------------------

    total_water = (
        crew * duration * water_per_person
    )

    total_oxygen = (
        crew * duration * oxygen_per_person
    )


# ------------------------------------------
# ADDITIONAL MISSION RESOURCE INPUTS
# ------------------------------------------

    col_food, col_power, col_reserve = st.columns(3)

    with col_food:
            food_per_person = st.number_input(
                "Food per Person (kg/day)",
                min_value=0.5,
                max_value=5.0,
                value=1.8,
                step=0.1,
                key="food_per_person"
            )

    with col_power:
        power_per_person = st.number_input(
            "Power per Person (kWh/day)",
            min_value=0.5,
            max_value=50.0,
            value=5.0,
            step=0.5,
            key="power_per_person"
        )

    with col_reserve:
        reserve_percent = st.number_input(
            "Emergency Reserve (%)",
            min_value=0,
            max_value=100,
            value=20,
            step=5,
            key="reserve_percent"
        )


    # Base resource calculations
    total_food = crew * duration * food_per_person
    total_power = crew * duration * power_per_person

    # Emergency reserve
    reserve_factor = 1 + reserve_percent / 100

    water_with_reserve = total_water * reserve_factor
    oxygen_with_reserve = total_oxygen * reserve_factor
    food_with_reserve = total_food * reserve_factor
    power_with_reserve = total_power * reserve_factor


    # --------------------------------------------
    # DISPLAY RESULTS
    # --------------------------------------------

    st.markdown(
        "### Mission Resource Requirements"
    )

    a, b = st.columns(2)

    a.metric(
        "Total Water Requirement",
        f"{water_with_reserve:.1f} litres"
    )

    b.metric(
        "Total Oxygen Requirement",
        f"{oxygen_with_reserve:.2f} kg"
    )

    st.divider()

    # --------------------------------------------
    # RESOURCE DISTRIBUTION
    # --------------------------------------------

   
   
if page == "Survival Planner":

    resource_data = pd.DataFrame({
        "Resource": [
            "Water",
            "Oxygen",
            "Food",
            "Electrical Power"
        ],
        "Total Requirement": [
            water_with_reserve,
            oxygen_with_reserve,
            food_with_reserve,
            power_with_reserve
        ]
    })

    resource_fig = go.Figure(
        data=[
            go.Bar(
                x=resource_data["Resource"],
                y=resource_data["Total Requirement"],
                marker_color=[
                    "#35B9FF",
                    "#FF8B5A",
                    "#42E878",
                    "#FFD43B"
                ],
                text=[
                    f"{water_with_reserve:.1f} L",
                    f"{oxygen_with_reserve:.2f} kg",
                    f"{food_with_reserve:.1f} kg",
                    f"{power_with_reserve:.1f} kWh"
                ],
                textposition="auto"
            )
        ]
    )

    resource_fig.update_layout(
        title="Mission Resource Estimation",
        template="plotly_dark",
        yaxis_title="Total Resource Requirement",
        xaxis_title="Resource",
        height=450
    )

    st.plotly_chart(
        resource_fig,
        use_container_width=True,
        key="survival_resource_chart"
    )

    st.info(
        "These are simplified estimates. Actual mission "
        "requirements depend on recycling systems, habitat "
        "technology, crew activity and mission design."
    )


elif page == "Scientific Data":

    st.markdown("## 🧪 MARS SCIENTIFIC DATA")
    st.write(
        "Scientific reference data for the Jezero Crater exploration region."
    )

    st.divider()

    # KEY MISSION DATA
    col1, col2, col3 = st.columns(3)

    col1.metric("Mapped Landmarks", len(LANDMARKS))
    col2.metric("Reference Latitude", f"{JEZERO_LAT:.4f}° N")
    col3.metric("Reference Longitude", f"{JEZERO_LON:.4f}° E")

    st.divider()

    # LANDMARK COORDINATE DATABASE
    st.markdown("### 📍 Exploration Landmark Database")

    scientific_data = pd.DataFrame([
        {
            "Location": name,
            "Latitude (°N)": coordinates[0],
            "Longitude (°E)": coordinates[1]
        }
        for name, coordinates in LANDMARKS.items()
    ])

    st.dataframe(
        scientific_data,
        use_container_width=True,
        hide_index=True
    )

    # COORDINATE COVERAGE
    st.markdown("### 🛰️ Exploration Coordinate Coverage")

    col1, col2 = st.columns(2)

    col1.metric(
        "Northernmost Latitude",
        f"{scientific_data['Latitude (°N)'].max():.4f}°"
    )

    col2.metric(
        "Southernmost Latitude",
        f"{scientific_data['Latitude (°N)'].min():.4f}°"
    )

    st.divider()

    st.markdown("### 🔬 Scientific Data Sources")

    st.write("""
    - Mars MOLA elevation and colour-relief terrain data
    - NASA CTX high-resolution imagery
    - Jezero Crater exploration landmark coordinates
    - Uploaded GeoTIFF terrain elevation datasets
    """)

    st.info(
        "This dashboard currently displays configured landmark "
        "coordinates. Elevation, slope and environmental measurements "
        "will be calculated only when the corresponding source data "
        "is loaded."
    )

    
# ============================================================
# MARS-NAV: GEOTIFF SCIENTIFIC TERRAIN ANALYSIS
# ============================================================

st.header("🪐 Mars GeoTIFF Scientific Analysis")

uploaded_dem = st.file_uploader(
    "Upload a Mars elevation GeoTIFF",
    type=["tif", "tiff"],
    key="mars_scientific_dem"
)

if uploaded_dem is not None:

    try:
        with rasterio.open(uploaded_dem) as dataset:

            # ------------------------------------------------
            # 1. GEO TIFF VALIDATION
            # ------------------------------------------------

            if dataset.count < 1:
                st.error("GeoTIFF contains no raster bands.")
                st.stop()

            original_width = dataset.width
            original_height = dataset.height

            dem_crs = dataset.crs
            original_transform = dataset.transform

            # ------------------------------------------------
            # 2. MEMORY OPTIMIZATION
            # ------------------------------------------------

            max_dimension = 1200

            scale = min(
                1.0,
                max_dimension / max(original_height, original_width)
            )

            out_height = max(1, int(original_height * scale))
            out_width = max(1, int(original_width * scale))

            # ------------------------------------------------
            # 3. ELEVATION EXTRACTION
            # ------------------------------------------------

            
            band = dataset.read(
                1,
                out_shape=(out_height, out_width),
                masked=True,
                out_dtype="float32",
                resampling=Resampling.bilinear
            )

            # Store the processed elevation data

            elevation = np.asarray(
                band.filled(np.nan),
                dtype=np.float32
            )

            saved_dem = {
    "elevation": elevation
    }

            # Preserve adjusted geographic transformation

            output_transform = (
                original_transform
                * Affine.scale(
                    original_width / out_width,
                    original_height / out_height
                )
            )

        # ----------------------------------------------------
        # 4. VALIDATE ELEVATION ARRAY
        # ----------------------------------------------------

        valid_mask = np.isfinite(elevation)
        valid_values = elevation[valid_mask]

        if valid_values.size == 0:
            st.error("No valid elevation pixels found.")
            st.stop()

        # ----------------------------------------------------
        # 5. SCIENTIFIC ELEVATION STATISTICS
        # ----------------------------------------------------

        min_elevation = float(np.min(valid_values))
        max_elevation = float(np.max(valid_values))
        mean_elevation = float(np.mean(valid_values))
        median_elevation = float(np.median(valid_values))
        std_elevation = float(np.std(valid_values))

        elevation_range = max_elevation - min_elevation

        st.subheader("📊 Elevation Statistics")

        col1, col2, col3 = st.columns(3)

        col1.metric("Minimum", f"{min_elevation:.2f}")
        col2.metric("Maximum", f"{max_elevation:.2f}")
        col3.metric("Mean", f"{mean_elevation:.2f}")

        col4, col5, col6 = st.columns(3)

        col4.metric("Median", f"{median_elevation:.2f}")
        col5.metric("Std. Deviation", f"{std_elevation:.2f}")
        col6.metric("Elevation Range", f"{elevation_range:.2f}")

        st.caption(
            "Elevation values use the source GeoTIFF's vertical units. "
            "No units are assumed."
        )

        # ----------------------------------------------------
        # 6. TERRAIN SLOPE ANALYSIS
        # ----------------------------------------------------

        st.subheader("⛰️ Terrain Slope Analysis")

        if dem_crs is not None and dem_crs.is_projected:

            x_resolution = abs(output_transform.a)
            y_resolution = abs(output_transform.e)

            if x_resolution > 0 and y_resolution > 0:

                safe_elevation = np.where(
                    valid_mask,
                    elevation,
                    mean_elevation
                )

                dz_dy, dz_dx = np.gradient(
                    safe_elevation,
                    y_resolution,
                    x_resolution
                )

                slope_degrees = np.degrees(
                    np.arctan(
                        np.sqrt(dz_dx**2 + dz_dy**2)
                    )
                )

                slope_valid = slope_degrees[valid_mask]

                mean_slope = float(np.mean(slope_valid))
                max_slope = float(np.max(slope_valid))

                st.metric(
                    "Mean Terrain Slope",
                    f"{mean_slope:.2f}°"
                )

                st.metric(
                    "Maximum Terrain Slope",
                    f"{max_slope:.2f}°"
                )

                # Rover terrain classification

                safe_terrain = np.sum(slope_valid <= 10)
                moderate_terrain = np.sum(
                    (slope_valid > 10) & (slope_valid <= 25)
                )
                steep_terrain = np.sum(slope_valid > 25)

                total = len(slope_valid)

                st.write("Terrain Classification")

                st.write(
                    f"Safe terrain (≤10°): "
                    f"{safe_terrain / total * 100:.2f}%"
                )

                st.write(
                    f"Moderate terrain (10°–25°): "
                    f"{moderate_terrain / total * 100:.2f}%"
                )

                st.write(
                    f"Steep terrain (>25°): "
                    f"{steep_terrain / total * 100:.2f}%"
                )

                # ------------------------------------------------
                # 7. HILLSHADE GENERATION
                # ------------------------------------------------

                slope_rad = np.arctan(
                    np.sqrt(dz_dx**2 + dz_dy**2)
                )

                aspect = np.arctan2(-dz_dy, dz_dx)

                azimuth = np.radians(315)
                altitude = np.radians(45)

                hillshade = (
                    np.sin(altitude) * np.cos(slope_rad)
                    + np.cos(altitude) * np.sin(slope_rad)
                    * np.cos(azimuth - aspect)
                )

                hillshade = np.clip(
                    hillshade * 255,
                    0,
                    255
                )

                hillshade[~valid_mask] = np.nan

                st.subheader("🌄 Mars Terrain Hillshade")

                st.image(
                    hillshade.astype(np.uint8),
                    caption="Terrain illumination model",
                    clamp=True,
                    use_container_width=True
                )

                # ------------------------------------------------
                # 8. SLOPE MAP
                # ------------------------------------------------

                st.subheader("Terrain Slope Map")

                slope_display = slope_degrees.copy()
                slope_display[~valid_mask] = np.nan

                st.plotly_chart(
                    go.Figure(
                        go.Heatmap(
                            z=slope_display,
                            colorscale="Turbo",
                            colorbar=dict(
                                title="Slope (degrees)"
                            )
                        )
                    ).update_layout(
                        title="Mars Terrain Slope Distribution",
                        height=550
                    ),
                    use_container_width=True
                )

            else:
                st.warning(
                    "Invalid GeoTIFF pixel resolution. "
                    "Slope analysis skipped."
                )

        else:
            st.warning(
                "The GeoTIFF has no projected CRS. "
                "Slope calculations require appropriate "
                "horizontal distance units. "
                "Elevation analysis is still available."
            )

        # ----------------------------------------------------
        # 9. 2D ELEVATION MAP
        # ----------------------------------------------------

        st.subheader("🗺️ Mars Elevation Map")

        elevation_display = elevation.copy()
        elevation_display[~valid_mask] = np.nan

        st.plotly_chart(
            go.Figure(
                go.Heatmap(
                    z=elevation_display,
                    colorscale="Earth",
                    colorbar=dict(title="Elevation")
                )
            ).update_layout(
                title="Mars Digital Elevation Model",
                height=550
            ),
            use_container_width=True
        )

        # ----------------------------------------------------
        # 10. 3D TERRAIN MODEL
        # ----------------------------------------------------

        st.subheader("🌋 3D Mars Terrain Model")

        # Additional reduction for smoother 3D rendering

        terrain_stride = max(
            1,
            int(max(elevation.shape) / 250)
        )

        terrain_3d = elevation[::terrain_stride, ::terrain_stride]

        st.plotly_chart(
            go.Figure(
                go.Surface(
                    z=terrain_3d,
                    colorscale="Earth",
                    colorbar=dict(title="Elevation"),
                    connectgaps=False
                )
            ).update_layout(
                title="3D Mars Digital Terrain",
                height=650,
                scene=dict(
                    xaxis_title="Raster Column",
                    yaxis_title="Raster Row",
                    zaxis_title="Elevation"
                )
            ),
            use_container_width=True
        )

        # ----------------------------------------------------
        # 11. GEO TIFF METADATA
        # ----------------------------------------------------

        st.subheader("🛰️ GeoTIFF Metadata")

        metadata_col1, metadata_col2 = st.columns(2)

        metadata_col1.write(
            f"Original Width: {original_width}"
        )

        metadata_col1.write(
            f"Original Height: {original_height}"
        )

        metadata_col1.write(
            f"Raster Bands: {dataset.count if False else 1}"
        )

        metadata_col2.write(
            f"CRS: {dem_crs}"
        )

        metadata_col2.write(
            f"Processed Width: {out_width}"
        )

        metadata_col2.write(
            f"Processed Height: {out_height}"
        )

        st.success(
            "Mars GeoTIFF scientific analysis completed."
        )

    except Exception as error:
        st.error(
            f"GeoTIFF processing error: {error}"
        )

else:
    st.info(
        "Upload a Mars elevation GeoTIFF to begin "
        "scientific terrain analysis."
    )


    