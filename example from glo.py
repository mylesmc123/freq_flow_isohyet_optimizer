import math
import os

import altair as alt
import folium
import geopandas as gpd
import pandas as pd
from branca.element import Template, MacroElement
from shapely.geometry import Polygon

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ USER-DEFINED VARIABLES AND INPUT ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# Variable 'base_dir' is the project home directory where this script resides, defined by user
base_dir = r'C:\Projects\design-storm'

# Variable 'basin_shapefile_dir' is the directory where the HEC-HMS subbasin delineation .shp exists, defined by user
basin_shapefile_dir = base_dir + os.sep + r'GisData\BasinShapefiles'

# Variable 'subbasin_shapefile' is the name of the shapefile with the HEC-HMS subbasin delineations, defined by user
subbasin_shapefile = 'Trinity_CWMS_Subbasins.shp'

# Variable 'subbasin_col_name' represents the name of the column in the 'subbasin_shapefile' specified above where the
# subbasin names are stored. This needs editing by the user to exactly match the shapefile column name.
subbasin_col_name = 'Name'

# Variable 'areas' is a list of areas in square miles in the ARF relationships, defined by user
areas = [10, 25, 50, 100, 200, 300, 400,
         600, 800, 1000, 1500, 2000, 2667, 3500,
         4000, 4500, 5000, 6000, 6500, 7000, 8000,
         9000, 10000]

# Variable 'dar_values' contains the reduction factors corresponded to the variable 'areas', defined by user
dar_values = [1.000, 0.977, 0.960, 0.940, 0.902, 0.875, 0.855,
              0.834, 0.818, 0.804, 0.775, 0.752, 0.726, 0.699,
              0.685, 0.672, 0.658, 0.637, 0.626, 0.617, 0.599,
              0.581, 0.564]

# Variable 'ellipse_ratio' is the ratio between major and minor axis of the elliptical storm, defined by user
ellipse_ratio = '2.5'

# Variable 'basin_name' is the name of the study basin, used to label output, defined by user
basin_name = "Trinity"

# Variable 'output_name' is a unique identifier used to name this script's output files, defined by user
output_name = 'dar48hr'

# Variable 'output_dir' is the directory where output from this script will be located
output_dir = base_dir + os.sep + r'ScriptResults\1_Prepare_DAR_Ellipses'

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ CREATE DEPTH AREA REDUCTION POLYGON ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# This section, and beyond, includes variables that should generally not be changed. Thus no special editing is needed.

# Variable 'epsg' defines a projection for elliptical area calculations
epsg = 5070  # USA_Contiguous_Albers_Equal_Area_Conic_USGS_version in meters

# Variable 'epsg_out' defines the output reference system for the DAR ellipses output
epsg_out = 5070  # USA_Contiguous_Albers_Equal_Area_Conic_USGS_version in meters

# Variable 'theta0' is the initial orientation of the design storm, 0 means the DAR ellipses are positioned horizontally
theta0 = 0

# Set some unit conversion factors
mi2_to_m2 = 1609.34 * 1609.34
deg_to_rad = math.pi / 180

# Read in the shapefile with HEC-HMS subbasin delineations
gdf_basin = gpd.read_file(basin_shapefile_dir + os.sep + subbasin_shapefile)
gdf_basin = gdf_basin[[subbasin_col_name, 'geometry']]
gdf_basin = gdf_basin.to_crs(epsg=str(epsg))  # convert to defined projection

gdf_basin['Precip'] = 0

basin_dissolved = gdf_basin.dissolve(by='Precip')

basin_centroid = basin_dissolved['geometry'].centroid

storm_center_x = basin_centroid.x
storm_center_y = basin_centroid.y

# Create polygon features (ellipse) based on the 'areas' in the ARF relationship, the ellipticity, and orientation
print("\n***** Building DAR polygon ellipses...*****\n")
polygon_objects = []
for area in range(len(areas)):
    # Calculate the major and minor axis
    b = math.sqrt(areas[area] * mi2_to_m2 / (math.pi * float(ellipse_ratio)))
    a = (float(ellipse_ratio) * b)
    # Calculate areal reduction factor within the current ellipse
    arf = float(dar_values[area])
    # Calculate sin and cos arfs based on orientation
    beta = float(theta0) * deg_to_rad
    sinBeta = math.sin(beta)
    cosBeta = math.cos(beta)
    # Use 360 points per ellipse to delineate its shape
    steps = 360
    i = 0
    polygon = []
    while i < 360:
        # Calculate the coordinate for each point
        alpha = i * deg_to_rad
        sinAlpha = math.sin(alpha)
        cosAlpha = math.cos(alpha)
        X = float(storm_center_x) + (float(a) * cosAlpha * cosBeta - float(b) * sinAlpha * sinBeta)
        Y = float(storm_center_y) + (float(a) * cosAlpha * sinBeta + float(b) * sinAlpha * cosBeta)
        i += (360 / steps)
        polygon += ((X, Y),)
    polygon_object = Polygon(polygon)
    polygon_objects += (polygon_object,)

# Create a geopandas dataframe with elliptical geometries, areas, and dar values
df_ellipse = pd.DataFrame({'area': areas,
                          'dar': dar_values})
gdf_ellipse = gpd.GeoDataFrame(df_ellipse, geometry=polygon_objects).set_crs(epsg=str(epsg))

gdf_ellipse = gdf_ellipse.iloc[::-1]

gdf_ellipse = gdf_ellipse.to_crs(epsg_out)

gdf_ellipse.to_file(output_dir + os.sep + '1a_Ellipses_Vector_' + output_name + '.shp')

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ CREATE A DAR ELLIPSE MAP (HTML) ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

print("\n***** Creating a map... *****\n")
basin_dissolved['geometry'] = basin_dissolved['geometry'].buffer(0.1).simplify(60)
basin_dissolved = basin_dissolved.to_crs(epsg='4326')
basin_centroid = basin_centroid.to_crs(epsg='4326')
gdf_ellipse = gdf_ellipse.to_crs(epsg='4326')

# Identify center coordinates for map
bounds = basin_dissolved.total_bounds
lon_low = bounds[2]
lon_high = bounds[0]
lat_low = bounds[1]
lat_high = bounds[3]

map_center_lon = (lon_low + lon_high) / 2.0
map_center_lat = (lat_low + lat_high) / 2.0

# Create a folium map object
mapa = folium.Map([map_center_lat, map_center_lon],
                  zoom_start=8,
                  tiles='cartodbpositron',
                  control_scale=True
                  )

# Add basin layer to map
style = {'color': 'gray', 'fillColor': 'gray', 'fillOpacity': 0.2}
folium.GeoJson(data=basin_dissolved,
               name=basin_name + " Basin",
               tooltip=basin_name + " Basin",
               style_function=lambda x: style
               ).add_to(mapa)

# Create a DAR line plot to be visualized as a popup
df_plot = gdf_ellipse[['area', 'dar']]
line_plot = alt.Chart(df_plot, title='Depth-Area-Reduction Factor: ' + basin_name + " Basin").mark_line().encode(
    x=alt.X('area', axis=alt.Axis(title='Area (sqmi)')),
    y=alt.Y('dar', axis=alt.Axis(title='DAR Factor')),
).properties(width=400, height=300)

# Add plot to popup
popup = folium.Popup(max_width=1000)
folium.VegaLite(line_plot, width=450, height=350).add_to(popup)

# Create a centroid icon and add the popup to it
feature_group = folium.FeatureGroup(name='Basin Centroid', show=True)
folium.Marker(
    location=[basin_centroid.y, basin_centroid.x],
    popup=popup,
    tooltip='Basin Centroid: click for DAR plot...',
    icon=folium.Icon(icon="cloud", color="black")
).add_to(feature_group)
feature_group.add_to(mapa)

# Add choropleth layer of dar ellipses to map
folium.Choropleth(
    geo_data=gdf_ellipse,
    name="Depth Area Reduction Ellipses",
    data=gdf_ellipse,
    columns=["area", "dar"],
    key_on="feature.properties.area",
    fill_color="BuPu",
    fill_opacity=0.8,
    line_opacity=0.1,
    legend_name="Depth Area Reduction Factor",
    smooth_factor=0,
    Highlight=True,
    line_color="#0000",
    show=True,
    overlay=True
).add_to(mapa)

# Add hover functionality
style_function = lambda x: {'fillColor': '#ffffff',
                           'color': '#000000',
                           'fillOpacity': 0.001,
                           'weight': 0.1}
highlight_function = lambda x: {'fillColor': '#000000',
                               'color': '#000000',
                               'fillOpacity': 0.50,
                               'weight': 0.1}
hover = folium.features.GeoJson(
    data=gdf_ellipse,
    name="Ellipses Overlay",
    style_function=style_function,
    control=True,
    highlight_function=highlight_function,
    tooltip=folium.features.GeoJsonTooltip(
        fields=['area', 'dar'],
        aliases=['Area [mi2]', 'DAR Factor'],
        style="background-color: white; color: #333333; font-family: arial; font-size: 12px; padding: 10px;"
    )
)
mapa.add_child(hover)
mapa.keep_in_front(hover, feature_group)

# Add various background tile layers
folium.TileLayer('cartodbdark_matter', name="cartodb dark", control=True).add_to(mapa)
folium.TileLayer('openstreetmap', name="open street map", control=True, opacity=0.4).add_to(mapa)
folium.TileLayer('stamenterrain', name="stamen terrain", control=True, opacity=0.6).add_to(mapa)
folium.TileLayer('stamenwatercolor', name="stamen watercolor", control=True, opacity=0.6).add_to(mapa)
folium.TileLayer('stamentoner', name="stamen toner", control=True, opacity=0.6).add_to(mapa)


# Add a layer controller
folium.LayerControl(collapsed=True).add_to(mapa)

# Create a legend using HTML and JavaScript
template = """
{% macro html(this, kwargs) %}

<!doctype html>
<html lang="en">
<body>

 
<div id='maplegend' class='maplegend' 
    style='position: absolute; z-index:9999; border:2px solid grey; background-color:rgba(255, 255, 255, 0.8);
     border-radius:6px; padding: 10px; font-size:18px; left: 6px; bottom: 40px;'>
     
<div class='legend-title'>Legend</div>
<div class='legend-scale'>
  <ul class='legend-labels'>
    <li><span style='background:gray;opacity:0.5;'></span>Basin</li>
    <li><span style='background:purple;opacity:0.7;'></span>DAR Ellipses</li>
    

  </ul>
</div>
</div>
 
</body>
</html>

<style type='text/css'>
  .maplegend .legend-title {
    text-align: left;
    margin-bottom: 5px;
    font-weight: bold;
    font-size: 90%;
    }
  .maplegend .legend-scale ul {
    margin: 0;
    margin-bottom: 5px;
    padding: 0;
    float: left;
    list-style: none;
    }
  .maplegend .legend-scale ul li {
    font-size: 80%;
    list-style: none;
    margin-left: 0;
    line-height: 18px;
    margin-bottom: 2px;
    }
  .maplegend ul.legend-labels li span {
    display: block;
    float: left;
    height: 16px;
    width: 30px;
    margin-right: 5px;
    margin-left: 0;
    border: 1px solid #999;
    }
  .maplegend .legend-source {
    font-size: 80%;
    color: #777;
    clear: both;
    }
  .maplegend a {
    color: #777;
    }
</style>
{% endmacro %}"""

macro = MacroElement()
macro._template = Template(template)
mapa.get_root().add_child(macro)

# Save map to html
mapa.save(output_dir + os.sep + '1b_Ellipses_Map_' + output_name + '.html')

print("****************************************************************")
print("Depth-Area-Reduction Ellipses have been created")
print("The script results have been saved in: " + output_dir)
print("Run Complete.")
print("****************************************************************")