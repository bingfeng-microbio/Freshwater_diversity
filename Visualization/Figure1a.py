# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
import numpy as np
from matplotlib.font_manager import FontProperties

# Read coordinate data from a tab-delimited file, skipping the header row
data = np.loadtxt('map.txt', delimiter='\t', skiprows=1)

# Extract latitude, longitude, and raw size values
latitudes = data[:, 0]
longitudes = data[:, 1]
original_sizes = data[:, 2]

# Clip point sizes to a defined range
min_size = 5
max_size = 40
sizes = np.clip(original_sizes, min_size, max_size)

# Create a Basemap object using the Robinson projection
m = Basemap(projection='robin', resolution='l', lon_0=0)

# Create the plotting window
plt.figure(figsize=(12, 6))

# Draw map boundaries and land areas
m.drawcoastlines(linewidth=0.1)
m.drawcountries(linewidth=0.1)  # Add country borders
m.drawmapboundary(fill_color='none')
m.fillcontinents(color='#DCDDDD')

# Add gridlines for latitude and longitude
parallels = np.arange(-90., 91., 30.)
meridians = np.arange(-180., 181., 60.)
m.drawparallels(parallels, labels=[1, 0, 0, 0], linewidth=0.25)  # Latitude lines
m.drawmeridians(meridians, labels=[0, 0, 0, 1], linewidth=0.25)  # Longitude lines

# Plot the points using scaled sizes
x, y = m(longitudes, latitudes)
m.scatter(x, y, s=sizes * 15, c='#8EB1DC', edgecolors='k', linewidth=0.1, alpha=0.8, zorder=10)

# Create a custom legend to represent point sizes
for size in [20, 100, 500]:
    plt.scatter([], [], s=size, label=f'Point Size: {size}', c='gray', edgecolors='k', alpha=0.8)

# Set the title and legend font properties
font_prop = FontProperties(family='Arial', size=7)
plt.title('Global Coordinate Points Distribution', fontproperties=font_prop)
plt.legend(scatterpoints=1, frameon=False, labelspacing=1, loc='upper left', prop=font_prop)

# Save the figure as a high-resolution PDF
plt.savefig('global_map.pdf', dpi=300, bbox_inches='tight')

# Display the plot
plt.show()