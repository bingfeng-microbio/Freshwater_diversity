import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from matplotlib import rcParams
import matplotlib as mpl
from matplotlib.colors import ListedColormap

# Set global font to Arial
rcParams['font.family'] = 'Arial'

# Define category color mapping
categories = ["0", "1", "2", "3", "4"]
category_colors = ["#CCCCCC", "#053061", "#7A90A5", "#C55258", "#6E0220"]  # Color corresponding to each category
cmap = ListedColormap(category_colors)

def plot_maps(input_folder, output_folder):
    """
    Plot categorical data on a global map.

    Parameters:
        input_folder (str): Path to the input folder containing CSV files.
        output_folder (str): Path to the output folder to save the maps.
    """
    # Ensure the output folder exists
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Iterate over all CSV files in the input folder
    for file_name in os.listdir(input_folder):
        if file_name.endswith('.csv'):
            file_path = os.path.join(input_folder, file_name)
            print(f"Processing file: {file_path}")

            # Read the CSV file
            data = pd.read_csv(file_path, header=0, low_memory=False)

            # Verify required columns exist
            if not all(col in data.columns for col in ['Longitude', 'Latitude', 'Hotspot_Count']):
                raise ValueError("CSV file must contain 'Longitude', 'Latitude', and 'Hotspot_Count' columns.")

            # Extract longitude, latitude, and category values
            lon = data['Longitude'].astype(float).to_numpy()
            lat = data['Latitude'].astype(float).to_numpy()
            category_values = data['Hotspot_Count'].astype(str).to_numpy()

            # Map category values to indices
            category_indices = []
            for cat in category_values:
                try:
                    category_indices.append(categories.index(cat))
                except ValueError:
                    category_indices.append(0)  # Default to "Others"

            category_indices = np.array(category_indices)

            # Set plotting style
            plt.style.use('ggplot')
            plt.figure(figsize=(14, 9))

            # Initialize map projection
            proj = ccrs.Robinson()
            ax = plt.axes(projection=proj)

            # Add geographical features
            ax.add_feature(cfeature.LAND, edgecolor='black', facecolor='#F0F0F0')  # Land
            ax.add_feature(cfeature.COASTLINE, linewidth=1.5)  # Coastlines
            ax.add_feature(cfeature.OCEAN, facecolor='white')  # Oceans

            # Plot the categorical data as a scatter plot
            sc = ax.scatter(lon, lat, c=category_indices, cmap=cmap, vmin=0, vmax=len(categories)-1,
                            alpha=0.8, s=1, linewidths=0, marker='s',
                            transform=ccrs.PlateCarree(), zorder=1)

            # Add colorbar
            cbar = plt.colorbar(sc, ax=ax, orientation='vertical', pad=0.05, aspect=30)
            cbar.set_label('Categories', fontsize=10)
            cbar.ax.tick_params(labelsize=8)

            # Set category labels on the colorbar
            cbar.set_ticks(range(len(categories)))
            cbar.ax.set_yticklabels(categories)

            # Add title to the plot
            ax.set_title(f"Categorical Map - {file_name}", fontsize=16, weight='bold')

            # Save the map as high-resolution PNG
            output_path = os.path.join(output_folder, f"{os.path.splitext(file_name)[0]}.png")
            plt.savefig(output_path, format='png', dpi=600)
            plt.close()

            print(f"Map saved to: {output_path}")

# Example usage
input_folder = 'input'  # Input folder path
output_folder = 'output'  # Output folder path
plot_maps(input_folder, output_folder)
