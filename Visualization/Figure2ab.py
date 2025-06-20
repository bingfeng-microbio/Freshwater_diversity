import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from matplotlib import rcParams
import matplotlib as mpl

# Set global font to Arial
rcParams['font.family'] = 'Arial'

# Use the default RdBu reversed colormap
cmap = plt.cm.RdBu_r

def plot_maps(input_folder, output_folder, lower_percentile=5, upper_percentile=95):
    """
    Plot spatial maps of Shannon diversity with percentile-based color scale truncation.

    Parameters:
        input_folder (str): Path to the input folder containing CSV files.
        output_folder (str): Path to the output folder for saving the maps.
        lower_percentile (float): Lower percentile cutoff for color normalization (default is 5%).
        upper_percentile (float): Upper percentile cutoff for color normalization (default is 95%).
    """
    # Ensure the output directory exists
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Iterate over all CSV files in the input folder
    for file_name in os.listdir(input_folder):
        if file_name.endswith('.csv'):
            file_path = os.path.join(input_folder, file_name)
            print(f"Processing file: {file_path}")

            # Read the data in chunks and filter numeric values
            chunks = pd.read_csv(file_path, header=None, chunksize=10000)
            data = pd.concat(chunk.apply(pd.to_numeric, errors='coerce').dropna() for chunk in chunks)

            # Extract longitude, latitude, and Shannon diversity values
            lon = data.iloc[:, 0].to_numpy()
            lat = data.iloc[:, 1].to_numpy()
            shannon_values = data.iloc[:, 2].to_numpy()

            # Compute percentile-based bounds for color normalization
            lower_bound = np.percentile(shannon_values, lower_percentile)
            upper_bound = np.percentile(shannon_values, upper_percentile)

            # Set plotting style
            plt.style.use('ggplot')
            plt.figure(figsize=(14, 9))

            # Initialize map projection
            proj = ccrs.Robinson()
            ax = plt.axes(projection=proj)

            # Add geographic features
            ax.add_feature(cfeature.LAND, edgecolor='black', facecolor='lightgrey')  # Land
            ax.add_feature(cfeature.COASTLINE, linewidth=1.5)  # Coastline
            ax.add_feature(cfeature.OCEAN, facecolor='white')  # Ocean

            # Normalize color scale based on percentile range
            norm = mpl.colors.Normalize(vmin=lower_bound, vmax=upper_bound)

            # Plot the data as a scatter plot
            sc = ax.scatter(lon, lat, c=shannon_values, cmap=cmap, norm=norm,
                            alpha=0.8, s=1, linewidths=0, marker='s',
                            transform=ccrs.PlateCarree(), zorder=1)

            # Add colorbar
            cbar = plt.colorbar(sc, ax=ax, orientation='vertical', pad=0.05, aspect=30)
            cbar.set_label('Shannon Diversity', fontsize=10)
            cbar.ax.tick_params(labelsize=8)

            # Add value ticks to the colorbar
            cbar_ticks = np.linspace(lower_bound, upper_bound, 5)
            cbar.set_ticks(cbar_ticks)
            cbar.ax.set_yticklabels([f"{tick:.2f}" for tick in cbar_ticks])

            # Set title for the plot
            ax.set_title(f"Shannon Diversity - {file_name}", fontsize=16, weight='bold')

            # Save the map as a high-resolution PNG
            output_path = os.path.join(output_folder, f"{os.path.splitext(file_name)[0]}.png")
            plt.savefig(output_path, format='png', dpi=600)
            plt.close()

            print(f"Map saved to: {output_path}")

# Example usage
input_folder = 'predictions_2023'  # Input folder path
output_folder = 'diversity_map'  # Output folder path
plot_maps(input_folder, output_folder, lower_percentile=5, upper_percentile=95)
