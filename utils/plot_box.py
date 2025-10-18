import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats

def boxplot(*data_groups):
    """
    Creates a styled boxplot for the given data groups and calculates p-values
    using the Kruskal-Wallis test.

    Parameters:
    - data_groups: variable number of array-like. Each represents a group.
    """

    # Perform Kruskal-Wallis test
    h_value, p_value_kruskal = stats.kruskal(*data_groups)
    print(f"Kruskal-Wallis p-value: {p_value_kruskal:.2e}")

    # Create the boxplot
    fig, ax = plt.subplots()
    bp = ax.boxplot(data_groups, patch_artist=True)

    # Customizing the color and style to match the uploaded image
    colors = ['#D7191C', '#2C7BB6', '#FDAE61', '#ABDDA4']  # Colors for the groups
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
    
    # Customizing the whiskers, fliers, caps, and median lines
    for whisker in bp['whiskers']:
        whisker.set(color='#000000', linewidth=1.2)
    for cap in bp['caps']:
        cap.set(color='#000000', linewidth=1.2)
    for median in bp['medians']:
        median.set(color='#FFFFFF', linewidth=1.2)
    for flier in bp['fliers']:
        flier.set(marker='o', color='#000000', alpha=0.5)

    # Setting the y-axis and x-axis labels
    ax.set_ylabel('Value')
    ax.set_xticklabels([f'Group {i}' for i in range(1, len(data_groups) + 1)])

    # Removing the top and right spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Adjusting the limits if necessary
    ax.set_ylim(0, max(max(np.concatenate(data_groups)), 1.0) * 1.2)

    # Show the plot
    plt.show()

# Example usage of the function with sample data
np.random.seed(10)
sample_data_group1 = np.random.normal(0.5, 0.1, 100)
sample_data_group2 = np.random.normal(0.5, 0.1, 100)
sample_data_group3 = np.random.normal(0.5, 0.1, 100)
sample_data_group4 = np.random.normal(0.5, 0.1, 100)
create_boxplot_and_calculate_pvalues(sample_data_group1, sample_data_group2, sample_data_group3, sample_data_group4)