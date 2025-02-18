import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as tri
import pyvista as pv
from matplotlib.colors import Normalize

class FEPlotter:
    @staticmethod
    def plot_triangulated(points, elements, values_list, titles=None, filename=None, value_range=None):
        """
        Plot multiple nodal solutions using triangular elements and save them in a single figure.

        Parameters:
        points (ndarray): (N, 2) array of node coordinates.
        elements (ndarray): (M, 3) array of triangular element connectivity.
        values_list (list of ndarray or single ndarray): List of (N,) arrays, each representing a nodal solution field.
        titles (list of str): Titles for each subplot.
        filename (str): If provided, saves the figure to the specified filename.
        value_range (tuple): (vmin, vmax) to fix the color scale. If None, automatically computed.
        """
        points = np.asarray(points, dtype=float)
        elements = np.asarray(elements, dtype=int)

        # Ensure values_list is a list of NumPy arrays
        if not isinstance(values_list, list):
            values_list = [values_list]
        values_list = [np.asarray(values, dtype=float).flatten() for values in values_list]

        # Compute global min/max for consistent color scaling
        if value_range is None:
            vmin, vmax = np.min(np.concatenate(values_list)), np.max(np.concatenate(values_list))
        else:
            vmin, vmax = value_range  # Use user-defined range
        norm = Normalize(vmin=vmin, vmax=vmax)

        num_plots = len(values_list)
        cols = min(2, num_plots)  # Up to 2 plots per row
        rows = (num_plots + cols - 1) // cols  # Compute number of rows dynamically

        fig, axes = plt.subplots(rows, cols, figsize=(cols * 5, rows * 5), constrained_layout=True)
        axes = np.array(axes).reshape(-1)  # Flatten for easy indexing

        if titles is None:
            titles = [f"FE Solution {i}" for i in range(num_plots)]

        contour_plots = []  # Store contour plots for colorbar reference

        for i, (values, ax) in enumerate(zip(values_list, axes)):
            if len(values) != len(points):
                raise ValueError(f"Mismatch: {len(points)} nodes but {len(values)} values.")

            triang = tri.Triangulation(points[:, 0], points[:, 1], elements)
            contour = ax.tricontourf(triang, values, levels=1000, cmap="jet",vmin=vmin, vmax=vmax,norm=norm)
            # ax.triplot(triang, 'k-', alpha=0.3)  # Optional: mesh visualization
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_frame_on(False)
            ax.set_aspect('equal')  # Maintain correct aspect ratio
            contour_plots.append(contour)

        # Hide empty subplots (if num_plots < rows*cols)
        for i in range(num_plots, len(axes)):
            fig.delaxes(axes[i])

        # Add a single colorbar spanning all subplots
        cbar_ax = fig.add_axes([1.05, 0.2, 0.02, 0.6])
        cbar = fig.colorbar(contour_plots[0], ax=ax, orientation='vertical', fraction=0.05, pad=0.04,cax=cbar_ax)
        cbar.ax.tick_params(labelsize=18)
        cbar.set_ticks(np.linspace(vmin, vmax, num=5))

        if filename:
            plt.savefig(filename, dpi=300, bbox_inches="tight")
            print(f"Saved figure as: {filename}")

        plt.show()

    @staticmethod
    def plot_triangulated_error(points, elements, values_list, titles=None, filename=None):
        """
        Plot multiple nodal solutions using triangular elements and save them in a single figure.

        Parameters:
        points (ndarray): (N, 2) array of node coordinates.
        elements (ndarray): (M, 3) array of triangular element connectivity.
        values_list (list of ndarray or single ndarray): List of (N,) arrays, each representing a nodal solution field.
        titles (list of str): Titles for each subplot.
        filename (str): If provided, saves the figure to the specified filename.
        value_range (tuple): (vmin, vmax) to fix the color scale. If None, automatically computed.
        """
        points = np.asarray(points, dtype=float)
        elements = np.asarray(elements, dtype=int)

        # Ensure values_list is a list
        if not isinstance(values_list, list):
            values_list = [values_list]
        values_list = [np.asarray(values, dtype=float).flatten() for values in values_list]

        num_plots = len(values_list)
        cols = min(2, num_plots)  # Up to 2 plots per row
        rows = (num_plots + cols - 1) // cols  # Compute number of rows dynamically

        fig, axes = plt.subplots(rows, cols, figsize=(cols * 5, rows * 5), constrained_layout=True)
        axes = np.array(axes).reshape(-1)  # Flatten for easy indexing

        if titles is None:
            titles = [f"FE Solution {i}" for i in range(num_plots)]

        for i, (values, ax, title) in enumerate(zip(values_list, axes, titles)):
            if len(values) != len(points):
                raise ValueError(f"Mismatch: {len(points)} nodes but {len(values)} values.")
            vmax = np.max(values)
            vmin = np.minimum(0.0, np.min(values))
            norm = Normalize(vmin=vmin, vmax=vmax)
            triang = tri.Triangulation(points[:, 0], points[:, 1], elements)
            contour = ax.tricontourf(triang, values, levels=1000, cmap="jet", norm=norm)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_frame_on(False)
            ax.set_aspect('equal')  # Maintain correct aspect ratio

            # Add a colorbar for each subplot
            
            cbar = fig.colorbar(contour, ax=ax, orientation='vertical', fraction=0.05, pad=0.04)
            cbar.ax.tick_params(labelsize=18)
            cbar.set_ticks(np.linspace(vmin, vmax, num=5))

        # Hide empty subplots (if num_plots < rows * cols)
        for i in range(num_plots, len(axes)):
            fig.delaxes(axes[i])

        if filename:
            plt.savefig(filename, dpi=300, bbox_inches="tight")
            print(f"Saved figure as: {filename}")

        plt.show()


    def plot_mesh(points, elements, filename=None):
        """
        Plot the triangular mesh using given node coordinates and element connectivity. 

        Parameters:
        points (ndarray): (N, 2) array of node coordinates.
        elements (ndarray): (M, 3) array of triangular element connectivity.
        filename (str): If provided, saves the figure to the specified filename.
        """
        points = np.asarray(points, dtype=float)
        elements = np.asarray(elements, dtype=int)  

        fig, ax = plt.subplots(figsize=(6, 6))
        triang = tri.Triangulation(points[:, 0], points[:, 1], elements)

        ax.triplot(triang, 'k-',linewidth=0.5, alpha=1.0)  # Plot mesh edges
        # ax.set_frame_on(False)
        ax.set_aspect('equal')  # Maintain correct aspect ratio

        if filename:
            plt.savefig(filename, dpi=300, bbox_inches="tight")
            print(f"Saved mesh plot as: {filename}")

        plt.show()
