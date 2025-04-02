import numpy as np
from typing import Union, Any
from numpy import floating
import scipy.stats as stats


def norm(vx, vy, vz) -> np.ndarray:
    """ Compute the norm of the vector field.

        Parameters:
        ----------
        vx, vy, vz: np.ndarray. Components of the vector field.

        Returns:
        --------
        norm: np.ndarray. Norm of the vector.
    """
    return np.sqrt(vx ** 2 + vy ** 2 + vz ** 2)


def spearman_correlation_coefficient(v1: np.ndarray, v2: np.ndarray, axis: int = None) -> np.ndarray:
    """
        Compute the Spearman correlation coefficient between two images.

        Parameters:
        ----------
        v1, v2: np.ndarray. Input images.
        axis: Union[int, tuple]. Axis along which to compute the metric (i.e., samples/time axis).

        Returns:
        --------
        spearman_corr: np.ndarray. Spearman correlation coefficient between v1 and v2.
    """

    # Reshape based on axis
    if axis is not None:

        # Extract shape
        shape = v1.shape
        # Move the specified axis to the front and reshape
        v1 = np.moveaxis(v1, axis, 0).reshape(shape[axis], -1)
        v2 = np.moveaxis(v2, axis, 0).reshape(shape[axis], -1)

        # Compute the Spearman correlation coefficient for each sample
        sc = []
        for i in range(shape[axis]):
            sc.append(stats.spearmanr(v1[i], v2[i])[0])

        return np.stack(sc)

    else:
        # Spearman correlation coefficient
        sc, _ = stats.spearmanr(v1.flatten(), v2.flatten())

        return sc


def vertical_poynting_flux(vx: np.ndarray, vy: np.ndarray, vz: np.ndarray,
                           bx: np.ndarray, by: np.ndarray, bz: np.ndarray) -> np.ndarray:
    """ Compute the vertical Poynting flux.

        Parameters:
        ----------
        vx, vy, vz: np.ndarray. Components of the velocity vector field.
        bx, by, bz: np.ndarray. Components of the magnetic field vector.

        Returns:
        --------
        metric: np.ndarray. Vertical Poynting flux between v1 and v2.
    """
    return 0.25/np.pi * ((bx ** 2 + by ** 2)*vz - (vx*bx + vy*by)*bz)


class Metrics:
    """ Class for computing various metrics between two vector fields."""

    def __init__(self, v1x: np.ndarray, v2x: np.ndarray,
                 v1y: np.ndarray = None, v2y: np.ndarray = None,
                 v1z: np.ndarray = None, v2z: np.ndarray = None) -> None:
        """ Initialize the Metrics class with two vector fields.
            Note that v1 is the reference field.

            Parameters:
            ----------
            v1x, v2x: np.ndarray. x component of the vector fields.
            v1y, v2y: np.ndarray, optional. y component of the vector fields. Defaults to None.
            v1z, v2z: np.ndarray, optional. z component of the vector fields. Defaults to None.
        """

        # Assign null values to y and z components if not provided
        self.v1x, self.v2x = v1x, v2x
        self.v1y, self.v2y = np.zeros_like(v1x) if v1y is None else v1y, np.zeros_like(v2x) if v2y is None else v2y
        self.v1z, self.v2z = np.zeros_like(v1x) if v1z is None else v1z, np.zeros_like(v2x) if v2z is None else v2z

    def squared_error(self) -> np.ndarray:
        """ Compute the squared error between two vector fields.

            Returns:
            --------
            metric: np.ndarray. Squared error between v1 and v2.
        """
        return (self.v1x - self.v2x) ** 2 + (self.v1y - self.v2y) ** 2 + (self.v1z - self.v2z) ** 2

    def absolute_error(self) -> np.ndarray:
        """ Compute the absolute error between two vector fields.

            Returns:
            --------
            metric: np.ndarray. Absolute error between v1 and v2.
        """
        return np.sqrt(self.squared_error())

    def absolute_relative_error(self) -> np.ndarray:
        """ Compute the absolute relative error between two vector fields.
            Note that v1 is the reference field.

            Returns:
            --------
            metric: np.ndarray. Absolute relative error between v1 and v2.
        """
        return self.absolute_error() / norm(self.v1x, self.v1y, self.v1z)

    def cosine_similarity(self) -> np.ndarray:
        """ Compute the cosine similarity between two vector fields.

            Returns:
            --------
            metric: np.ndarray. Cosine similarity between v1 and v2.
        """
        # Dot product of the two vectors
        dot_prod = self.v1x * self.v2x + self.v1y * self.v2y + self.v1z * self.v2z
        # Compute norms of the vectors
        norm1 = norm(self.v1x, self.v1y, self.v1z)
        norm2 = norm(self.v2x, self.v2y, self.v2z)

        # Compute cosine similarity index
        return dot_prod / (norm1 * norm2)

    def cosine_similarity_index(self, axis: Union[int, tuple] = None) -> floating[Any]:
        """ Compute the cosine similarity index between two vector fields.

            Parameters:
            ----------
            axis: tuple, optional. Defaults to None. Axis along which to compute the metric (i.e., spatial axes).

            Returns:
            --------
            metric: np.ndarray. Cosine similarity index between v1 and v2.
        """
        return np.mean(self.cosine_similarity(), axis=axis)

    def mean_squared_error(self, axis: Union[int, tuple] = None) -> floating[Any]:
        """ Compute the mean squared error between two vector fields.

            Parameters:
            ----------
            axis: tuple, optional. Defaults to None. Axis along which to compute the metric (i.e., spatial axes).

            Returns:
            --------
            metric: np.ndarray. Mean squared error between v1 and v2.
        """
        return np.mean(self.squared_error(), axis=axis)

    def mean_absolute_error(self, axis: Union[int, tuple] = None) -> floating[Any]:
        """ Compute the mean absolute error between two vector fields.

            Parameters:
            ----------
            axis: tuple, optional. Defaults to None. Axis along which to compute the metric (i.e., spatial axes).

            Returns:
            --------
            metric: np.ndarray. Mean absolute error between v1 and v2.
        """
        return np.mean(self.absolute_error(), axis=axis)

    def mean_absolute_relative_error(self, axis: Union[int, tuple] = None) -> floating[Any]:
        """ Compute the mean absolute relative error between two vector fields.

            Parameters:
            ----------
            axis: tuple, optional. Defaults to None. Axis along which to compute the metric (i.e., spatial axes).

            Returns:
            --------
            metric: np.ndarray. Mean absolute relative error between v1 and v2.
        """
        return np.mean(self.absolute_relative_error(), axis=axis)

    def median_absolute_relative_error(self, axis: Union[int, tuple] = None) -> floating[Any]:
        """ Compute the median absolute relative error between two vector fields.

            Parameters:
            ----------
            axis: tuple, optional. Defaults to None. Axis along which to compute the metric (i.e., spatial axes).

            Returns:
            --------
            metric: np.ndarray. Median absolute relative error between v1 and v2.
        """
        return np.median(self.absolute_relative_error(), axis=axis)

    def normalized_error(self, axis: Union[int, tuple] = None) -> floating[Any]:
        """ Compute the normalized error between two vector fields.

            Parameters:
            ----------
            axis: tuple, optional. Defaults to None. Axis along which to compute the metric (i.e., spatial axes).

            Returns:
            --------
            metric: np.ndarray. Normalized error between v1 and v2.
        """

        # Compute the normalized error based on the definition from Schrijver et al. (2005)
        return np.sum(self.absolute_error(), axis=axis) / np.sum(norm(self.v1x, self.v1y, self.v1z), axis=axis)

    def vector_correlation_coefficient(self, axis: Union[int, tuple] = None) -> floating[Any]:
        """ Compute the vector correlation coefficient between two vector fields.

            Parameters:
            ----------
            axis: tuple, optional. Defaults to None. Axis along which to compute the metric (i.e., spatial axes).

            Returns:
            --------
            metric: np.ndarray. Vector correlation coefficient between v1 and v2.
        """

        # Compute based on the definition from Schrijver et al. (2005)
        coef = np.sum((self.v1x * self.v2x + self.v1y * self.v2y + self.v1z * self.v2z), axis=axis) / \
               (np.sqrt(np.sum(self.v1x ** 2 + self.v1y ** 2 + self.v1z ** 2, axis=axis)) *
                np.sqrt(np.sum(self.v2x ** 2 + self.v2y ** 2 + self.v2z ** 2, axis=axis)))

        return coef

    def energy_ratio(self, axis: Union[int, tuple] = None) -> floating[Any]:
        """ Compute the energy ratio between two vector fields.

            Parameters:
            ----------
            axis: tuple, optional. Defaults to None. Axis along which to compute the metric (i.e., spatial axes).

            Returns:
            --------
            metric: np.ndarray. Energy ratio between v1 and v2.
        """
        return (np.sum(self.v2x ** 2 + self.v2y ** 2 + self.v2z ** 2, axis=axis)/
                np.sum(self.v1x ** 2 + self.v1y ** 2 + self.v1z ** 2, axis=axis))


if __name__ == '__main__':
    """ Compute metrics between two vector fields.

        Parameters
        ----------
        None.

        Returns
        -------
        metrics: dict. Dictionary containing the computed metrics.
    """

    # Example vector fields with random values
    # Dimensions are (samples, height, width)
    sample_axis = 0
    spatial_axes = (1, 2)
    n_samples = 10  # Could be interpreted as time
    nx = ny = 48
    b_muram_x, b_muram_y, b_muram_z = [np.random.rand(n_samples, ny, nx) for _ in range(3)]
    v_muram_x, v_muram_y, v_muram_z = [np.random.rand(n_samples, ny, nx) for _ in range(3)]
    v_model_x, v_model_y, v_model_z = [np.random.rand(n_samples, ny, nx) for _ in range(3)]

    # Compute the vertical Poynting flux
    s_muram_z = vertical_poynting_flux(v_muram_x, v_muram_y, v_muram_z,
                                       b_muram_x, b_muram_y, b_muram_z)
    s_model_z = vertical_poynting_flux(v_model_x, v_model_y, v_model_z,
                                       b_muram_x, b_muram_y, b_muram_z)

    # ---------------------------------------------------------------------
    # Examples of computations accounting for all vector components at once
    # ---------------------------------------------------------------------

    # Initialize the Metrics class:
    # v1 is the reference field.
    # v2 is the model field.
    # Note that the z component is optional.
    metrics = Metrics(v1x=v_muram_x, v2x=v_model_x, v1y=v_muram_y, v2y=v_model_y)

    # Compute metrics over the entire data at once
    metrics_for_v_over_all_samples = {
        'vector_correlation_coefficient': metrics.vector_correlation_coefficient(),
        'cosine_similarity_index': metrics.cosine_similarity_index(),
        'normalized_error': metrics.normalized_error(),
        'mean_absolute_relative_error': metrics.mean_absolute_relative_error(),
        'median_absolute_relative_error': metrics.median_absolute_relative_error(),
        'energy_ratio': metrics.energy_ratio()
    }

    # Compute metrics over the entire data at once
    metrics_for_v_per_sample = {
        'vector_correlation_coefficient': metrics.vector_correlation_coefficient(axis=spatial_axes),
        'cosine_similarity_index': metrics.cosine_similarity_index(axis=spatial_axes),
        'normalized_error': metrics.normalized_error(axis=spatial_axes),
        'mean_absolute_relative_error': metrics.mean_absolute_relative_error(axis=spatial_axes),
        'median_absolute_relative_error': metrics.median_absolute_relative_error(axis=spatial_axes),
        'energy_ratio': metrics.energy_ratio(axis=spatial_axes)
    }


    # ---------------------------------------------------------------------
    # Examples of computations for a single vector component (e.g., vx)
    # ---------------------------------------------------------------------

    # Initialize the Metrics class:
    # v1 is the reference field.
    # v2 is the model field.
    # Note that you can just compare images by assigning them to v1x and v2x.
    metrics_vx = Metrics(v1x=v_muram_x, v2x=v_model_x)

    # Compute metrics over the entire data at once
    metrics_for_vx_over_all_samples  = {
        'vector_correlation_coefficient': metrics_vx.vector_correlation_coefficient(),
        'spearman_correlation_coefficient': spearman_correlation_coefficient(v_muram_x, v_model_x),
        'normalized_error': metrics_vx.normalized_error(),
        'mean_absolute_relative_error': metrics_vx.mean_absolute_relative_error(),
        'median_absolute_relative_error': metrics_vx.median_absolute_relative_error(),
        'energy_ratio': metrics_vx.energy_ratio()
    }

    # Compute metrics over the entire data at once
    metrics_for_vx_per_sample = {
        'vector_correlation_coefficient': metrics_vx.vector_correlation_coefficient(axis=spatial_axes),
        'spearman_correlation_coefficient': spearman_correlation_coefficient(v_muram_x, v_model_x, axis=sample_axis),
        'normalized_error': metrics_vx.normalized_error(axis=spatial_axes),
        'mean_absolute_relative_error': metrics_vx.mean_absolute_relative_error(axis=spatial_axes),
        'median_absolute_relative_error': metrics_vx.median_absolute_relative_error(axis=spatial_axes),
        'energy_ratio': metrics_vx.energy_ratio(axis=spatial_axes)
    }

    # ---------------------------------------------------------------------
    # Examples of computations for a single vector component (e.g., Sz)
    # ---------------------------------------------------------------------

    # Initialize the Metrics class:
    # v1 is the reference field.
    # v2 is the model field.
    # Note that you can just compare images by assigning them to v1x and v2x.
    metrics_sz = Metrics(v1x=s_muram_z, v2x=s_model_z)

    # Compute metrics over the entire data at once
    metrics_for_sz_over_all_samples = {
        'vector_correlation_coefficient': metrics_sz.vector_correlation_coefficient(),
        'spearman_correlation_coefficient': spearman_correlation_coefficient(v_muram_x, v_model_x),
        'normalized_error': metrics_sz.normalized_error(),
        'mean_absolute_relative_error': metrics_sz.mean_absolute_relative_error(),
        'median_absolute_relative_error': metrics_sz.median_absolute_relative_error(),
        'energy_ratio': metrics_sz.energy_ratio()
    }

    # Compute metrics over the entire data at once
    metrics_for_sz_per_sample = {
        'vector_correlation_coefficient': metrics_sz.vector_correlation_coefficient(axis=spatial_axes),
        'spearman_correlation_coefficient': spearman_correlation_coefficient(v_muram_x, v_model_x, axis=sample_axis),
        'normalized_error': metrics_sz.normalized_error(axis=spatial_axes),
        'mean_absolute_relative_error': metrics_sz.mean_absolute_relative_error(axis=spatial_axes),
        'median_absolute_relative_error': metrics_sz.median_absolute_relative_error(axis=spatial_axes),
        'energy_ratio': metrics_sz.energy_ratio(axis=spatial_axes)
    }
