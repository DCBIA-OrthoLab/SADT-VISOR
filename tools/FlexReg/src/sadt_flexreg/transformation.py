import numpy as np
import vtk

import logging

logger = logging.getLogger(__name__)

def RotationMatrix(axis, theta):
    """
    Return the rotation matrix associated with counterclockwise rotation about
    the given axis by theta radians.

    Parameters
    ----------
    axis : np.array
        Axis of rotation
    theta : float
        Angle of rotation in radians
    
    Returns
    -------
    np.array
        Rotation matrix
    """

    axis = np.asarray(axis)
    axis = axis / np.linalg.norm(axis)
    a = np.cos(theta / 2.0)
    b, c, d = -axis * np.sin(theta / 2.0)
    aa, bb, cc, dd = a * a, b * b, c * c, d * d
    bc, ad, ac, ab, bd, cd = b * c, a * d, a * c, a * b, b * d, c * d
    return np.array([[aa + bb - cc - dd, 2 * (bc + ad), 2 * (bd - ac)],
                    [2 * (bc - ad), aa + cc - bb - dd, 2 * (cd + ab)],
                    [2 * (bd + ac), 2 * (cd - ab), aa + dd - bb - cc]])






def TransformSurf(surf,matrix):
    assert isinstance(surf,vtk.vtkPolyData)
    surf_copy = vtk.vtkPolyData()
    surf_copy.DeepCopy(surf)
    surf = surf_copy

    transform = vtk.vtkTransform()
    transform.SetMatrix(np.reshape(matrix,16))
    surf = RotateTransform(surf,transform)

    return surf


def RotateTransform(surf, transform):

    transformFilter = vtk.vtkTransformPolyDataFilter()
    transformFilter.SetTransform(transform)
    transformFilter.SetInputData(surf)
    transformFilter.Update()
    return transformFilter.GetOutput()


def ApplyTransform(input, transform):
    """Apply `transform` to a surface.

    Upstream dispatched on type here and called `TransformDict` and
    `TransformList` for the other two, neither of which exists in this module:
    the file was copied from AREG_IOS without them, so both branches raised
    `NameError` on a path only a real run reaches. It never showed because the
    one caller in this tool passes a surface.

    Reduced to what FlexReg transforms, and anything else is refused by name
    rather than falling through silently -- returning an unchanged input would
    turn a wrong type into a registration that quietly did nothing.
    """
    if not isinstance(input, vtk.vtkPolyData):
        raise TypeError(
            "ApplyTransform expects a vtkPolyData, got {}.".format(type(input).__name__)
        )
    return TransformSurf(input, transform)