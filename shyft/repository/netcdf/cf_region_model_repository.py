"""
Read region CF-compliant netCDF files with cell data.

This part of code is a reworked version of former orchestration2, that
depended heavily on yaml & dictionaries as a general approach. (Deprecated)

Current state is that it's not dependent on yaml anymore, configs goes into constructor etc.,
but there are still some more improvement to be done regarding use of untyped dictionaries
Ref. Note2 below. for future directions and changes.

Note1: It does require a specific content/layout of the supplied netcdf files
      this should be clearly stated.

Note2: The configuration classes are currently very loosely specified,
       and fueled by Yaml that will (sooner or later) lead to errors deep inside
       the repository. The goal is to fix these issues through a series of changes.

"""

from __future__ import absolute_import
from builtins import range
from six import iteritems

from abc import ABCMeta, abstractmethod

from os import path
import numpy as np
from netCDF4 import Dataset
from pyproj import Proj
from pyproj import transform
from shapely.geometry import Polygon
from .. import interfaces
from shyft import api
from shyft import shyftdata_dir
from shyft.orchestration.configuration.config_interfaces import RegionConfig, ModelConfig


class RegionModelRepository(interfaces.RegionModelRepository):
    """
    Repository that delivers fully specified shyft api region_models
    based on data found in netcdf files.

    Netcdf dataset assumptions:
        * Group "elevation" with variables:
            * epsg: string identifying the coordinate system
            * xcoord: array of floats
            * ycoord: array of floats
            * elevation: float array of dim (xcoord, ycoord)
        * Group "catchments" with variables:
            * catchments: int array of dim (xcoord, ycoord)
            * catchment_indices: int array of possible indices
        * Group "forest-fraction" with variables:
            * forest-fraction: float array of dim (xcoord, ycoord)
        * Group "lake-fraction" with variables:
            * lake-fraction: float array of dim (xcoord, ycoord)
        * Group "reservoir-fraction" with variables:
            * reservoir-fraction: float array of dim (xcoord, ycoord)
        * Group "glacier-fraction" with variables:
            * glacier-fraction: float array of dim (xcoord, ycoord)

    Limitations:
        This RegionModelRepository currently provide only ONE model pr. region-model instance.
        The .get_region_model(region_id,..) ignores the region_id, as there is only one model
        Future extensions:
         Pass maps of configurations into constructor, {'region-id', configuration}*
         Then do a lookup in this map to provide the model.
         Alternatively, and maybe better: Use this Repository as a component in another
         fully specified Repository that keeps several *named/identified* models).
    """

    def __init__(self, region_config, model_config):#, region_model, epsg):
        """
        Parameters
        ----------
        region_config: subclass of interface RegionConfig
            Object containing regional information, like
            catchment overrides, and which netcdf file to read
        model_config: subclass of interface ModelConfig
            Object containing model information, i.e.
            information concerning interpolation and model
            parameters
        region_model: shyft.api type
            model to construct. Has cell constructor and region/catchment
            parameter constructor.
        epsg: string
            Coordinate system for result region model
        """
        if not isinstance(region_config, RegionConfig) or \
           not isinstance(model_config, ModelConfig):
            raise interfaces.InterfaceError()
        self._rconf = region_config
        self._mconf = model_config
        self._region_model = model_config.model_type() # region_model
        self._mask = None
        self._epsg = self._rconf.domain()["EPSG"] # epsg
        self._data_file = path.join(shyftdata_dir, self._rconf.repository()["data_file"])

    @property
    def mask(self):
        """
        Get the mask for cells that have actual info.
        Returns
        -------
            mask : np.array of type bool
        """
        if self._mask is not None:
            return self._mask
        with Dataset(self._data_file) as dset:
            mask = dset.groups['catchments'].variables[
                "catchments"][:].reshape(-1) != 0
        self._mask = mask
        return mask

    def get_region_model(self, region_id, catchments=None):
        """
        Return a fully specified shyft api region_model for region_id, based on data found
        in netcdf dataset.

        Parameters
        -----------
        region_id: string
            unique identifier of region in data

        catchments: list of unique integers
            catchment indices when extracting a region consisting of a subset
            of the catchments has attribs to construct params and cells etc.

        Returns
        -------
        region_model: shyft.api type
        """

        with Dataset(self._data_file) as dset:
            grp = dset.groups["elevation"]
            xcoord = grp.variables["xcoord"][:]
            ycoord = grp.variables["ycoord"][:]
            dataset_epsg = None
            if hasattr(grp, "epsg"):
                dataset_epsg = grp.epsg
            if hasattr(grp, "EPSG"):
                dataset_epsg = grp.EPSG
            if not dataset_epsg:
                raise interfaces.InterfaceError("netcdf: epsg attr not found in group elevation")
            if dataset_epsg != self._epsg:
                source_cs = "+proj=utm +zone={} +ellps={} +datum={} +units=m +no_defs".format(
                    int(self._epsg) - 32600, "WGS84", "WGS84")
                target_cs = "+proj=utm +zone={} +ellps={} +datum={} +units=m +no_defs".format(
                    int(dataset_epsg) - 32600, "WGS84", "WGS84")
                source_proj = Proj(source_cs)
                target_proj = Proj(target_cs)
                mesh2d = np.dstack(transform(source_proj, target_proj,
                                             *np.meshgrid(xcoord, ycoord))).reshape(-1, 2)
                dx = xcoord[1] - xcoord[0]
                dy = ycoord[1] - ycoord[0]
                x_corners = np.empty(len(xcoord) + 1, dtype=xcoord.dtype)
                y_corners = np.empty(len(ycoord) + 1, dtype=ycoord.dtype)
                x_corners[1:] = xcoord + dx/2.0
                x_corners[0] = xcoord[0] - dx/2.0
                y_corners[1:] = ycoord + dy/2.0
                y_corners[0] = ycoord[0] - dy/2.0
                xc, yc = transform(source_proj, target_proj, *np.meshgrid(x_corners, y_corners))
                areas = np.empty((len(xcoord), len(ycoord)), dtype=xcoord.dtype)
                for i in range(len(xcoord)):
                    for j in range(len(ycoord)):
                        pts = [(xc[j, i],         yc[j, i]),
                               (xc[j, i + 1],     yc[j, i + 1]),
                               (xc[j + 1, i + 1], yc[j + 1, i + 1]),
                               (xc[j + 1, i],     yc[j + 1, i])]
                        areas[i, j] = Polygon(pts).area
                areas = areas.flatten()[self.mask]
            else:
                mesh2d = np.dstack(np.meshgrid(xcoord, ycoord)).reshape(-1, 2)
                areas = np.ones(len(xcoord)*len(ycoord), dtype=xcoord.dtype)[self.mask]*(
                    xcoord[1] - xcoord[0])*(ycoord[1] - ycoord[0])
            elevation = grp.variables["elevation"][:]
            coordinates = np.hstack((mesh2d, elevation.reshape(-1, 1)))[self.mask]
            catchments = dset.groups["catchments"].variables[
                "catchments"][:].reshape(-1)[self.mask]
            c_ids = dset.groups["catchments"].variables["catchment_indices"][:]

            def frac_extract(name):
                g = dset.groups  # Alias for readability
                return g[name].variables[name][:].reshape(-1)[self.mask]
            ff = frac_extract("forest-fraction")
            lf = frac_extract("lake-fraction")
            rf = frac_extract("reservoir-fraction")
            gf = frac_extract("glacier-fraction")
        # Construct bounding region
        box_fields = set(("upper_left_x", "upper_left_y", "step_x", "step_y", "nx", "ny", "EPSG"))
        if box_fields.issubset(self._rconf.domain()):
            tmp = self._rconf.domain()
            epsg = tmp["EPSG"]
            x_min = tmp["upper_left_x"]
            x_max = x_min + tmp["nx"]*tmp["step_x"]
            y_max = tmp["upper_left_x"]
            y_min = y_max - tmp["ny"]*tmp["step_y"]
            bounding_region = BoundingBoxRegion(np.array([x_min, x_max]),
                                                np.array([y_min, y_max]), epsg, self._epsg)
        else:
            bounding_region = BoundingBoxRegion(xcoord, ycoord, dataset_epsg, self._epsg)

        # Construct region parameter:
        name_map = {"gamma_snow": "gs", "priestley_taylor": "pt",
                    "kirchner": "kirchner", "actual_evapotranspiration": "ae",
                    "skaugen": "skaugen", "hbv_snow": "snow"}
        region_parameter = self._region_model.parameter_t()
        for p_type_name, value_ in iteritems(self._mconf.model_parameters()):
            if p_type_name in name_map:
                if hasattr(region_parameter, name_map[p_type_name]):
                    sub_param = getattr(region_parameter, name_map[p_type_name])
                    for p, v in iteritems(value_):
                        setattr(sub_param, p, v)
            elif p_type_name == "p_corr_scale_factor":
                region_parameter.p_corr.scale_factor = value_

        # TODO: Move into yaml file similar to p_corr_scale_factor
        radiation_slope_factor = 0.9

        # Construct cells
        cell_vector = self._region_model.cell_t.vector_t()
        for pt, a, c_id, ff, lf, rf, gf in zip(coordinates, areas, catchments, ff, lf, rf, gf):
            cell = self._region_model.cell_t()
            cell.geo = api.GeoCellData(api.GeoPoint(*pt), a, c_id, radiation_slope_factor,
                                       api.LandTypeFractions(gf, lf, rf, ff, 0.0))
            cell_vector.append(cell)

        # Construct catchment overrides
        catchment_parameters = self._region_model.parameter_t.map_t()
        for k, v in iteritems(self._rconf.parameter_overrides()):
            if k in c_ids:
                param = self._region_model.parameter_t(region_parameter)
                for p_type_name, value_ in iteritems(v):
                    if p_type_name in name_map:
                        sub_param = getattr(param, name_map[p_type_name])
                        for p, pv in iteritems(value_):
                            setattr(sub_param, p, pv)
                    elif p_type_name == "p_corr_scale_factor":
                        param.p_corr.scale_factor = value_
                    else:
                        # Avoid unknown params to go unadvertised
                        raise RegionConfigError(
                            "parameter {} is not in the set of allowed ones".format(p_type_name))

                catchment_parameters[k] = param
        region_model = self._region_model(cell_vector, region_parameter, catchment_parameters)
        region_model.bounding_region = bounding_region
        return region_model


class BoundingBoxRegion(interfaces.BoundingRegion):

    def __init__(self, x, y, point_epsg, target_epsg):
        self._epsg = str(point_epsg)
        x_min = x.ravel().min()
        x_max = x.ravel().max()
        y_min = y.ravel().min()
        y_max = y.ravel().max()
        self.x = np.array([x_min, x_max, x_max, x_min], dtype="d")
        self.y = np.array([y_max, y_max, y_min, y_min], dtype="d")
        self.x, self.y = self.bounding_box(target_epsg)
        self._epsg = str(target_epsg)

    def bounding_box(self, epsg):
        epsg = str(epsg)
        if epsg == self.epsg():
            return np.array(self.x), np.array(self.y)
        else:
            source_cs = "+proj=utm +zone={} +ellps={} +datum={} +units=m +no_defs".format(
                int(self.epsg()) - 32600, "WGS84", "WGS84")
            target_cs = "+proj=utm +zone={} +ellps={} +datum={} +units=m +no_defs".format(
                int(epsg) - 32600, "WGS84", "WGS84")
            source_proj = Proj(source_cs)
            target_proj = Proj(target_cs)
            return [np.array(a) for a in transform(source_proj, target_proj, self.x, self.y)]

    def bounding_polygon(self, epsg):
        return self.bounding_box(epsg)

    def epsg(self):
        return self._epsg