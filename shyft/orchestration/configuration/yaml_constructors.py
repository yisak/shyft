import os
from shyft.repository.netcdf import region_model_repository,GeoTsRepository
from shyft.repository import geo_ts_repository_collection
import yaml


class YamlContent(object):
    """
    Concrete class for yaml content.
    """

    def __init__(self, config_file):
        self._config_file = config_file
        with open(config_file) as cfg_file:
            config = yaml.load(cfg_file)
        # Expose all keys in yaml file as attributes
        self.__dict__.update(config)

    def __repr__(self):
        srepr = "%s(" % self.__class__.__name__
        for key in self.__dict__:
            srepr += "%s=%r, " % (key, self.__dict__[key])
        srepr = srepr[:-2]
        return srepr + ")"


class RegionConfig(region_model_repository.RegionConfig):
    """
    Yaml based region configuration, using a YamlContent instance
    for holding the content.
    """

    def __init__(self, config_file):
        self._config = YamlContent(config_file)

    def parameter_overrides(self):
        return getattr(self._config, "parameter_overrides", {})

    def domain(self):
        return self._config.domain

    def repository(self):
        return self._config.repository


class ModelConfig(region_model_repository.ModelConfig):
    """
    Yaml based model configuration, using a YamlContent instance
    for holding the content.
    """

    def __init__(self, config_file):
        self._config = YamlContent(config_file)

    def interpolation_parameters(self):
        return self._config.parameters["interpolation"]

    def model_parameters(self):
        return self._config.parameters["model"]


def nc_geo_ts_repo_constructor(params):
    station_file = params["stations_met"]
    if not os.path.isabs(station_file):
        # Relative paths will be prepended the data_dir
        station_file = os.path.join(params['data_dir'], station_file)
    return GeoTsRepository(params, station_file, "")


def nc_region_model_repo_constructor(region_config, model_config, model_t, epsg):
    return region_model_repository.RegionModelRepository(region_config, model_config, model_t, epsg)


r_m_repo_constructors ={'netcdf':nc_region_model_repo_constructor}
geo_ts_repo_constructors ={'source_repo1':nc_geo_ts_repo_constructor}
