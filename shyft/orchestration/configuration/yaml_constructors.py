import os
from shyft.repository.netcdf import cf_region_model_repository,cf_geo_ts_repository
#from shyft.repository import geo_ts_repository_collection


def nc_geo_ts_repo_constructor(params):
    return cf_geo_ts_repository.GeoTsRepository(params)


def nc_region_model_repo_constructor(region_config, model_config):
    return cf_region_model_repository.RegionModelRepository(region_config, model_config)


r_m_repo_constructors ={'netcdf':nc_region_model_repo_constructor}
geo_ts_repo_constructors ={'source_repo1':nc_geo_ts_repo_constructor}
