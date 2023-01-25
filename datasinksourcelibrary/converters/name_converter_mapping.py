from datasinksourcelibrary.converters.influx_flat_converter import InfluxFlatConverter


def converter_mapping(name, return_all=False):

    mapping_ = {
         'influx_flat': InfluxFlatConverter
    }

    if return_all:
        return mapping_

    return mapping_[name]
