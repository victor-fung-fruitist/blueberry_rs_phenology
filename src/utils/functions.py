import datetime
import functools
import gc
import json
import pathlib

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import planetary_computer
import pystac
import pystac_client
import rioxarray
import stackstac
import xarray as xr


# Global Variables
# ===========

data_dir = pathlib.Path("data")


# Data Ingestion
# ===========


def get_items_in_aoi(
    aoi: gpd.GeoDataFrame,
    time_of_interest: str,
    host_url: str = "https://planetarycomputer.microsoft.com/api/stac/v1",
    collections_of_interest: list[str] = ["sentinel-2-l2a"],
    cloud_fraction_upper_bound_perc: float = 50,  # percent of image covered by clouds
) -> pystac.ItemCollection:
    """
    Get items in an AOI from the Planetary Computer API

    Args:
        aoi : gpd.GeoDataFrame - the area of interest to clip to
        time_of_interest : str - the time of interest
        collections_of_interest : list[str] - the collections of interest
        cloud_fraction_upper_bound_perc : float - the upper bound of cloud fraction


    Returns
        pystac.ItemCollection - the items in the AOI

    """

    area_of_interest: dict = (
        # convert to json then extract the geometry
        json.loads(
            # convert to json then
            aoi["geometry"].to_json(),
        )["features"][0]["geometry"]
    )

    # Environment setup
    catalog: pystac_client.Client = pystac_client.Client.open(
        host_url,
        modifier=planetary_computer.sign_inplace,
    )

    if collections_of_interest == ["sentinel-2-l2a"]:
        search: pystac_client.ItemSearch = catalog.search(
            collections=collections_of_interest,
            intersects=area_of_interest,
            datetime=time_of_interest,
            query={
                "eo:cloud_cover": {
                    "lt": cloud_fraction_upper_bound_perc,
                },
            },
        )

        print(
            f"Found {len(search.item_collection())} Items with cloud cover < {cloud_fraction_upper_bound_perc}%"
        )
        return search.item_collection()

    elif collections_of_interest == ["naip"]:
        bbox: list = aoi.total_bounds

        search = catalog.search(
            collections=["naip"],
            bbox=bbox,
            datetime=int(time_of_interest),
        )

        print(f"Found {len(search.item_collection())} Items")

        return search.item_collection()


def _lazy_load_items(
    items: pystac.ItemCollection,
    aoi: gpd.GeoDataFrame,
    assets: list[str] = [
        "B02",
        "B03",
        "B04",
        "B05",
        "B06",
        "B07",
        "B08",
        "B8A",
        "B11",
        "B12",
        "SCL",
    ],
) -> xr.Dataset:
    """
    Lazy load items

    Args:
        items : pystac.ItemCollection - the items to load
        aoi : gpd.GeoDataFrame - the area of interest to clip to
        assets : list[str] - the assets to load

    Returns
        xr.Dataset - the lazily loaded data (in the form of dask graphs)
    """

    bounds_latlon: tuple[float, float, float, float] = tuple(
        aoi.copy()
        .assign(
            geometry=lambda x: x["geometry"].to_crs(
                crs=4326,
            ),
        )
        .total_bounds
    )  # native bounds make it much faster (lat lon refers to coordinates in CRS = 4326)

    return stackstac.stack(
        items,
        assets=assets,
        epsg=aoi.crs.to_epsg(),
        bounds_latlon=bounds_latlon,
    )


# Data Preprocessing
# ===========


def _mask_cloud_and_snow(
    data: xr.DataArray,
    mask_var: str = "SCL",
    mask_values: list[int] = [
        8,  # cloud medium probability
        9,  # cloud high probability
        10,  # thin cirrus
        11,  # snow
    ],
) -> xr.DataArray:
    """
    Mask clouds in the data

    Args:
        data : xr.DataArray - the data to mask

    Returns
        xr.DataArray - the masked data

    """

    return data.where(
        lambda da: ~da.sel(band=mask_var).isin(mask_values),
        other=np.nan,
    )


def _harmonize_to_old(
    data: xr.DataArray,
) -> xr.DataArray:
    """
    Harmonize new Sentinel-2 data to the old baseline; reference - https://sentinels.copernicus.eu/web/sentinel/-/copernicus-sentinel-2-major-products-upgrade-upcoming

    Parameters
    ----------
    data: xarray.DataArray
        A DataArray with four dimensions: time, band, y, x

    Returns
    -------
    harmonized: xarray.DataArray
        A DataArray with all values harmonized to the old
        processing baseline.
    """

    # The harmonization is only necessary for data after the cutoff date (see reference)
    cutoff: datetime.datetime = datetime.datetime(2022, 1, 25)

    # The offset is the value to subtract from the data
    offset: int = 1000

    # The bands that need to be harmonized
    bands: list[str] = [
        "B01",
        "B02",
        "B03",
        "B04",
        "B05",
        "B06",
        "B07",
        "B08",
        "B8A",
        "B09",
        "B10",
        "B11",
        "B12",
    ]

    # Separate the old and new data
    old: xr.DataArray = data.sel(time=slice(cutoff))

    # Check which bands are in the data
    bands_to_process: list[str] = list(set(bands) & set(data.band.data.tolist()))

    # Separate the new data which doesn't need to be harmonized
    new: xr.DataArray = data.sel(
        time=slice(cutoff, None),
    ).drop_sel(
        band=bands_to_process,
        errors="ignore",
    )

    # If there are bands to process, harmonize them
    if len(new["time"]) > 0:
        new_harmonized: xr.DataArray = data.sel(
            time=slice(cutoff, None),
            band=bands_to_process,
        ).clip(
            min=offset,
        )
        new_harmonized -= offset

        # append the harmonized data to the new untouched data
        new: xr.DataArray = xr.concat([new, new_harmonized], "band").sel(
            band=data.band.data.tolist()
        )

        return xr.concat([old, new], dim="time")
    else:
        # If there are no bands to process, return the data as is
        return old


def _assign_no_data(
    ds: xr.Dataset,
) -> xr.Dataset:
    """
    Assign no data values (0 for Sentinel-2 band intensity)

    Args:
        ds : xr.Dataset - the dataset to assign no data values

    Returns
        xr.Dataset - the dataset with no data values assigned
    """

    return ds.where(
        lambda x: x > 0,
        other=np.nan,
    )  # sentinel-2 uses 0 as nodata


def _clip_to_aoi(
    da: xr.DataArray,
    aoi: gpd.GeoDataFrame,
) -> xr.DataArray:
    """
    Clip an xarray DataArray to the area of interest

    Args:
        da: xr.DataArray - the data to clip
        aoi : gpd.GeoDataFrame - the area of interest to clip to

    Returns
        xr.DataArray - the clipped data
    """

    return da.pipe(
        lambda x: x.rio.clip_box(
            *aoi.copy()
            .to_crs(
                crs=x.rio.crs,
            )
            .total_bounds,
        )
        # faster than using the crs arg
    ).pipe(
        lambda x: x.rio.clip(
            aoi.copy().to_crs(
                crs=x.rio.crs,
            )["geometry"]
        )
        # faster than using the crs arg
    )


def _convert_da_to_ds(
    da: xr.DataArray,
    band_name: str = "band",
) -> xr.Dataset:
    """
    Convert a DataArray to a Dataset

    Args:
        da : xr.DataArray - the DataArray to convert
        band_name : str - the name of the band

    Returns
        xr.Dataset - the converted Dataset
    """

    return da.to_dataset(
        dim=band_name,
    )


def ingest_img(
    aoi: gpd.GeoDataFrame,
    from_year: int = 2019,
    to_year: int = 2023,
    collections_of_interest: list[str] = ["sentinel-2-l2a"],
    cloud_fraction_upper_bound_perc: float = 10,
) -> xr.Dataset:
    """
    Ingest images from the Planetary Computer API

    Args:
        aoi : gpd.GeoDataFrame - the area of interest to clip to
        from_year : int - the start year
        to_year : int - the end year

    Returns
        xr.Dataset - the loaded data
    """

    items: pystac = functools.reduce(
        lambda x, y: x + y,
        [
            get_items_in_aoi(
                aoi=aoi,
                time_of_interest=f"{year}-01-01/{year}-12-31",
                collections_of_interest=collections_of_interest,
                cloud_fraction_upper_bound_perc=cloud_fraction_upper_bound_perc,
            )
            for year in range(from_year, to_year + 1)
        ],
    )

    da: xr.DataArray = (
        _lazy_load_items(
            items=items,
            aoi=aoi,
        )
        .pipe(
            _clip_to_aoi,
            aoi=aoi,
        )
        .pipe(
            _harmonize_to_old,
        )
        .pipe(
            _mask_cloud_and_snow,
        )
        .pipe(
            _assign_no_data,
        )
    )

    # convert outputs to xr.dataset
    ds: xr.Dataset = da.pipe(_convert_da_to_ds)

    return ds


# Feature Engineering (Primary)
# ===========


def cal_ndvi(
    ds: xr.Dataset,
    red: str,
    nir: str,
) -> xr.Dataset:
    """
    Calculate NDVI (Normalized Difference Vegetation Index)

    Args:
        ds : xr.Dataset - the dataset to calculate NDVI on
        red : str - variable for the red band
        nir : str - variable for the NIR band

    Returns
        xr.Dataset - the dataset with NDVI calculated
    """

    return ds.assign(ndvi=lambda x: (x[nir] - x[red]) / (x[nir] + x[red]))


def cal_nbr2(
    ds: xr.Dataset,
    swir16: str,
    swir22: str,
) -> xr.Dataset:
    """
    Calculate NBR2 (Normalized Burn Ratio 2)

    Args:
        ds : xr.Dataset - the dataset to calculate NBR2 on
        swir16 : str - variable for the SWIR16 band
        swir22 : str - variable for the SWIR22 band

    Returns
        xr.Dataset - the dataset with NBR2 calculated
    """

    return ds.assign(nbr2=lambda x: (x[swir16] - x[swir22]) / (x[swir16] + x[swir22]))


def cal_gvi(
    ds: xr.Dataset,
    green: str,
    nir: str,
) -> xr.Dataset:
    """
    Calculate GVI (Green Vegetation Index)

    Args:
        ds : xr.Dataset - the dataset to calculate GVI on
        green : str - variable for the green band
        nir : str - variable for the NIR band

    Returns
        xr.Dataset - the dataset with GVI calculated
    """

    return ds.assign(gvi=lambda x: (x[nir] - x[green]) / (x[nir] + x[green]))


def cal_bsi(
    ds: xr.Dataset,
    swir22: str,
    nir: str,
    red: str,
    blue: str,
) -> xr.Dataset:
    """
    Calculate BSI (Bare Soil Index)

    Args:
        ds : xr.Dataset - the dataset to calculate BSI on
        swir22 : str - variable for the SWIR22 band
        nir : str - variable for the NIR band
        red : str - variable for the red band
        blue : str - variable for the blue band

    Returns
        xr.Dataset - the dataset with BSI calculated
    """

    return ds.assign(
        bsi=lambda x: ((x[swir22] + x[red]) - (x[nir] + x[blue]))
        / ((x[swir22] + x[red]) + (x[nir] + x[blue]))
    )  # check definition


def compute_indices(
    ds: xr.Dataset,
    red: str = "B04",
    nir: str = "B08",
    swir16: str = "B11",
    swir22: str = "B12",
    green: str = "B03",
    blue: str = "B02",
) -> xr.Dataset:
    """
    Compute metrics

    Args:
        ds : xr.Dataset - the dataset to compute metrics on
        red : str - variable for the red band
        nir : str - variable for the NIR band
        swir16 : str - variable for the SWIR16 band
        swir22 : str - variable for the SWIR22 band
        green : str - variable for the green band
        blue : str - variable for the blue band

    Returns
        xr.Dataset - the dataset with metrics computed
    """

    return (
        ds.pipe(
            cal_ndvi,
            red=red,
            nir=nir,
        )
        .pipe(
            cal_nbr2,
            swir16=swir16,
            swir22=swir22,
        )
        .pipe(
            cal_gvi,
            green=green,
            nir=nir,
        )
        .pipe(
            cal_bsi,
            swir22=swir22,
            nir=nir,
            red=red,
            blue=blue,
        )
    )


# Feature Engineering (Secondary)
# ===========


def compute_ndvi_stats(
    ds: xr.Dataset,
    ndvi: str = "ndvi",
) -> tuple[
    xr.DataArray,
    xr.DataArray,
    xr.DataArray,
]:
    """
    Compute NDVI statistics (5-th percentile, 95 percentile, 95%-5% IQR)

    Args:
        ds : xr.Dataset - the dataset to compute NDVI statistics on
        ndvi : str - the variable for NDVI

    Returns
        tuple[xr.DataArray, xr.DataArray, xr.DataArray] - the NDVI statistics
    """

    # compute percentiles at once
    ndvi_q: xr.Dataset = ds[ndvi].quantile(
        q=[
            0.05,
            0.95,
        ],
        skipna=True,
        dim="time",
    )

    # extract the percentiles
    ndvi_q05: xr.DataArray = ndvi_q.sel(
        quantile=0.05,
    )
    ndvi_q95: xr.DataArray = ndvi_q.sel(
        quantile=0.95,
    )

    ndvi_iqr: xr.DataArray = ndvi_q95 - ndvi_q05

    return ndvi_q05, ndvi_q95, ndvi_iqr


# Pipeline Running
# ===========


def ingesting(
    aoi: gpd.GeoDataFrame,
    year: int,
    is_output: bool = True,
    skip_if_exists: bool = False,
    is_return_ds: bool = False,
    collections_of_interest: list[str] = ["sentinel-2-l2a"],
    cloud_fraction_upper_bound_perc: float = 10,
) -> None | xr.Dataset:
    """
    Run the ingestion pipeline, which involves these steps:
    1. Download the data from the Planetary Computer API
    2. Save the data to disk as netcdf for the next steps

    Args:
        aoi : gpd.GeoDataFrame - the area of interest
        year : int - the year of interest
        is_output : bool - whether to output the results
        skip_if_exists : bool - whether to skip if the file already exists
        is_return_ds : bool - whether to return the dataset

    Returns
        None | xr.Dataset
    """

    print(f"Begin downloading and saving: year={year}, aoi={aoi['name'].values[0]}")

    """
    Step 0: Setup
    """
    # setup output directory
    output_dir: pathlib.Path = pathlib.Path(
        data_dir,
        "05_model_inputs",
        aoi["name"].values[0],
    )

    output_fname: pathlib.Path = pathlib.Path(
        output_dir,
        f"ds_{year}.nc",
    )

    if skip_if_exists:
        # check if file has size > 5 MB and exists
        if pathlib.Path(
            output_dir,
            f"ds_{year}.nc",
        ).exists():
            # only skip if the file size is > 5 MB
            if output_fname.stat().st_size > 5_000_000:
                print(
                    f"Skipping download and save: year={year}, aoi={aoi['name'].values[0]}"
                )

                ds: xr.Dataset = xr.open_dataset(
                    output_fname,
                )

                return ds

    """
    Step 1: Download
    """
    # lazy load the items
    ds: xr.Dataset = ingest_img(
        aoi=aoi,
        from_year=year,
        to_year=year,
        collections_of_interest=collections_of_interest,
        cloud_fraction_upper_bound_perc=cloud_fraction_upper_bound_perc,
    )

    """
    Step 2: Save
    """
    if is_output:
        # create the directory if it doesn't exist
        output_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        # convert attributes to string for saving to netcdf intermediate files
        for attr in ds.attrs:
            ds.attrs[attr] = str(ds.attrs[attr])

        try:
            ds.drop_vars(
                ["proj:bbox"],
                errors="ignore",
            ).to_netcdf(
                path=output_fname,
                engine="netcdf4",
                mode="w",
                encoding={
                    var: {
                        "zlib": True,
                        "complevel": 9,
                    }
                    for var in ds.keys()
                },
            )
        except ValueError:
            print(f"Error saving {output_fname}")

    print(f"Done download and save: year={year}, aoi={aoi['name'].values[0]}")

    if is_return_ds:
        return ds
    else:
        return None


def feature_engineering(
    aoi: gpd.GeoDataFrame,
    year: int,
    ds_in: xr.Dataset = None,
    month_of_interest: list[int] = [
        1,
        2,
        3,
        4,
        5,
        6,
        7,
        8,
        9,
        10,
        11,
        12,
    ],
    is_output: bool = True,
    output_dir: pathlib.Path = None,
    skip_if_exists: bool = False,
) -> None:
    """
    Run the feature engineering pipeline, which involves these steps:
    1. Load the Sentinel-2 images
    2. Compute NDVI statistics (5-th percentile, 95 percentile, 95%-5% IQR)
    3. Save the results to disk as tif for soil property modeling

    Args:
        aoi : gpd.GeoDataFrame - the area of interest
        year : int - the year of interest
        ds_in : xr.Dataset - the dataset to use
        month_of_interest : list[int] - the months of interest
        is_output : bool - whether to output the results
        output_dir : pathlib.Path - the output directory
        skip_if_exists : bool - whether to skip if the file already exists

    Returns
        None
    """

    print(f"Begin feature engineering: year={year}, aoi={aoi['name'].values[0]}")

    """
    Step 0: Setup
    """
    if output_dir is None:
        output_dir: pathlib.Path = pathlib.Path(
            data_dir,
            "08_reporting",
            aoi["name"].values[0],
            "ndvi_scaled",
            f"{'all_12_months' if len(month_of_interest) == 12 else 'mar_to_oct'}",
        )

    if skip_if_exists:
        # check if output exists
        if (
            pathlib.Path(
                output_dir,
                f"ndvi_q95_{year}.tif",
            ).exists()
            & pathlib.Path(
                output_dir,
                f"ndvi_iqr_{year}.tif",
            ).exists()
        ):
            # only skip if the file size is > 500 KB
            if (
                pathlib.Path(
                    output_dir,
                    f"ndvi_q95_{year}.tif",
                )
                .stat()
                .st_size
                > 500_000
            ) & (
                pathlib.Path(
                    output_dir,
                    f"ndvi_iqr_{year}.tif",
                )
                .stat()
                .st_size
                > 500_000
            ):
                print(f"Skipping modeling: year={year}, aoi={aoi['name'].values[0]}")

                return None

    """
    Step 1: Load the Sentinel-2 images
    """
    input_fname: pathlib.Path = pathlib.Path(
        data_dir,
        "05_model_inputs",
        aoi["name"].values[0],
        f"ds_{year}.nc",
    )

    if ds_in is None:
        try:
            print(f"File found; using {input_fname}")

            ds_in: xr.Dataset = xr.open_dataset(
                input_fname,
                cache=False,
            )

        except FileNotFoundError:
            raise FileNotFoundError(f"File not found: {input_fname}")

    """
    Step 2: Compute NDVI statistics
    """
    ds: xr.Dataset = ds_in.pipe(
        lambda x: x.sel(
            time=x["time"].dt.month.isin(month_of_interest),
        )
    ).pipe(
        cal_ndvi,
        red="B04",
        nir="B08",
    )

    ndvi_q05, ndvi_q95, ndvi_iqr = compute_ndvi_stats(
        ds=ds,
    )

    """
    Step 3: Save the results
    """
    if is_output:
        # create the directory if it doesn't exist
        output_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        (ndvi_q95 * 10_000).astype("int16").rio.write_crs(
            4326,
        ).rio.to_raster(
            raster_path=pathlib.Path(
                output_dir,
                f"ndvi_q95_{year}.tif",
            ),
            compress="LZW",
        )

        (ndvi_iqr * 10_000).astype("int16").rio.write_crs(
            4326,
        ).rio.to_raster(
            raster_path=pathlib.Path(
                output_dir,
                f"ndvi_iqr_{year}.tif",
            ),
            compress="LZW",
        )

    print(f"Done feature engineering: year={year}, aoi={aoi['name'].values[0]}")

    return None


def modeling(
    aoi: gpd.GeoDataFrame,
    year: int,
    ds_in: xr.Dataset = None,
    month_of_interest: list[int] = [
        1,
        2,
        3,
        4,
        5,
        6,
        7,
        8,
        9,
        10,
        11,
        12,
    ],
    bare_soil_counts_threshold: int = 1,
    masking_method: str = "Zizala2022",
    is_output: bool = True,
    is_sense_check: bool = False,
    output_dir: pathlib.Path = None,
    skip_if_exists: bool = False,
) -> None:
    """
    Run the modeling pipeline, which involves these steps:
    1. Load the Sentinel-2 images
    2. Compute indices and bare soil composite
    3. Save the results to disk as tif for soil property modeling

    Args:
        aoi : gpd.GeoDataFrame - the area of interest
        year : int - the year of interest
        ds_in : xr.Dataset - the dataset to use
        month_of_interest : list[int] - the months of interest
        bare_soil_counts_threshold : int - the bare soil counts threshold
        masking_method : str - the masking method
        is_output : bool - whether to output the results
        is_sense_check : bool - whether to perform sense check
        output_dir : pathlib.Path - the output directory
        skip_if_exists : bool - whether to skip if the file already exists

    Returns
        None
    """

    print(
        f"Begin modeling: year={year}, aoi={aoi['name'].values[0]}, method={masking_method}"
    )

    """
    Step 0: Setup
    """
    if output_dir is None:
        output_dir = pathlib.Path(
            data_dir,
            "07_model_outputs",
            aoi["name"].values[0],
            masking_method,
            f"{'all_12_months' if len(month_of_interest) == 12 else 'mar_to_oct'}"
            f"_threshold_{bare_soil_counts_threshold}",
        )

    output_fname: pathlib.Path = pathlib.Path(
        output_dir,
        f"bare_soil_composite_{year}.tif",
    )

    if skip_if_exists:
        # check if output exists
        if output_fname.exists():
            # only skip if the file size is > 500 KB
            if output_fname.stat().st_size > 500_000:
                print(f"Skipping modeling: year={year}, aoi={aoi['name'].values[0]}")

                return None

    """
    Step 1. Load the Sentinel-2 images
    """
    # check if input data exists
    input_fname: pathlib.Path = pathlib.Path(
        data_dir,
        "05_model_inputs",
        aoi["name"].values[0],
        f"ds_{year}.nc",
    )

    if ds_in is None:
        try:
            print(f"File found; using {input_fname}")

            ds_in: xr.Dataset = xr.open_dataset(
                input_fname,
                cache=False,
            )

        except FileNotFoundError:
            raise FileNotFoundError(f"File not found: {input_fname}")

    """
    Step 2: Compute indices and bare soil composite
    """
    ds: xr.Dataset = ds_in.pipe(
        lambda x: x.sel(
            time=x["time"].dt.month.isin(month_of_interest),
        )
    ).pipe(
        compute_indices,
        red="B04",
        nir="B08",
        swir16="B11",
        swir22="B12",
        green="B03",
        blue="B02",
    )

    # compute the bare soil composite
    bare_soil_composite = compute_bare_soil_composite(
        ds=ds,
        bare_soil_counts_threshold=bare_soil_counts_threshold,
        masking_method=masking_method,
    )

    """
    Step 3: Save the results
    """
    if is_output:
        # create the directory if it doesn't exist
        output_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        # save the outputs
        bare_soil_composite[
            [
                "B04",  # red
                "B03",  # green
                "B02",  # blue
                "B05",
                "B06",
                "B07",
                "B08",
                "B8A",
                "B11",
                "B12",
                "bare_soil_counts",
                "non_bare_soil_counts",
            ]
        ].rio.to_raster(
            raster_path=output_fname,
            compress="LZW",
        )

    if is_sense_check:
        # sense check
        fig, axs = plt.subplots(
            1,
            2,
            figsize=(15, 10),
        )

        ds["ndvi"].isel(
            time=0,
        ).plot(
            ax=axs[0],
        )
        axs[0].set_title("NDVI")

        xr.plot.imshow(
            bare_soil_composite[["B04", "B03", "B02"]].to_array(name="band") / 10_000,
            ax=axs[1],
            # robust=True,
        )

        axs[1].set_title("Bare Soil Composite")

        # turn off axes
        for ax in axs:
            ax.axis("off")

        # add year to the title
        fig.suptitle(year)

        # plt.show()

    print(
        f"Done modeling: year={year}, aoi={aoi['name'].values[0]}, method={masking_method}"
    )

    gc.collect()

    return None


def creating_combined_composite(
    aoi: gpd.GeoDataFrame,
    from_year: int,
    to_year: int,
    masking_method: str,
    month_of_interest: list[int] = range(1, 12 + 1),
    bare_soil_counts_threshold: int = 1,
    is_sense_check: bool = True,
    is_output: bool = True,
    output_dir: pathlib.Path = None,
    skip_if_exists=False,
) -> xr.DataArray:
    """
    Create the combined composite across years, which involves these steps:
    1. Load the bare soil composite from years of interest
    2. Compute the combined composite
    3. Output the results

    Args:
        aoi : gpd.GeoDataFrame - the area of interest
        from_year : int - the start year
        to_year : int - the end year
        masking_method : str - the masking method
        month_of_interest : list[int] - the months of interest
        bare_soil_counts_threshold : int - the bare soil counts threshold
        is_sense_check : bool - whether to perform sense check
        is_output : bool - whether to output the results
        output_dir : pathlib.Path - the output directory
        skip_if_exists : bool - whether to skip if the file already exists

    Returns
        xr.DataArray
    """

    print(
        "Running create_combined_composite for ",
        aoi["name"].values[0],
        " using ",
        masking_method,
    )

    """
    Step 0: Setup
    """
    prefix: str = f"{'all_12_months' if len(month_of_interest) == 12 else 'mar_to_oct'}"
    suffix: str = f"_threshold_{bare_soil_counts_threshold}"

    if output_dir is None:
        output_dir = pathlib.Path(
            data_dir,
            "08_reporting",
            aoi["name"].values[0],
            masking_method,
            prefix + suffix,
        )

    output_fname: pathlib.Path = pathlib.Path(
        output_dir,
        "bare_soil_composite_final.tif",
    )

    if skip_if_exists:
        if output_fname.exists():
            # only skip if the file size is > 500 KB
            if output_fname.stat().st_size > 500_000:
                print(
                    "Skipping create_combined_composite for ",
                    aoi["name"].values[0],
                    " using ",
                    masking_method,
                )
                return None

    """
    Step 1: Load the bare soil composite from years of interest
    """
    input_dir: pathlib.Path = pathlib.Path(
        data_dir,
        "07_model_outputs",
        aoi["name"].values[0],
        masking_method,
        prefix + suffix,
    )

    # check if the input directory exists
    if not input_dir.exists():
        raise FileNotFoundError(f"File not found: {input_dir}")

    # load all yearly bare soil composite over the years of interest
    bsc = xr.concat(
        [
            rioxarray.open_rasterio(
                filename=pathlib.Path(
                    input_dir,
                    f"bare_soil_composite_{year_of_interest}.tif",
                ),
                masked=True,
                cache=False,
            ).expand_dims(
                year=[year_of_interest],
            )
            for year_of_interest in range(from_year, to_year + 1)
        ],
        pd.Index(
            range(from_year, to_year + 1),
            name="year",
        ),
    )

    """
    Step 2: Compute the combined composite

    First compute the weighted average of the bare soil and non bare soil pixels
    Then take the bare soil pixels and fill NA with the non bare soil pixels
    """
    # extract mask for bare soil and non bare soil
    bare_soil_counts: xr.DataArray = bsc.sel(band=11)

    non_bare_soil_counts: xr.DataArray = bsc.sel(band=12)

    # separate the bare soil and non bare soil pixels
    bare_soil_pixels: xr.DataArray = xr.where(
        cond=bare_soil_counts >= bare_soil_counts_threshold,
        x=bsc,
        y=np.nan,
    )

    non_bare_soil_pixels: xr.DataArray = xr.where(
        cond=bare_soil_counts < bare_soil_counts_threshold,
        x=bsc,
        y=np.nan,
    )

    # compute the weighted average of the bare soil and non bare soil pixels
    bare_soil_layer: xr.DataArray = bare_soil_pixels.weighted(
        weights=bare_soil_counts.fillna(0),
    ).mean(
        dim="year",
        skipna=True,
    )

    non_bare_soil_layer: xr.DataArray = non_bare_soil_pixels.weighted(
        weights=non_bare_soil_counts.fillna(0),
    ).mean(
        dim="year",
        skipna=True,
    )

    # combine the bare soil and non bare soil layers
    bsc_final: xr.DataArray = bare_soil_layer.fillna(
        non_bare_soil_layer,
    )

    del (
        # bsc,
        bare_soil_pixels,
        non_bare_soil_pixels,
        # bare_soil_layer,
        # non_bare_soil_layer,
    )  # delete intermediate data in case of memory issues

    gc.collect()

    """
    Step 3: Output the results
    """
    if is_output:
        # create the directory if it doesn't exist
        output_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        # output to tif
        bsc_final.sel(band=slice(1, 10)).transpose("band", "y", "x").rio.to_raster(
            raster_path=output_fname,
            dtype="int16",
            compress="LZW",
        )

    if is_sense_check:
        # plot the results for sense check

        fig, ax = plt.subplots(
            2,
            2,
            figsize=(10, 10),
            tight_layout=True,
            sharex=True,
            sharey=True,
        )

        bsc_final.sel(
            band=[1, 2, 3],
        ).plot.imshow(
            robust=True,
            ax=ax[0, 0],
        )
        ax[0, 0].set_title("Combined Composite (bare soil filled with non bare soil)")

        bsc.mean(
            dim="year",
            skipna=True,
        ).sel(
            band=[1, 2, 3],
        ).plot.imshow(
            robust=True,
            ax=ax[0, 1],
        )

        ax[0, 1].set_title("Mean Composite (for reference)")

        bare_soil_layer.sel(
            band=[1, 2, 3],
        ).plot.imshow(  # red, green, blue
            robust=True,
            ax=ax[1, 0],
        )

        ax[1, 0].set_title("bare soil (weighted average)")

        non_bare_soil_layer.sel(
            band=[1, 2, 3],
        ).plot.imshow(  # red, green, blue
            robust=True,
            ax=ax[1, 1],
        )

        ax[1, 1].set_title("non bare soil (weighted average)")

        for a in ax:
            for i in a:
                # add degree to axes ticker
                i.xaxis.set_major_formatter(
                    plt.FuncFormatter(lambda x, _: f"{x:.02f}°")
                )
                i.yaxis.set_major_formatter(
                    plt.FuncFormatter(lambda x, _: f"{x:.02f}°")
                )
                # face tickers inside
                i.tick_params(axis="both", direction="in")
                # no labels for x and y
                i.set_xlabel("")
                i.set_ylabel("")
                # remove grid
                i.grid(False)

        # plt.show()

        if output_dir is None:
            output_dir = pathlib.Path(
                data_dir,
                "07_model_outputs",
                aoi["name"].values[0],
                masking_method,
                prefix + suffix,
            )

        # create the directory if it doesn't exist
        output_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        fig.savefig(
            pathlib.Path(
                output_dir.parent,
                f"{prefix + suffix}.png",
            )
        )

    print(
        "Done create_combined_composite for ",
        aoi["name"].values[0],
        " using ",
        masking_method,
    )

    gc.collect()

    return bsc_final


# Supplementary Functions (QA and Troubleshooting)
# ===========


def ingesting_by_month(
    aoi: gpd.GeoDataFrame,
    year: int,
    month: int,
    is_output: bool = True,
    skip_if_exists: bool = False,
) -> None | xr.Dataset:
    """
    Run the ingestion pipeline, which involves these steps:
    1. Download the data from the Planetary Computer API
    2. Save the data to disk as netcdf for the next steps

    Args:
        aoi : gpd.GeoDataFrame - the area of interest
        year : int - the year of interest
        is_output : bool - whether to output the results
        skip_if_exists : bool - whether to skip if the file already exists
        is_return_ds : bool - whether to return the dataset

    Returns
        None | xr.Dataset
    """

    print(
        f"Begin downloading and saving: year={year}, {month=}, aoi={aoi['name'].values[0]}"
    )

    """
    Step 0: Setup
    """
    # setup output directory
    output_dir: pathlib.Path = pathlib.Path(
        data_dir,
        "05_model_inputs",
        aoi["name"].values[0],
    )

    output_fname: pathlib.Path = pathlib.Path(
        output_dir,
        f"ds_{year}-{str(month).zfill(2)}.nc",
    )

    if skip_if_exists:
        # check if file has size > 5 MB and exists
        if pathlib.Path(output_fname).exists():
            # only skip if the file size is > 5 MB
            if output_fname.stat().st_size > 5_000_000:
                print(
                    f"Skipping download and save: year={year}, {month=}, aoi={aoi['name'].values[0]}"
                )

                ds: xr.Dataset = xr.open_dataset(
                    output_fname,
                )

                return ds

    """
    Step 1: Download
    """
    try:
        items = get_items_in_aoi(
            aoi=aoi,
            time_of_interest=f"{year}-{str(month).zfill(2)}",
            cloud_fraction_upper_bound_perc=50,
        )

        da: xr.DataArray = (
            _lazy_load_items(
                items=items,
                aoi=aoi,
            )
            .pipe(
                _clip_to_aoi,
                aoi=aoi,
            )
            .pipe(
                _mask_cloud_and_snow,
            )
            .pipe(
                _harmonize_to_old,
            )
            .pipe(
                _assign_no_data,
            )
        )

        # convert outputs to xr.dataset
        ds: xr.Dataset = da.pipe(_convert_da_to_ds)

    except Exception as e:
        print(e)

        return None

    """
    Step 2: Save
    """
    if is_output:
        # create the directory if it doesn't exist
        output_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        # convert attributes to string for saving to netcdf intermediate files
        for attr in ds.attrs:
            ds.attrs[attr] = str(ds.attrs[attr])

        try:
            ds.drop_vars(
                ["proj:bbox"],
                errors="ignore",
            ).to_netcdf(
                path=pathlib.Path(output_fname),
                engine="netcdf4",
                mode="w",
                encoding={
                    var: {
                        "zlib": True,
                        "complevel": 9,
                    }
                    for var in ds.keys()
                },
            )

        except ValueError:
            print(f"Error saving {output_fname}")

    print(f"Done download and save: year={year}, {month=} aoi={aoi['name'].values[0]}")


# Data Output
# ===========


def output_results(
    bare_soil_composite: xr.DataArray,
    ndvi_q95: xr.DataArray,
    ndvi_iqr: xr.DataArray,
    output_dir: pathlib.Path,
    year_of_interest: int,
) -> None:
    """
    Output the image

    Args:

        bare_soil_composite : xr.DataArray - the bare soil composite
        ndvi_q95 : xr.DataArray - the 95th percentile NDVI
        ndvi_iqr : xr.DataArray - the IQR of NDVI
        output_dir : pathlib.Path - the output directory
        year_of_interest : int - the year of interest

    Returns
        None

    """

    bare_soil_composite[
        [
            "B04",  # red
            "B03",  # green
            "B02",  # blue
            "B05",
            "B06",
            "B07",
            "B08",
            "B8A",
            "B11",
            "B12",
            "bare_soil_mask",
        ]
    ].rio.to_raster(
        raster_path=pathlib.Path(
            output_dir, f"bare_soil_composite_{year_of_interest}.tif"
        ),
        dtype="uint16",
        compress="LZW",
    )
    ndvi_q95.rio.to_raster(
        raster_path=pathlib.Path(output_dir, f"ndvi_q95_{year_of_interest}.tif"),
        compress="LZW",
    )
    ndvi_iqr.rio.to_raster(
        raster_path=pathlib.Path(output_dir, f"ndvi_iqr_{year_of_interest}.tif"),
        compress="LZW",
    )

    print(f"Outputted image to {output_dir}")
