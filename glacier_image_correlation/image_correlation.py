#! /usr/bin/env python

import xarray as xr
import rasterio as rio
import rioxarray
import numpy as np
import os
from autoRIFT import autoRIFT
from scipy.interpolate import interpn
import pystac
import pystac_client
import stackstac
from dask.distributed import Client
import geopandas as gpd
from shapely.geometry import shape
import dask
import warnings
import argparse

# silence some warnings from stackstac and autoRIFT
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=UserWarning)

def download_s2(img1_product_name, img2_product_name, bbox):
    '''
    Download a pair of Sentinel-2 images acquired on given dates over a given bounding box.
    Ensures stacks use single-band assets, are in epsg=4326, and are clipped/resampled to a common grid.
    '''
    URL = "https://earth-search.aws.element84.com/v1"
    catalog = pystac_client.Client.open(URL)

    # helper to get items for a product
    def get_items_for_product(product_name):
        search = catalog.search(
            collections=["sentinel-2-l2a"],
            query=[f's2:product_uri={product_name}']
        )
        return list(search.item_collection())

    img1_items = get_items_for_product(img1_product_name)
    img2_items = get_items_for_product(img2_product_name)

    if len(img1_items) == 0 or len(img2_items) == 0:
        raise ValueError("Could not find one of the product names in STAC catalog.")

    # Inspect available assets in first item and choose single-band asset ids
    # Common Sentinel-2 naming in this catalog uses 'red', 'green', 'blue', 'nir' (or B02/B03/B04/B08)
    def choose_assets(item):
        keys = set(item.assets.keys())
        # prefer explicit names if available
        for candidate in (["nir","red","green","blue"], ["B08","B04","B03","B02"], ["nir-jp2","red-jp2","green-jp2","blue-jp2"]):
            if all(k in keys for k in candidate):
                # order we want: blue, green, red, nir (but later we just use nir)
                return candidate
        # fallback: try to pick any single-band jp2 assets with 'nir' or 'B08'
        preferred = [k for k in keys if 'nir' in k or 'B08' in k]
        if preferred:
            # try to build a minimal list
            assets = []
            for want in ["blue","green","red","nir","B02","B03","B04","B08"]:
                if want in keys:
                    assets.append(want)
            if assets:
                return assets[:4]
        # as last resort, return all single-asset keys (may include 'visual' so we'll filter later)
        return list(keys)

    assets_to_use = choose_assets(img1_items[0])
    # ensure we don't accidentally include 'visual' multi-band asset
    assets_to_use = [a for a in assets_to_use if 'visual' not in a and a.endswith(('-jp2','')) or not a == 'visual']
    # If still empty, explicit fallback
    if len(assets_to_use) == 0:
        assets_to_use = ["B02","B03","B04","B08"]


    
    print(f"DEBUG: img1_items count = {len(img1_items)}")
    print(f"DEBUG: img2_items count = {len(img2_items)}")
    
    if len(img1_items) == 0 or len(img2_items) == 0:
        raise ValueError("No Sentinel-2 items were found for the given product names or bounding box.")
    
    # Optional: print bbox info for first item
    print("DEBUG: First img1 item bbox:", getattr(img1_items[0], "bbox", None))
    print("DEBUG: First img2 item bbox:", getattr(img2_items[0], "bbox", None))

    print("DEBUG: img1 asset keys:", list(img1_items[0].assets.keys()))
    print("DEBUG: img2 asset keys:", list(img2_items[0].assets.keys()))

    
    
    # Stack items with explicit assets and epsg
    # Load only the NIR band (B08) with chunking to avoid memory overflow
    # Stack Sentinel-2 NIR band (B08) and apply chunking afterwards for memory efficiency

    # --- Choose available NIR band dynamically ---
    nir_candidates = ["nir08", "nir", "nir09", "B08", "B8A"]
    available_bands = list(img1_items[0].assets.keys())
    nir_band = next((b for b in nir_candidates if b in available_bands), None)
    
    if not nir_band:
        raise ValueError(f"No NIR band found in available assets: {available_bands}")
    
    print(f"DEBUG: Using NIR band {nir_band}")
    
    # --- Stack Sentinel-2 NIR band and chunk ---
    img1_full = stackstac.stack(img1_items, epsg=4326, assets=[nir_band]).chunk({"x": 1024, "y": 1024})
    img2_full = stackstac.stack(img2_items, epsg=4326, assets=[nir_band]).chunk({"x": 1024, "y": 1024})




    # Compute intersection bbox from items' bboxes (in lon/lat): (minx,miny,maxx,maxy)
    def items_bounds(candidate_items):
        bboxes = [it.bbox for it in candidate_items if it.bbox is not None]
        if not bboxes:
            return None
        minx = max(b[0] for b in bboxes)
        miny = max(b[1] for b in bboxes)
        maxx = min(b[2] for b in bboxes)
        maxy = min(b[3] for b in bboxes)
        return (minx, miny, maxx, maxy)

    b1 = items_bounds(img1_items)
    b2 = items_bounds(img2_items)
    if b1 is None or b2 is None:
        # fallback to provided bbox if item bboxes missing
        bounds = tuple(gpd.GeoDataFrame({'geometry':[shape(bbox)]}).total_bounds)  # (minx,miny,maxx,maxy)
    else:
        # intersection of the two sets of items
        inter = (max(b1[0], b2[0]), max(b1[1], b2[1]), min(b1[2], b2[2]), min(b1[3], b2[3]))
        # if intersection is invalid, fallback to provided bbox
        if inter[0] >= inter[2] or inter[1] >= inter[3]:
            bounds = tuple(gpd.GeoDataFrame({'geometry':[shape(bbox)]}).total_bounds)
        else:
            bounds = inter

    # Clip both stacks to the intersection bounds (ensuring same grid)
    # Collapse time dimension before reprojection

    img1_single = img1_full.isel(time=0)
    img2_single = img2_full.isel(time=0)
    
    img1_clipped = img1_single.rio.reproject("EPSG:4326").rio.clip_box(*bounds, crs="EPSG:4326")
    img2_clipped = img2_single.rio.reproject("EPSG:4326").rio.clip_box(*bounds, crs="EPSG:4326")


    # If shapes or coords differ slightly, resample img2 to img1's grid (use xarray interp)
    # choose reference grid
    ref = img1_clipped
    try:
        img2_on_ref = img2_clipped.interp(x=ref.x, y=ref.y, method="nearest")
    except Exception:
        # fallback: align by selecting overlapping coordinate ranges
        xmin = max(img1_clipped.x.min().item(), img2_clipped.x.min().item())
        xmax = min(img1_clipped.x.max().item(), img2_clipped.x.max().item())
        ymin = max(img1_clipped.y.min().item(), img2_clipped.y.min().item())
        ymax = min(img1_clipped.y.max().item(), img2_clipped.y.max().item())
        img1_clipped = img1_clipped.sel(x=slice(xmin, xmax), y=slice(ymax, ymin))
        img2_clipped = img2_clipped.sel(x=slice(xmin, xmax), y=slice(ymax, ymin))
        img2_on_ref = img2_clipped

    # convert to dataset where band is a coordinate for consistency
    img1_ds = img1_clipped.to_dataset(dim="band")
    img2_ds = img2_on_ref.to_dataset(dim="band")

    return img1_ds, img2_ds
 

def run_autoRIFT(img1, img2, skip_x=3, skip_y=3, min_x_chip=16, max_x_chip=64,
                 preproc_filter_width=3, mpflag=4, search_limit_x=30, search_limit_y=30):
    '''
    Configure and run autoRIFT feature tracking with Sentinel-2 data for large mountain glaciers
    Ensures arrays are same dtype/shape and builds grid safely.
    '''
    # convert to numpy arrays (float32) and ensure 2D
    I1 = np.array(img1, dtype=np.float32)
    I2 = np.array(img2, dtype=np.float32)

    # If input has extra dims (e.g., time or band), reduce to 2D using squeeze
    if I1.ndim > 2:
        I1 = np.squeeze(I1)
    if I2.ndim > 2:
        I2 = np.squeeze(I2)

    # Ensure shapes match: crop to minimal common shape
    rows = min(I1.shape[0], I2.shape[0])
    cols = min(I1.shape[1], I2.shape[1])
    I1 = I1[:rows, :cols]
    I2 = I2[:rows, :cols]

    # Prepare autoRIFT object
    obj = autoRIFT()
    obj.MultiThread = mpflag

    obj.I1 = I1
    obj.I2 = I2

    obj.SkipSampleX = skip_x
    obj.SkipSampleY = skip_y

    # Kernel sizes to use for correlation
    obj.ChipSizeMinX = min_x_chip
    obj.ChipSizeMaxX = max_x_chip
    obj.ChipSize0X = min_x_chip
    # oversample ratio, balancing precision and performance for different chip sizes
    obj.OverSampleRatio = {obj.ChipSize0X:16, obj.ChipSize0X*2:32, obj.ChipSize0X*4:64}

    # generate grid safely (xGrid are column indices, yGrid are row indices)
    m, n = obj.I1.shape
    # protect against too-small images
    if n <= (obj.SkipSampleX + 10) or m <= (obj.SkipSampleY + 10):
        raise ValueError("Image too small for chosen SkipSample and chip sizes")

    xGrid = np.arange(obj.SkipSampleX + 10, n - obj.SkipSampleX, obj.SkipSampleX)
    yGrid = np.arange(obj.SkipSampleY + 10, m - obj.SkipSampleY, obj.SkipSampleY)
    nd = xGrid.size
    md = yGrid.size

    obj.xGrid = np.int32(np.tile(xGrid[np.newaxis, :], (md, 1)))
    obj.yGrid = np.int32(np.tile(yGrid[:, np.newaxis], (1, nd)))

    # Build noDataMask at the grid points: True where either image has non-positive values (i.e., nodata)
    # Indexing: rows=y, cols=x
    # sample values at grid points
    I1_samples = obj.I1[obj.yGrid, obj.xGrid]
    I2_samples = obj.I2[obj.yGrid, obj.xGrid]
    valid_mask = (I1_samples > 0) & (I2_samples > 0)
    noDataMask = np.logical_not(valid_mask)

    # Initialize Dx0/Dy0 if not present (same shape as grid)
    if not hasattr(obj, "Dx0") or obj.Dx0 is None:
        obj.Dx0 = np.zeros_like(obj.xGrid, dtype=np.float32)
    if not hasattr(obj, "Dy0") or obj.Dy0 is None:
        obj.Dy0 = np.zeros_like(obj.xGrid, dtype=np.float32)

    # set search limits
    obj.SearchLimitX = np.full_like(obj.xGrid, search_limit_x)
    obj.SearchLimitY = np.full_like(obj.xGrid, search_limit_y)

    # set search limit and offsets in nodata areas
    obj.SearchLimitX = obj.SearchLimitX * (~noDataMask)
    obj.SearchLimitY = obj.SearchLimitY * (~noDataMask)
    obj.Dx0 = obj.Dx0 * (~noDataMask)
    obj.Dy0 = obj.Dy0 * (~noDataMask)
    obj.Dx0[noDataMask] = 0
    obj.Dy0[noDataMask] = 0
    obj.NoDataMask = noDataMask

    print("preprocessing images")
    obj.WallisFilterWidth = preproc_filter_width
    obj.preprocess_filt_lap()  # preprocessing with laplacian filter
    obj.uniform_data_type()

    print("starting autoRIFT")
    obj.runAutorift()
    print("autoRIFT complete")

    # convert displacement to m (if pixel size = 10 m, so Dx in pixels * 10)
    obj.Dx_m = obj.Dx * 10
    obj.Dy_m = obj.Dy * 10

    return obj


def prep_outputs(obj, img1_ds, img2_ds):
    '''
    Interpolate pixel offsets to original dimensions, calculate total horizontal velocity
    '''

    # interpolate to original dimensions 
    x_coords = obj.xGrid[0, :]
    y_coords = obj.yGrid[:, 0]
    
    # Create a mesh grid for the img dimensions
    x_coords_new, y_coords_new = np.meshgrid(
        np.arange(obj.I2.shape[1]),
        np.arange(obj.I2.shape[0])
    )
    
    # Perform bilinear interpolation using scipy.interpolate.interpn
    Dx_full = interpn((y_coords, x_coords), obj.Dx, (y_coords_new, x_coords_new), method="linear", bounds_error=False)
    Dy_full = interpn((y_coords, x_coords), obj.Dy, (y_coords_new, x_coords_new), method="linear", bounds_error=False)
    Dx_m_full = interpn((y_coords, x_coords), obj.Dx_m, (y_coords_new, x_coords_new), method="linear", bounds_error=False)
    Dy_m_full = interpn((y_coords, x_coords), obj.Dy_m, (y_coords_new, x_coords_new), method="linear", bounds_error=False)
    
    # add variables to img1 dataset
    img1_ds = img1_ds.assign({'Dx':(['y', 'x'], Dx_full),
                              'Dy':(['y', 'x'], Dy_full),
                              'Dx_m':(['y', 'x'], Dx_m_full),
                              'Dy_m':(['y', 'x'], Dy_m_full)})
    # calculate x and y velocity
    img1_ds['veloc_x'] = (img1_ds.Dx_m/(img2_ds.time.isel(time=0) - img1_ds.time.isel(time=0)).dt.days)*365.25
    img1_ds['veloc_y'] = (img1_ds.Dy_m/(img2_ds.time.isel(time=0) - img1_ds.time.isel(time=0)).dt.days)*365.25
    
    # calculate total horizontal velocity
    img1_ds['veloc_horizontal'] = np.sqrt(img1_ds['veloc_x']**2 + img1_ds['veloc_y']**2)

    return img1_ds

def get_parser():
    parser = argparse.ArgumentParser(description="Run autoRIFT to find pixel offsets for two Sentinel-2 images")
    parser.add_argument("img1_product_name", type=str, help="product name of first Sentinel-2 image")
    parser.add_argument("img2_product_name", type=str, help="product name of second Sentinel-2 image")
    return parser

def main():
    parser = get_parser()
    args = parser.parse_args()

    # hardcoding a bbox for now
    bbox = {
    "type": "Polygon",
    "coordinates": [
          [
            [
              84.31369037937048,
              28.774215826417233
            ],
            [
              84.31369037937048,
              28.65931200469386
            ],
            [
              84.47924591303064,
              28.65931200469386
            ],
            [
              84.47924591303064,
              28.774215826417233
            ],
            [
              84.31369037937048,
              28.774215826417233
            ]
          ]
        ],
    }

    # download Sentinel-2 images
    img1_ds, img2_ds = download_s2(args.img1_product_name, args.img2_product_name, bbox)
    # grab near infrared band only
    img1 = img1_ds.nir.squeeze().values
    img2 = img2_ds.nir.squeeze().values
    
    # scale search limit with temporal baseline assuming max velocity 1000 m/yr (100 px/yr)
    search_limit_x = search_limit_y = round(((((img2_ds.time.isel(time=0) - img1_ds.time.isel(time=0)).dt.days)*100)/365.25).item())
    
    # run autoRIFT feature tracking
    obj = run_autoRIFT(img1, img2, search_limit_x=search_limit_x, search_limit_y=search_limit_y)
    # postprocess offsets
    ds = prep_outputs(obj, img1_ds, img2_ds)

    # write out velocity to tif
    ds.veloc_horizontal.rio.to_raster(f'S2_{args.img1_product_name[11:19]}_{args.img2_product_name[11:19]}_horizontal_velocity.tif')

if __name__ == "__main__":
   main()
