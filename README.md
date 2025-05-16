### U-Net Land Cover Model

Uses Sentinel-2 Data.

#### Raw Data source (API Based):

- https://www.sentinel-hub.com/
- https://earthengine.google.com/
- https://browser.dataspace.copernicus.eu/

#### Labeled Data source:

- https://www.geoportal.gov.ph/ `See 2020 Land Cover Map of CAR`
- https://livingatlas.arcgis.com/landcoverexplorer

> Use WSL! Highly Recommended!

```bash
conda create -n landcover-unet
conda activate landcover-unet
conda install -c conda-forge --file conda-requirements.txt
pip install -r requirements.txt
```
