# Bengaluru Ecological Stress Peaks

This project identifies ecological stress peaks in Bengaluru by combining vegetation loss, built-up expansion, water loss, and heat gain into a composite ecological stress index.

## What this repository contains

- `Data and methods/`  
  Contains processed ecological stress data, GeoJSON outputs, and methodology metadata.

- `Scripts used/`  
  Contains Python scripts used to build samples, render maps, and generate ecological stress peak visualizations.

## Main idea

Instead of looking only at heat, this project identifies areas where multiple forms of ecological stress peak together:

- Vegetation loss
- Built-up gain
- Water loss
- Heat gain

These are combined into a ward-level ecological stress index for Bengaluru.

## Files included

### Data and methods

- `ecological_stress_index.csv`
- `ecological_stress_index.geojson`
- `ecological_stress_index_method.json`
- `ecological_stress_heightmap_meta.json`
- `ecological_stress_mesh_meta.json`

### Scripts used

- `18_build_ecological_stress_samples.py`
- `19_render_ecological_stresspeaks_blender.py`
- `21_render_redone_2d_maps.py`

## Note

Raw raster inputs are not included because of file size and source constraints. This repository contains processed outputs, methodology files, and scripts used for visualization.

## Author

Pooja Harish
