#!/usr/bin/env python3
import sys
import numpy as np
import scipy.signal
import geopandas as gpd
from pathlib import Path
from shapely.affinity import translate
from shapely.ops import transform as shp_transform
from pyproj import Transformer
from rasterio.features import rasterize

from bhume import load, write_predictions, score
from bhume.geo import open_imagery
from bhume.baseline import global_median_shift

def get_edges(geom):
    """Extraction of exterior + interior logic for complex boundary shapes."""
    edges = []
    if geom.geom_type == 'Polygon':
        edges.append(geom.exterior)
        edges.extend(geom.interiors)
    elif geom.geom_type == 'MultiPolygon':
        for p in geom.geoms:
            edges.append(p.exterior)
            edges.extend(p.interiors)
    return edges

def main(village_dir):
    village = load(village_dir)
    print(f"Hybrid Processing {village.slug}...")

    # 1. Start with Global Median Shift (Fast Baseline)
    try:
        base_preds = global_median_shift(village)
        base_preds = base_preds.set_index('plot_number')
    except Exception as e:
        print(f"Error calculating global shift: {e}")
        return

    # 2. Setup Data Structures
    plots = village.plots.copy()
    plots['total_rec'] = plots['recorded_area_sqm'].fillna(0) + \
                        plots['pot_kharaba_ha'].fillna(0) * 10000

    results = []
    total = len(plots)

    with open_imagery(village.imagery_path) as img_src:
        to_img = Transformer.from_crs('EPSG:4326', img_src.crs, always_xy=True).transform
        from_img = Transformer.from_crs(img_src.crs, 'EPSG:4326', always_xy=True).transform
        
        bnd_src = None
        if village.boundaries_path:
            bnd_src = open_imagery(village.boundaries_path)

        for i, (pn, plot) in enumerate(plots.iterrows()):
            # Progress tracking (every 200 plots)
            if (i+1) % 200 == 0:
                print(f"  Progress: {i+1}/{total} plots processed...")

            map_area = plot['map_area_sqm']
            rec_area = plot['total_rec']
            ratio = map_area / rec_area if rec_area > 0 else 1.0
            area_conf = np.exp(-abs(1-ratio))

            # A) Hard Filter: flag plots with extreme area mismatch
            if ratio < 0.5 or ratio > 2.0:
                results.append({
                    'plot_number': pn, 'status': 'flagged', 'geometry': plot['geometry'],
                    'confidence': 0.05, 'method_note': f'flag: area mismatch {ratio:.2f}'
                })
                continue
            
            # Start with the global shift geometry
            final_geom = base_preds.loc[pn, 'geometry']
            status = 'corrected'
            method = 'global_shift fallback'
            conf = 0.3 * area_conf  # Default confidence for global shift

            # B) Local X-Corr Enhancement (Only for reliable area-ratio plots)
            if bnd_src and 0.8 <= ratio <= 1.2:
                try:
                    # Use small padding (10m) for speed
                    geom_local = shp_transform(to_img, plot['geometry'])
                    pad_m = 10
                    minx, miny, maxx, maxy = geom_local.bounds
                    window = bnd_src.window(minx-pad_m, miny-pad_m, maxx+pad_m, maxy+pad_m)
                    
                    bnd_patch = bnd_src.read(1, window=window).astype(float)
                    
                    # C) Speed Check: Skip if too small
                    if bnd_patch.shape[0] >= 5 and bnd_patch.shape[1] >= 5:
                        template = rasterize(
                            [(e, 255) for e in get_edges(geom_local)],
                            out_shape=bnd_patch.shape,
                            transform=bnd_src.window_transform(window),
                            fill=0, all_touched=True
                        ).astype(float)
                        
                        t_std = template.std()
                        p_std = bnd_patch.std()
                        if t_std > 1e-3 and p_std > 1e-3:
                            t_norm = (template - template.mean()) / t_std
                            p_norm = (bnd_patch - bnd_patch.mean()) / p_std
                            
                            # FFT Convolve is significantly faster for large sets
                            # flip template for correlation equivalence
                            corr = scipy.signal.fftconvolve(p_norm, t_norm[::-1, ::-1], mode='same')
                            
                            y, x = np.unravel_index(np.argmax(corr), corr.shape)
                            dy_px = y - corr.shape[0] // 2
                            dx_px = x - corr.shape[1] // 2
                            
                            res_x, res_y = bnd_src.res
                            dx_m, dy_m = dx_px * res_x, -dy_px * res_y 
                            
                            # Limit shift to 15m to prevent false snapping
                            if abs(dx_m) <= 15 and abs(dy_m) <= 15:
                                final_geom = shp_transform(from_img, translate(geom_local, dx_m, dy_m))
                                peak_val = np.max(corr)
                                # Combined Confidence
                                conf_match = min(1.0, peak_val / (template.sum() / 30 + 1))
                                conf = area_conf * conf_match
                                method = f'hybrid_xcorr shift={dx_m:.1f},{dy_m:.1f}'
                except:
                    pass # Silently fallback to global shift on geometry errors

            results.append({
                'plot_number': pn, 
                'status': status, 
                'geometry': final_geom,
                'confidence': round(max(0.01, min(0.98, conf)), 3),
                'method_note': method
            })

    if bnd_src: bnd_src.close()

    # 3. Save Results and Final Score
    preds = gpd.GeoDataFrame(results).set_crs('EPSG:4326')
    out_path = Path(village_dir) / 'predictions.geojson'
    write_predictions(out_path, preds)
    
    print(f"\nCompleted {len(results)} plots.")
    print(f"\nFinal score for {village.slug}:")
    print(score(preds, village))

if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else 'data/34855_vadnerbhairav_chandavad_nashik')
