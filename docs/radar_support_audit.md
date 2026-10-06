# Radar-supported reconstruction regions: VoD audit

This audit asks a narrow question: **where can the current radar stack support
an addition to a faulty LiDAR scan?** It does not require every missing LiDAR
return to be recoverable. Outside the support region, the intended behavior is
to make no reconstruction.

The candidate region is made before clean LiDAR or annotations are read:

1. Load the Doppler-radial-shifted, verified 20-scan radar stack and align it
   to the current LiDAR frame. Apply the same faulty-LiDAR-only height filter
   used by the cross-modal trainer.
2. Within the forward 3D working volume, retain radar returns with at least
   three returns (including themselves) in a 1.5 m neighborhood. These are
   **radar seeds**. The settings are exploratory, not calibrated confidence.
3. A site is radar-supported when it lies within a tested 3D radius of a seed.
   The script sweeps 0.5, 1.0, and 2.0 m by default. The region is the union of
   those neighborhoods, not a ground-truth or predicted object box.

The audit then reads the clean scan. `faulty_source_ids` gives an exact index
mapping to surviving clean points; missing returns are clean indices absent
from that mapping. Labeled 3D boxes classify those missing returns by Car,
Pedestrian, Cyclist, or Background. They never select radar seeds or supported
sites.

From the repository root on Windows PowerShell:

```powershell
python -m scripts.audit_radar_supported_missing_lidar `
  --samples-root outputs_local/vod_range5_full_local/samples `
  --vod-root '../View-Of-Delft dataset/view_of_delft_PUBLIC' `
  --output-root outputs_local/radar_support_audit_val_full
```

For a quick evenly spaced pilot, add `--limit 50` and use another output root.
The script writes `summary.json` and `per_frame.csv`. Report missing-return
counts alongside each percentage, especially for small object classes. The
script estimates how much of the forward 3D volume the selected regions occupy
by testing 4,096 uniformly drawn sites per frame. A low
whole-scene supported fraction is not a failure of the intended selective
reconstruction. What matters next is whether the supported regions contain
useful missing object surfaces and whether a model can reconstruct those
surfaces without hallucinating road or clutter.

For a stricter exploratory setting, add
`--neighbor-radius-m 1 --min-neighbors 8 --min-scans 2`. This requires
locally repeated radar returns from at least two scan ordinals. It can exclude
useful moving-object evidence, so inspect its class and object-instance counts
before treating it as an inference gate.

This geometric audit does not estimate the correctness of generated points or
prove object-detection gain. Its 3D radius and local radar count must be fixed
using training data or a separate tuning split before a held-out comparison.

## Exploratory full-validation run

All 1,296 cached VoD validation frames were audited on 2026-10-05, with the
verified Doppler-radial 20-scan radar variant. Both settings below used a 1 m
support radius. Percentages count missing clean points inside the support
region; the volume column is a uniform 3D probe estimate.

| Local radar rule | Selected volume | All missing | Car | Pedestrian | Cyclist |
| --- | ---: | ---: | ---: | ---: | ---: |
| 3 returns within 1.5 m, one scan allowed | 6.22% | 65.31% | 87.52% | 97.52% | 97.59% |
| 8 returns within 1 m, at least two scans | 3.78% | 55.87% | 84.63% | 93.91% | 94.69% |

The object percentages are point-weighted. With the stricter rule, at least
one missing return was supported in 890 of 1,215 cars, 1,059 of 1,218
pedestrians, and 1,251 of 1,353 cyclists that had missing returns within the
working volume. Those instance counts reveal cases that aggregate point
coverage conceals. A radar point within 1 m of a clean object return does not
prove the radar measured that object or its detailed shape. These are
proximity and selectivity measurements, not a reconstruction result.
