# Doppler-aligned VoD radar stack: validation comparison

The baseline `radar_20frames_verified` aligns ego motion only. The earlier
`radar_20frames_verified_motion_aware` stack deletes older returns when
`abs(v_r_compensated) >= 1 m/s`. Neither corrects object motion.

`scripts/generate_vod_doppler_radar.py` creates two **separate** variants:

- `radar_20frames_verified_doppler_radial`: shift each historical return in
  its **source radar** horizontal line of sight by compensated Doppler times
  estimated elapsed seconds, then apply the verified ego transform. Retain
  every original return.
- `radar_20frames_verified_doppler_window`: the same shift, retaining an old
  return only while `abs(v_r_compensated) * elapsed_seconds *
  lateral_speed_ratio <= dispersion_tolerance_m`. The preview uses ratio 1
  and tolerance 2 m. Every current return is unchanged.

This is **DoppDrive-inspired**, not a reproduction of DoppDrive. The paper's
direction-dependent heading prior `g(theta)` and original radar timestamps
are unavailable in this VoD copy. VoD's seventh point field is an ordinal
scan index. We supplied an explicit 0.10022 s synchronized-frame period as an
approximation. The converted local sample metadata has a median 0.100224 s
interval over frames `00545`–`00564`, but an implausible 118 s jump at
`00544`–`00545`; it cannot be accepted as complete radar timing metadata.
Radar velocity's positive sign was selected with these **training** frames:
reversing the sign reduced current Cyclist-box concentration. Do not select
signs, tolerance, or timing using validation/test labels.

## Local training-clip preview

Twenty-one targets (`00544`–`00564`) had existing verified 20-scan stacks.
Totals count the same source scans repeatedly because target histories overlap.
Cyclist boxes include the VoD `Cyclist`, `bicycle`, and `rider` labels. The
count is the number of radar returns within **current-time** 3D annotation
boxes. It is a concentration proxy, not a localization error, true-positive
count, detector mAP, or LiDAR reconstruction score. The clip has no Car boxes.

| Input | All stacked returns | Old fast returns | Returns in current Cyclist boxes |
| --- | ---: | ---: | ---: |
| Ego aligned only | 60,451 | 5,688 | 1,616 |
| Old velocity-age deletion | 55,249 | 486 | 1,480 |
| Doppler radial shift | 60,451 | 5,688 | 2,258 |
| Radial shift plus age window | 58,644 | 3,881 | 2,007 |

For target `00564` alone, the Cyclist-box counts are **44, 40, 111, 78**
in the same order. Reversing the radial sign gave 1,497 across the 21 targets
and 39 for `00564`, supporting the selected sign in this clip. None of this
establishes better 3D detection. The shifted view also shows returns outside
boxes, so it cannot be used as a point-level geometry target on its own.

The reproducible per-frame training results are in
`reports/vod_doppler/training_preview_per_frame.csv`.

## Full validation split

The same parameters selected above were applied to all **1,296 official VoD
validation frames**. Validation labels were read only for this measurement.
All methods use the same raw radar scans and target frames. Historical source
scans occur in multiple overlapping target stacks, so the counts below are
stacked-return counts, not counts of independent radar observations.

| Radar stack | Total returns | In current Cyclist boxes | In current Car boxes | In current Pedestrian boxes |
| --- | ---: | ---: | ---: | ---: |
| Ego alignment only | 8,341,349 | 49,941 | 95,308 | 34,577 |
| Old velocity-age deletion | 7,635,074 | 47,567 | 93,223 | 29,854 |
| Doppler radial shift | 8,341,349 | **71,812** (+43.8%) | **101,416** (+6.4%) | **57,995** (+67.7%) |
| Radial shift plus 2 m age window | 7,896,901 | 56,669 (+13.5%) | 94,798 (-0.5%) | 55,861 (+61.6%) |

The radial shift increases Cyclist-box counts in 908 frames, decreases them in
109, and leaves them unchanged in 279. For Pedestrian boxes those counts are
812, 86, and 398. As a sign control, reversing the radial shift produced
47,929 Cyclist-box and 27,559 Pedestrian-box returns, below even the ego-only
counts. This supports the displacement sign without establishing that all
shifted points are physically correct.

The full [per-frame CSV](../reports/vod_doppler/validation_per_frame.csv) and
[frame 05072 comparison](../reports/vod_doppler/validation_05072.png) are saved
with the repository. In that illustrative frame, the Cyclist-box counts are
103, 77, 205, and 164 in table order. Some shifted returns remain outside
annotated boxes. Box concentration alone cannot measure false geometry, LiDAR
reconstruction quality, or 3D detection AP. The window discards useful Car
evidence on aggregate, so **radial-only** is the more informative first
reconstruction ablation; retain the window as a separate comparison.

### Interactive 3D inspection

Open [the self-contained ten-frame viewer](../reports/vod_doppler/interactive_10_frames.html)
in a browser. It includes the previously inspected `train:00544`,
`train:00564`, and `val:05072` frames, followed by seven validation IDs drawn
uniformly with seed 42: `00228`, `00051`, `03594`, `00501`, `00457`, `00285`,
and `00209`. Select a frame, drag to rotate all four synchronized radar panels,
wheel or pinch to zoom, and Shift-drag or use two fingers to pan. The history
slider reveals how old radar scans affect alignment. Toggles show current,
older slow, and older high-Doppler radar separately, alongside current-time
3D annotations and an optional calibrated clean LiDAR reference. The fourth
panel now shows **radial shift plus the observed faulty-LiDAR height gate**;
the clean LiDAR display does not set the gate. Refresh the file in your browser
if an older copy is already open. This viewer
uses a 0–50 m forward radar-coordinate crop and caps displayed clean LiDAR at
15,000 points per frame; box counts still use the full radar stacks.

To regenerate the file from locally available VoD stacks:

```bash
python -m scripts.interactive_vod_doppler_3d \
  --vod-root /path/to/view_of_delft_PUBLIC \
  --output reports/vod_doppler/interactive_10_frames.html \
  --fault-samples-root /path/to/full-scan-fault-samples \
  --previous train:00544 train:00564 val:05072 \
  --random-count 7 --seed 42
```

## Selected model input: radial shift and observed-LiDAR height gate

The new ray-diffusion trainer defaults to
`radar_20frames_verified_doppler_radial`. At **sample loading time**, it
transforms each radar stack to the current LiDAR frame, then keeps a radar
return only if its LiDAR-frame Z lies between the minimum and maximum Z of
the **observed faulty LiDAR** for that sample, inclusive. The clean LiDAR
teacher is never consulted for this input filter. The radial radar files on
disk remain unchanged, so distinct fault realizations of the same frame get
their own bounds. If fewer than two faulty LiDAR points survive, the gate is
skipped so a total LiDAR failure still has radar input. Training config,
checkpoints, and exports record whether the gate is enabled. The
`--no-radar-height-filter` switch permits a direct ablation.

Using the local `reconstruction_vod_radar5_unique` validation fault artifacts
as an **inspection sample** (the new full-scan training cache may differ), the
gate removes 1,798,505 / 8,341,349 radar returns (21.6%) over all 1,296
validation frames. It retains 71,761 / 71,812 returns inside annotated
Cyclist boxes, 101,304 / 101,416 inside Car boxes, and 57,980 / 57,995
inside Pedestrian boxes. In previously inspected frame `00457`, it keeps
4,932 / 6,707 radar returns and all 46 Cyclist, 72 Car, and 88 Pedestrian
box returns. The [per-frame height-filter CSV](../reports/vod_doppler/validation_height_filter.csv)
records those counts and the faulty-LiDAR Z bounds. This measures box
membership only: removed returns can still be valid surfaces outside object
boxes, and retained returns can still be ghosts.

## Generate on the professor machine

Update this repository on that machine first. Generate the full verified
20-scan stack for each split using `docs/verified_radar_cross_modal_encoders.md`
if those files are missing. Then:

```bash
BASE=/mnt/3D10B36523559581/Gianluca
REPO="$BASE/sensor-fusion-fault-localization"
PY="$BASE/svefusion-clean-cu118/bin/python"
VOD=/mnt/3D10B36523559581/View-of-Delft/view_of_delft_detection_PUBLIC/view_of_delft_PUBLIC
cd "$REPO"

for split in train val; do
  "$PY" -u -m scripts.generate_vod_doppler_radar \
    --vod-root "$VOD" --split "$split" \
    --output-variant radar_20frames_verified_doppler_radial \
    --frame-period-s 0.10022 --doppler-sign 1 \
    --num-workers 4
done
```

Generate `radar_20frames_verified_doppler_window` separately only for the
age-window ablation. The selected height gate is applied per faulty sample
by the model loader and does not require another radar directory. To measure
radar-box concentration for all validation frames on that machine:

```bash
"$PY" -m scripts.compare_vod_doppler_radar \
  --vod-root "$VOD" --split val --conditions ego radial \
  --output "$BASE/sensor_fusion_outputs/vod_doppler_validation_counts.csv"
```

For the reconstruction experiment, train two fresh runs with the same seed,
fault samples, geometry, epochs, and settings; vary only `--radar-variant`
between `radar_20frames_verified` and
`radar_20frames_verified_doppler_radial`, with the **same observed-LiDAR
height gate enabled** in both runs. Optionally include
`radar_20frames_verified_doppler_window` as a third ablation. Then export validation
reconstructions and run the **same** frozen LiDAR+radar detector, with its
usual radar input, on each. A radar concentration gain alone does not imply
improved reconstructed LiDAR or detector AP.

Reference: [DoppDrive, ICCV 2025](https://arxiv.org/html/2508.12330v1).
