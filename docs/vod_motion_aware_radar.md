# Motion-aware VoD 20-scan radar stack

The verified 20-scan stack aligns the radar sensor across ego motion, but an
independently moving target appears at its old locations. The new
`radar_20frames_verified_motion_aware` variant keeps every current return and
one prior scan of high-Doppler returns. Older returns with absolute
**compensated radial velocity** at least 1.0 m/s are removed. Returns below
that threshold retain the full 20-scan history. Each retained point keeps its
original XYZ, RCS, Doppler values, and ordinal scan age; there is no guessed
object-motion warp.

The threshold and one-scan history are starting values, not a validated VoD
optimum. Nearly tangential movers can have little radial velocity and remain
in the long history; noisy Doppler can also shorten static history. The source
contains scan ordinal but no elapsed seconds, so the gate operates in scans.
This is a conservative trail reduction, not full object-motion compensation.

Generate the separate variant on the professor machine before training. The
original `radar_20frames_verified` files stay available as a baseline:

```bash
BASE=/mnt/3D10B36523559581/Gianluca
REPO="$BASE/sensor-fusion-fault-localization"
PY="$BASE/svefusion-clean-cu118/bin/python"
VOD=/mnt/3D10B36523559581/View-of-Delft/view_of_delft_detection_PUBLIC/view_of_delft_PUBLIC
cd "$REPO"
for split in train val test; do
  "$PY" -u -m scripts.generate_vod_accumulated_radar \
    --vod-root "$VOD" --stack-sizes 20 --split "$split" \
    --alignment-source official-5 \
    --output-suffix verified_motion_aware --motion-aware \
    --moving-velocity-threshold-mps 1.0 --moving-max-age-scans 1 \
    --isolate-split-history --num-workers 4
done
```

This is an experimental ablation. The ray diffusion training command defaults
to the full `radar_20frames_verified` stack so that moving-object evidence is
preserved. To train with this truncated stack, pass
`--radar-variant radar_20frames_verified_motion_aware` explicitly. The
checkpoint records the selected variant, and export uses the same variant by
default.

For a local preview, 21 training targets produced 60,451 raw returns. The gate
removed 5,202 older high-Doppler returns (8.6%). In frame `00564`, it kept 4,536 of
4,971 returns, including all 279 current-scan returns. These counts verify
the implementation, not that detection or reconstruction has improved; compare
held-out metrics and visualizations between variants.

The preview's boxes describe objects at the **target frame time**. An older
radar return should be checked against annotations from its source frame, not
only against the target-frame boxes. Across 21 preview targets, 1,060 of the
5,202 filtered returns were inside their source-time 3D boxes, versus only 138
inside target-time boxes. With a 1 m margin on each box face, 3,597
were near source-time boxes. This is similar to the 348 of 507 current-scan
high-Doppler returns near current boxes. A high compensated radial velocity is
therefore a useful motion cue, not an object label; clutter, multipath,
unannotated motion, and radar position uncertainty remain.

For the 17 preview targets with a complete recent five-scan history, all
20,761 corresponding points matched the released VoD five-scan coordinates
within 0.000008 m. This checks the recent ego alignment. It does not prove
that the older 15 scans or individual moving targets are at their current
positions.

An audit against **source-time** boxes in the 21-frame preview found that this
gate removes 1,060 of 3,182 valid returns inside Cyclist/bicycle/rider boxes
(33.3%), and 3,573 of 13,320 returns within a 1 m margin of those boxes
(26.8%). In target frame `00564`, it removes 94 of 242 returns inside those
boxes (38.8%). The 21 target stacks share source scans, so the aggregate is
not 21 independent scenes. No Car boxes occur in this local preview. These
counts justify keeping the full stack as the training default; the optional
gate should not be called motion compensation or treated as an improvement.
