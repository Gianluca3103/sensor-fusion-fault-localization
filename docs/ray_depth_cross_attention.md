# Radar-only ray-depth relationship blueprint

The deployed relationship model predicts a LiDAR ray-depth blueprint from the
verified, motion-adjusted VoD 20-scan radar stack. Radar input has seven
fields: XYZ in the current LiDAR frame, RCS, radial Doppler, compensated
Doppler, and discrete scan ordinal. The scan ordinal is not elapsed seconds.
The blueprint is a conditioning feature map, **not** a point cloud.

## Training paths

1. Candidate queries use calibrated LiDAR ray directions, three distinct
   radar-derived depths, and two uniform fallback depths. Uniform candidates
   provide query locations but are never evidence on their own. Neither clean
   nor faulty LiDAR proposes a depth.
2. A training-only clean-LiDAR encoder and local cross-attention teacher learn
   first-return/no-return and depth on those same queries. Fine clean points
   are limited to the current ray tile for tractable attention; occupied 3D
   grid tokens still give broader context.
3. A separate radar encoder and local cross-attention student use radar tokens
   at two spatial scales. The student is supervised against clean first
   returns and, on radar-supported clean surfaces, against detached features
   from the pretrained clean teacher. The teacher is frozen in this phase.
4. The student alone outputs per-candidate features, evidence/null weights,
   bounded depth residuals, and first-return/no-return logits. These are the
   unchanged five-slot input interface of the range-view diffusion model.

The student receives no clean or faulty LiDAR. The observed-LiDAR arguments
in its call signature are retained for compatibility with the existing
diffusion trainer, but they are validated and ignored. Diffusion still uses
faulty LiDAR directly to retain observed returns and avoid adding points on
observed rays. The clean teacher is used only in stage-one training and is
never instantiated or called by diffusion or export.
The optional radar height filter, which derives bounds from faulty LiDAR, is
disabled by default in both stages to keep this relationship path radar-only.

The radar-only output is identical in stage-one training, diffusion training,
and inference. A new checkpoint carries
`relationship_version=radar_only_clean_teacher_v1`; older checkpoints cannot
initialize this student because the input path and candidate slots changed.

## Supervision and limitations

The clean scan provides nearest first-return depth and no-return targets on
calibrated rays. The student is trained to abstain when no radar-supported
candidate lies within the configured 3 m residual bound. Feature
distillation compares only radar-supported candidates near a clean return
for which the clean teacher has local evidence. It does not force a match to
an arbitrary clean point paired with an individual radar reflection.

Validation reports candidate coverage, depth-aware precision/recall/F1,
feature loss, and how many feature pairs contributed. Compare these on full
training and validation splits to detect a large generalization gap. These
metrics do not establish the correctness of generated clouds or downstream
detection; those require separate evaluation. Softmax return probabilities
are not calibrated confidence without a held-out calibration check.

See `docs/staged_ray_depth_training.md` for the teacher-first command and
`docs/radar_gated_range_diffusion.md` for the unchanged reconstruction head.
