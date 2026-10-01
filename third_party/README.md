# Third-party dependencies

`tools/openpcdet/bootstrap_openpcdet.sh` installs the official OpenPCDet
repository here as an ignored, isolated dependency at commit
`233f849829b6ac19afb8af8837a0246890908755`. OpenPCDet model code is not copied
into the thesis package.

`third_party/SVEFusion` is a Git submodule pinned to the adapted SVEFusion
detector. In an existing clone, run `git submodule update --init
third_party/SVEFusion` after pulling this repository. Its VoD comparison
workflow is in `third_party/SVEFusion/docs/vod_fault_comparison.md`. The
submodule uses the public fork at
`https://github.com/Gianluca3103/SVEFusion`.
