# Changelog

## 0.3.0

- Add render_quality: efficiency, normal (default), quality with0,1,10 post-step updates; preserve legacy plans on resume.
- Record selected and actual rendering policy in dataset evidence.
- Add isaac_dataset_configure for persistent user-provided paths and runtime limits; read unconfigured installs without guessing an Isaac interpreter.
- Support clone plus install.sh and native Hermes Git installation, with optional first-use configuration.
- Keep machine configuration, assets and experimental outputs outside the distributable source.
- Maintain an independent repository for plugin source and product documentation; keep validation reports in the private development project.

## 0.2.0

- Add reusable camera_count1–8, shared render groups, tail handling and durable per-image recovery.

## 0.1.0

- Independent semantic-face sampling, RGB/pose capture, grouped splits, bounded process batches and checkpoint recovery.
