# Roadmap

## Notebooks & Tutorials

Task-focused, minimal notebooks — each one answers a single "how do I..." question.

### Existing / In Progress

- [x] ML training data generation guide (`examples/ml_training_guide.ipynb`)
- [x] Satellite comparison (`examples/satellite_comparison.ipynb`)
- [ ] Earthquake focal mechanisms (`examples/earthquake_mechanisms.ipynb`)
- [ ] Time series anomaly detection (`examples/timeseries_anomaly_detection.ipynb`)

### Planned

- [ ] **Synthetic data for DL anomaly detection** — generate labeled normal/anomalous InSAR frames for autoencoder or VAE training
- [ ] **Synthetic data for semantic segmentation** — produce displacement maps with pixel-level binary masks for U-Net style models
- [ ] **Synthetic data for computer vision** — create image-like wrapped interferograms (fringe patterns) for classification and object detection tasks
- [ ] **Source mechanism exploration** — interactive notebook (ipywidgets) to see how strike, dip, rake, depth, and magnitude affect the displacement field in real time
- [ ] **Noise & atmosphere effects** — show how Gaussian noise levels and orbital ramps degrade signal, with SNR analysis
- [ ] **Multi-satellite comparison for ML** — generate the same earthquake seen by different satellites, explore how band/geometry affects model generalization
- [ ] **Batch dataset generation** — end-to-end workflow: generate N samples, save to disk (GeoTIFF/NetCDF), load into PyTorch/TensorFlow DataLoader
- [ ] **Quick start (5-minute intro)** — minimal notebook: install, generate one interferogram, plot it

## Package Features

### Near-term

- [ ] Finite fault (distributed slip) forward model for larger earthquakes (Mw > 6.5)
- [ ] Atmospheric phase screen (turbulence + stratified delay)
- [ ] Topographic phase contribution
- [ ] Google Colab one-click badges on all notebooks

### Medium-term

- [ ] Multi-source (multiple earthquakes in one scene)
- [ ] Postseismic viscoelastic relaxation (simple Maxwell rheology)
- [ ] Integration with real InSAR catalogs (USGS, GCMT) for realistic parameter sampling
- [ ] GPU-accelerated batch generation (CuPy backend)

### Long-term / Ideas

- [ ] Okada (1985) rectangular dislocation model as alternative to Davis point source
- [ ] Plugin interface for custom deformation models
- [ ] Benchmarking suite against analytical solutions
- [ ] Integration with ISCE / MintPy for hybrid real+synthetic workflows
