## License

This project is licensed under the **GNU General Public License v3.0 or later**.

See the [LICENSE](LICENSE) file for the full text.

If you use this software in your research, please cite:

 
@software{pbrAudio,
  author = {Malcom3D},
  title = {pbrAudio: Physically Audio Synthesis},
  year = {2025},
  url = {https://github.com/malcom3d/pbrAudio}
}

## pbrAudioShaders

Physically based rendered audio shaders.


### Physically plausible collisions sound for rigid body simulation.

#### Features

- **Physically-based reverse engeniering of rigidbody animation**: Uses Hertzian contact theory for accurate collisions modeling
- **Multi-object, multi-collision support**: Handles complex scenes with multiple interacting objects
- **Multiple collision type support**: Handles impact, scraping, sliding and rotation.
- **Material-aware synthesis**: Considers material properties (Young's modulus, density, damping, etc.)
- **Modal synthesis**: Generates audio from object vibration modes

References:
- https://pure.tue.nl/ws/portalfiles/portal/194387031/IPO_Rapport_1226.pdf
- https://www.cs.ubc.ca/labs/lci/papers/docs2001/van-foleyautomatic.pdf
- https://ccrma.stanford.edu/~rmichon/publications/doc/ICMC17-mtf.pdf
- https://github.com/grame-cncm/faust/tree/master-dev/tools/physicalModeling to generate audio physical modal model from 3D mesh


### Physically plausible acceleration noise for rigid body world simulation.

References:
- https://www.cs.cornell.edu/projects/Sound/proxy/FasterAccelerationNoise_SCA2012.pdf
- https://graphics.stanford.edu/courses/cs448z-21-spring/stuff/PAN_typoFix.pdf


### Physically plausible fracture sound for rigid body world simulation.

References [not confirmed]:
- https://www.cs.cornell.edu/projects/FractureSound/files/fractureSound.pdf


### Physically plausible collisions sound for nonlinear thin-shell simulation.

References [not confirmed]:
- https://www.cs.cornell.edu/projects/HarmonicShells/HarmonicShells09_large.pdf


### Physically plausible ground-sound model sound synthesis

References [not confirmed]:
- https://graphics.stanford.edu/papers/ground/assets/GroundSound_Combined.pdf


### Physically plausible crumpling sounds for 3D mesh.

References [not confirmed]:
- https://www.cs.columbia.edu/cg/crumpling/crumpling-sound-synthesis-siggraph-asia-2016-cirio-et-al.pdf for crumpling Sound


### Physically plausible explosion, fire and combustion sounds for 3D simulation.

References [not confirmed]:
- https://www.cs.cornell.edu/projects/Sound/fire/FireSound2011.pdf for fire and explosion sound
- https://onlinelibrary.wiley.com/doi/epdf/10.1002/cav.1970


### Physically plausible swings object, aeolian and aeroacoustic sound effects.

References [not confirmed]:
- https://pdfs.semanticscholar.org/345f/1f4b15366ad1be2dd083975d87cf579ea2b1.pdf
- https://mdpi-res.com/d_attachment/applsci/applsci-07-01177/article_deploy/applsci-07-01177.pdf

### Physically plausible sounds for fluid simulation.

References [not confirmed]:
- https://haonancheng.cn/attaches/2019%20GMOD%20Liquid-solid%20interaction%20sound%20synthesis.pdf
 - https://github.com/kangruix/FluidSound
- https://graphics.stanford.edu/papers/coupledbubbles/assets/coupledbubbles.pdf
- https://www.cs.cornell.edu/projects/Sound/bubbles/bubbles.pdf
 - https://www.cs.cornell.edu/projects/Sound/bubbles/bubbleDemo.py
- https://www.cs.cornell.edu/projects/HarmonicFluids/harmonicfluids.pdf
 - https://github.com/ashab015/Harmonic-Fluids
- https://gamma.cs.unc.edu/SoundingLiquids/soundingliquids.pdf


### Physically plausible sounds for electical discarge simulation.

References [not confirmed]:
- https://www.art-science.org/journal/v7n4/v7n4pp145/artsci-v7n4pp145.pdf


### Physically plausible sounds for thunder simulation.

References [not confirmed]:
- https://arxiv.org/pdf/2204.08026
 - https://github.com/bineferg/Thunder-Synthesis

 - with physically based animation of Lightning like:
  - https://gamma.cs.unc.edu/LIGHTNING/lightning.pdf
  - https://graphics.stanford.edu/~wicke/publications/lightning/bwg-asoed-06.pdf
  - https://storage.googleapis.com/pirk.io/papers/Amador-Herrera.etal-2024-Thunderstruck.pdf


### Physically plausible sounds for rain simulation.

References [not confirmed]:
- https://haonancheng.cn/attaches/2019%20SIGGRAPH.pdf


### Physically plausible sounds for physics-based cloth simulation.  

References [not confirmed]:
- https://www.cs.cornell.edu/projects/Sound/cloth/cloth2012.pdf


### Inverse-Foley synchronized rigid-body motions to sound.

References [not confirmed]:
- https://www.cs.cornell.edu/projects/Sound/ifa/ifa.pdf


### 3D simulation of acoustic shock waves propagation.

References [not confirmed]:
- https://theses.hal.science/tel-01360574/document


### Physically based cartoonized synth. 

References [not confirmed]:
- https://github.com/SkAT-VG/SDT


### Physically-based auralization of railway rolling noise synthesis

References [not confirmed]:
- https://pub.dega-akustik.de/ICA2019/data/articles/000819.pdf
- https://www.researchgate.net/profile/Reto-Pieren/publication/317341926_Auralization_of_railway_noise_Emission_synthesis_of_rolling_and_impact_noise/links/5939d68f458515320632d429/Auralization-of-railway-noise-Emission-synthesis-of-rolling-and-impact-noise.pdf

### Large scene memory footprint optimazion with eigenmode compression for modal sound model.

References [not confirmed]:
- https://www.cs.cornell.edu/projects/Sound/modec/modec.pdf
