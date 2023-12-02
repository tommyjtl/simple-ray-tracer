# Simple Ray Tracer

A very simple Ray Tracer implemented with Python.

## Features

### Implemented

- [x] Handles the input file keywords png, color, sphere, and one sun, with proper handling of sRGB gamma.
- [X] Implements the ray-sphere intersection algorithm.
- [X] Implement shadows with shadow rays, including preventing shadow acne.
- [X] Exposure control

### To be implemented

- [ ] Multiple light sources (suns) and shadows
- [ ] Move and rotates the camera
- [ ] Fisheye and panoramic cameras
- [ ] Plane-intersection
- [ ] Triangle-intersection
- [ ] Texture mapping
- [ ] IOR and refraction
- [ ] Anti-aliasing
- [ ] DoF
- [ ] Global illumination
- [ ] BVH

## Usage

```bash
python main.py <input_file>

# Example
python main.py raytracer-files/ray-sphere.txt
```

Below is a list of implemented input files. Or run `bash batch_test.sh` to test all of them.

```text
sphere
sun
color
overlap
behind
shadow-basic
expose1
expose2
suns
shadow-suns
```

## References

- [CS 418 SP 2023](https://cs418.cs.illinois.edu/website/mps/raytracer.html)
- [CS 418 – Raytracing](https://cs418.cs.illinois.edu/website/text/rays.html)