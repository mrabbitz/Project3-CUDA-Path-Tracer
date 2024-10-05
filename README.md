CUDA Path Tracer
================

**University of Pennsylvania, CIS 5650: GPU Programming and Architecture, Project 3**

* Michael Rabbitz
  * [LinkedIn](https://www.linkedin.com/in/mike-rabbitz)
* Tested on: Windows 10, i7-9750H @ 2.60GHz 32GB, RTX 2060 6GB (Personal)

***Insert coolest demo image(s) here***

## Part 1: Introduction

This project takes an incomplete skeleton of a C++/CUDA path tracer and implements physically-based visual improvements, mesh enhancements, and performance optimizations.

### Path Tracing Overview
Path tracing is a sophisticated rendering technique in computer graphics designed to achieve photorealistic images by accurately simulating the behavior of light in a scene. This technique flips the conventional perspective on light: instead of light traveling from sources to the eye, rays are cast from the camera into the scene, exploring how light interacts with the surfaces of objects in the scene and determining how those surfaces are illuminated. The Bidirectional Scattering Distribution Function (BSDF) plays a key role in this process, governing how light scatters when it hits a surface, accounting for both reflection and refraction.

As rays bounce off surfaces, they generate multiple reflections and refractions until they reach a light source or exit the scene. Path tracing employs Monte Carlo integration to estimate pixel colors by averaging many random samples, enhancing image quality at the cost of increased rendering time. A standout feature of path tracing is its ability to simulate global illumination, capturing the complex interplay of light as it bounces between surfaces. While it delivers high-quality results and effectively handles various materials and lighting conditions, path tracing can be computationally demanding and may introduce noise, which can be mitigated by increasing the sample count.

|BSDF = BRDF + BTDF|"Scattering" in Path Tracing using BSDF|
|:--:|:--:|
|![bsdf](img/bsdf.png)|![path_tracing](img/path_tracing.png)|

|BSDF = BRDF + BTDF|
|:--:|
|![bsdf](img/bsdf.png)|

|Global Illumination = Direct Illumination + Indirect Illumination|
|:--:|
|![global_illumination](img/global_illumination.png)|

|"Scattering" in Path Tracing|
|:--:|
|![path_tracing](img/path_tracing.png)|













