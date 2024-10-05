CUDA Path Tracer
================

**University of Pennsylvania, CIS 5650: GPU Programming and Architecture, Project 3**

* Michael Rabbitz
  * [LinkedIn](https://www.linkedin.com/in/mike-rabbitz)
* Tested on: Windows 10, i7-9750H @ 2.60GHz 32GB, RTX 2060 6GB (Personal)

***Insert coolest demo image(s) here***

## Part 1: Introduction

This project takes an incomplete skeleton of a C++/CUDA path tracer, transforms it to a functional state by implementing core features, then enhances it through physically-based visual improvements, mesh enhancements, and performance optimizations.

### Path Tracing Overview
Path tracing is a sophisticated rendering technique in computer graphics designed to achieve photorealistic images by accurately simulating the behavior of light in a scene. This technique flips the conventional perspective on light: instead of light traveling from sources to the eye, rays are cast from the camera into the scene, exploring how light interacts with the surfaces of objects in the scene and determining how those surfaces are illuminated. The Bidirectional Scattering Distribution Function (BSDF) plays a key role in this process, governing how light scatters when it hits a surface, accounting for both reflection and refraction.

As rays bounce off surfaces, they generate multiple reflections and/or refractions until they reach a light source or exit the scene. Path tracing employs Monte Carlo integration to estimate pixel colors by averaging many random samples, enhancing image quality at the cost of increased rendering time. A standout feature of path tracing is its ability to simulate global illumination, capturing the complex interplay of light as it bounces between surfaces. While it delivers high-quality results and effectively handles various materials and lighting conditions, path tracing can be computationally demanding and may introduce noise, which can be mitigated by increasing the sample count.

|Global Illumination = Direct Illumination + Indirect Illumination|
|:--:|
|<img src="img/global_illumination.png" alt="global_illumination" height="200"> <tr></tr>|
|*Left: Light Ray* ***directly*** *illuminating the point on the floor from the viewer's perspective via no intermediate bounce(s)* <tr></tr>|
|*Right: Light Ray* ***indirectly*** *illuminating the point on the floor from the viewer's perspective via an intermediate bounce*|

|BSDF = BRDF + BTDF|
|:--:|
|<img src="img/bsdf.png" alt="bsdf" height="400"> <tr></tr>|
|Source: [Wikipedia](https://en.wikipedia.org/wiki/Bidirectional_scattering_distribution_function)|

|"Scattering" in Path Tracing using BSDFs|
|:--:|
|<img src="img/path_tracing.png" alt="path_tracing" height="200"> <tr></tr>|
|This series illustrates a single ray cast from the eye in path tracing (yellow ray) as it focuses on a point on the floor. It demonstrates how global illumination at that point is achieved through multiple bounces of the ray, interacting with surfaces based on their BSDFs. Each bounce scatters additional rays according to the surface's BSDF, creating many ray paths. The contribution to the illumination of the initial point decreases with each bounce and ultimately concludes when all ray paths either hit a light source, exit the scene, or reach the bounce/depth limit.|

## Part 2: Core Features Implemented

### Ideal Diffuse (Lambertian) BSDF Evaluation
|Lambertian Reflectance|
|:--:|
|<img src="img/lambertian_diffuse.PNG" alt="lambertian_diffuse" height="200"> <tr></tr>|
|Source: [Wikipedia](https://en.wikipedia.org/wiki/Lambertian_reflectance)|

***Insert cool demo image(s) here***

### Perfect Specular Reflection (Mirrored) BSDF Evaluation
|Mirrored Reflectance|
|:--:|
|<img src="img/specular_reflection.png" alt="specular_reflection" height="200"> <tr></tr>|
|Source: [Wikipedia](https://en.wikipedia.org/wiki/Specular_reflection)|

***Insert cool demo image(s) here***

### Stochastic Sampled Antialiasing


### Path Continuation/Termination using Stream Compaction


### Path Segments contiguous in memory by Material before BSDF Evaluation and Shading


## Part 3: Physically-based Visual Improvements

### Dielectric BSDF Evaulation (Refraction, Fresnel effects)

### Metal BSDF Evaulation (Roughness, Fresnel effects)

## Part 4: Mesh Enhancements

### OBJ Loader and Renderer (Ray-Triangle Intersection)

## Part 5: Performance Optimizations

### Bounding Volume Hierarchy (BVH)
**Construction uses the Surface Area Hueristic (SAH) for Axis-Aligned Bounding Box (AABB) subdivision**












