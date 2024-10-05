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

|Global Illumination = Direct Illumination + Indirect Illumination|
|:--:|
|![global_illumination](img/global_illumination.png) <tr></tr>|
|*Left: Light Ray* ***directly*** *illuminating the point on the floor from the viewer's perspective via no intermediate bounce(s)* <tr></tr>|
|*Right: Light Ray* ***indirectly*** *illuminating the point on the floor from the viewer's perspective via an intermediate bounce*|

|BSDF = BRDF + BTDF|
|:--:|
|![bsdf](img/bsdf.png) <tr></tr>|
|Source: [Wikipedia](https://en.wikipedia.org/wiki/Bidirectional_scattering_distribution_function)|

|"Scattering" in Path Tracing using BSDFs|
|:--:|
|![path_tracing](img/path_tracing.png) <tr></tr>|
|This series illustrates a single ray cast from the eye in path tracing (yellow ray) as it focuses on a point on the floor. It demonstrates how global illumination at that point is achieved through multiple bounces of the ray, interacting with surfaces based on their BSDFs. Each bounce scatters many rays according to the surface's BSDF, creating many additional ray paths. The contribution to the illumination of the initial point decreases with each bounce until a given ray path either reaches a light source or the trace depth limit is reached.|













