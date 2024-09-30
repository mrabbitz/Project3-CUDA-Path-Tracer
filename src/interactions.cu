#include "interactions.h"

__host__ __device__ glm::vec3 calculateRandomDirectionInHemisphere(
    const glm::vec3& normal,
    thrust::default_random_engine& rng)
{
    thrust::uniform_real_distribution<float> u01(0, 1);

    float up = sqrt(u01(rng));          // cos(theta) - taking the square root to ensure that the directions closer to the surface normal are favored and we dont get too many samples near the edges of the hemisphere
    float over = sqrt(1 - up * up);     // sin(theta) = sqrt(1 - cos(theta) * cos(theta))
    float around = u01(rng) * TWO_PI;   // angle in the plane around the normal

    // Find a direction that is not the normal based off of whether or not the
    // normal's components are all equal to sqrt(1/3) or whether or not at
    // least one component is less than sqrt(1/3). Learned this trick from
    // Peter Kutz.
    //
    // Why do this?
    // We need two perpendicular vectors to the normal to generate a random direction in the hemisphere
    // This avoids degenerate cases where the cross product would be zero if the normal and the chosen vector were parallel

    glm::vec3 directionNotNormal;
    if (abs(normal.x) < SQRT_OF_ONE_THIRD)
    {
        directionNotNormal = glm::vec3(1, 0, 0);
    }
    else if (abs(normal.y) < SQRT_OF_ONE_THIRD)
    {
        directionNotNormal = glm::vec3(0, 1, 0);
    }
    else
    {
        directionNotNormal = glm::vec3(0, 0, 1);
    }

    // Use not-normal direction to generate two perpendicular directions
    glm::vec3 perpendicularDirection1 =
        glm::normalize(glm::cross(normal, directionNotNormal));
    glm::vec3 perpendicularDirection2 =
        glm::normalize(glm::cross(normal, perpendicularDirection1));

    return up * normal
        + cos(around) * over * perpendicularDirection1
        + sin(around) * over * perpendicularDirection2;
}

__host__ __device__ void scatterRay(
    PathSegment& pathSegment,
    const glm::vec3& intersect_point,
    const glm::vec3& normal,
    const Material& m,
    thrust::default_random_engine& rng)
{
    glm::vec3 scatter_direction;
    glm::vec3 color;

    if (m.hasReflective && m.hasRefractive)
    {

    }
    else if (m.hasReflective)
    {
        // Pure Specular Reflection
        scatter_direction = glm::reflect(pathSegment.ray.direction, normal);
        color = m.color;
    }
    else if (m.hasRefractive)
    {

    }
    else
    {
        // Lambertian Diffuse
        scatter_direction = calculateRandomDirectionInHemisphere(normal, rng);

        // No need to scale the reflected color by abs(cos(theta))
        // since calculateRandomDirectionInHemisphere biases the sample directions toward the normal of the surface by sampling the hemisphere cosine-weighted.
        // This implicitly incorporates the cosine term into the light scattering model.
        color = m.color;
    }

    pathSegment.ray.direction = glm::normalize(scatter_direction);
    pathSegment.ray.origin = intersect_point + .0001f * pathSegment.ray.direction;
    pathSegment.color *= color;
}
