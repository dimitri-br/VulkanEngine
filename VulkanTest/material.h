#pragma once
// Material information - holds info for the shader
struct Material {
    alignas(16) float shininess;
    alignas(16) glm::vec3 color;
};