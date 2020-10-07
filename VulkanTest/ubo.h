#pragma once

// This holds UBO data, like the model pos, view point and projection
struct UniformBufferObject {
    alignas(16) glm::mat4 model;
    alignas(16) glm::mat4 view;
    alignas(16) glm::mat4 proj;
    alignas(16) glm::mat4 lightSpace;
    alignas(16) glm::vec3 lightPos;
    alignas(16) glm::vec3 camPos;
};


// Holds our light info for the shadowmap
struct uboOffscreenVS {
    alignas(16) glm::mat4 MVP;
    alignas(16) std::array<glm::mat4, 10> objects;
};