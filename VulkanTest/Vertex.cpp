#define GLFW_INCLUDE_VULKAN // GLFW vulkan support (Vulkan is an API not window)



#include <GLFW/glfw3.h> // Windowing
#include <iostream> // IDK
#include <stdexcept> // Errors
#include <cstdlib> // STD lib
#include <vector> // For arrays
#include <optional> // For giving arrays no value until assigned to
#include <set>
#include <cstdint> // Necessary for UINT32_MAX
#include <algorithm> // Needed for min max
#include <glm/glm.hpp> // Needed for math functions like matrices etc
#include <glm/gtc/matrix_transform.hpp> // transformations
#include <glm/gtx/rotate_vector.hpp>
#include <chrono> // Time
#include <fstream> // Needed to load shaders
#include <array> // array stuff
#include <unordered_map> // used to check we aren't adding unnecessary verticies
#include <glm/gtx/hash.hpp> // used to hash
#include "Vertex.h"

// Tells vulkan what is in the struct (For this, Pos and Color, texcoords and normals)
// This function is needed to tell vulkan about how to use this struct
VkVertexInputBindingDescription Vertex::getBindingDescription() {
    VkVertexInputBindingDescription bindingDescription{};

    bindingDescription.binding = 0; // what binding the per-vertex data comes.
    bindingDescription.stride = sizeof(Vertex); // number of bytes between each entry
    bindingDescription.inputRate = VK_VERTEX_INPUT_RATE_VERTEX; // is it per-vertex or per-instance

    return bindingDescription;
}

std::array<VkVertexInputAttributeDescription, 4> Vertex::getAttributeDescriptions() {
    std::array<VkVertexInputAttributeDescription, 4> attributeDescriptions{};
    // Setup the pos
    attributeDescriptions[0].binding = 0; // what binding the per-vertex data comes.
    attributeDescriptions[0].location = 0; // This can be seen in the shader
    attributeDescriptions[0].format = VK_FORMAT_R32G32B32_SFLOAT; // The type of data. Use a 3d vector so we can use 3d models (as well as 2d)
    attributeDescriptions[0].offset = offsetof(Vertex, pos); // The offset between each value

    // Setup color
    attributeDescriptions[1].binding = 0; // what binding the per-vertex data comes.
    attributeDescriptions[1].location = 1; // This can be seen in the shader
    attributeDescriptions[1].format = VK_FORMAT_R32G32B32_SFLOAT; // We are sending it as a vec 3, so read it as a vec3 type float
    attributeDescriptions[1].offset = offsetof(Vertex, color); // offset of each value


    attributeDescriptions[2].binding = 0; // binding of per-vertex data
    attributeDescriptions[2].location = 2; // location in the shader
    attributeDescriptions[2].format = VK_FORMAT_R32G32_SFLOAT; // vec 2
    attributeDescriptions[2].offset = offsetof(Vertex, texCoord); // offsets for the value

    attributeDescriptions[3].binding = 0; // binding of per-vertex data
    attributeDescriptions[3].location = 3; // location in the shader
    attributeDescriptions[3].format = VK_FORMAT_R32G32B32_SFLOAT; // vec 3
    attributeDescriptions[3].offset = offsetof(Vertex, normal); // offsets for the value

    return attributeDescriptions; // return
}

bool Vertex::operator==(const Vertex& other) const {
    return pos == other.pos && color == other.color && texCoord == other.texCoord && normal == other.normal;
}

