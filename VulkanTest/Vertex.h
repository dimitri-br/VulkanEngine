#pragma once

// This holds vertex data - namely the position and color of the vertex
struct Vertex {
public:
    glm::vec3 pos; // pos of vertex
    glm::vec3 color; // color
    glm::vec2 texCoord; // Texture coordinate mapping
    glm::vec3 normal; // normals (lighting)

    // This function is needed to tell vulkan about how to use this struct
    static VkVertexInputBindingDescription getBindingDescription();

    static std::array<VkVertexInputAttributeDescription, 4> getAttributeDescriptions();

    // This is required to make the Vertex struct work in a map
    bool operator==(const Vertex& other) const;
};

