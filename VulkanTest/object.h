#pragma once
#include "transform.h"
#include "material.h"
#include "Vertex.h"
#include "btBulletDynamicsCommon.h"


// Holds descriptor set, material data and vertex data. Will be extended
struct Object {
public:
    int instance = 0; // Obejct instance
    // Vertex and index buffers must be stored as buffers on the GPU, as they need to be read by the device.
    std::vector<Vertex> vertices;
    std::vector<uint32_t> indices;
    VkBuffer vertexBuffer;
    VkDeviceMemory vertexBufferMemory;
    VkBuffer indexBuffer;
    VkDeviceMemory indexBufferMemory;


    VkImage textureImage; // this objects texture
    VkDeviceMemory textureImageMemory; // this objects texture memory
    VkImageView textureImageView; // in order to access the texture images, we need to access them through views.
    VkSampler textureSampler; // this objects sampler

    VkImage normalImage; // this objects texture
    VkDeviceMemory normalImageMemory; // this objects texture memory
    VkImageView normalImageView; // in order to access the texture images, we need to access them through views.
    VkSampler normalSampler; // this objects sampler

    // Descriptor sets specific to this object (So we can customize transformations and materials etc)
    std::vector<VkDescriptorSet> descriptorSets;

    // materials specific to this object
    std::vector<VkBuffer> materialBuffers;
    std::vector<VkDeviceMemory> materialBuffersMemory;

    std::vector<VkBuffer> transformBuffers;
    std::vector<VkDeviceMemory> transformBuffersMemory;

    // Must be locally defined so we can pass it to the shader
    Transform transform;

    // Hardcoded values in case we need to use them directly
    glm::vec3 position;
    glm::vec3 rotation;
    glm::vec3 scale;

    Material mat;

    std::string model_path = "./models/model.obj"; // Object model
    std::string texture_path = "./textures/texture.png"; // Object texture
    std::string normal_path = "./textures/normal.png"; // Object texture

    btCollisionShape* collisionShape;

    btDefaultMotionState* motionstate;

    btRigidBody* rigidBody;


    void init(std::string model, std::string texture, std::string normal, Material material, Transform trans);
    void setupPhysics(float mass, btCollisionShape* collider);

    Transform getTransform();
};







