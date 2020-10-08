#pragma once
#include "transform.h"
#include "material.h"
#include "Vertex.h"
#include "object.h"
#include "Light.h"
#include "ubo.h"

const float YAW = -90.0f;
const float PITCH = 0.0f;

class Renderer
{
public:
    void run();

private:
    VkInstance instance;
    // No need to cleanup as it is done implicitly with VkInstance.
    VkPhysicalDevice physicalDevice = VK_NULL_HANDLE;

    VkDevice device;

    VkQueue graphicsQueue;

    VkSwapchainKHR swapChain;
    std::vector<VkImage> swapChainImages; // array of swapchain images. See the end of create swapchain to see how to get these images.
    VkFormat swapChainImageFormat;
    VkExtent2D swapChainExtent;

    VkCommandPool commandPool; // Stores all the commands to be sent to the GPU. Manages memory and allocation

    std::vector<VkImageView> swapChainImageViews; // Image views are how you modify an image in the swapchain. It is literally a view onto the image.

    std::vector<VkFramebuffer> swapChainFramebuffers; // Stores the images from the framebuffer

    VkFramebuffer offscreenFramebuffer;
    std::vector <VkDescriptorSet> offscreenSet;

    std::vector<VkCommandBuffer> commandBuffers; // Stores every command. They're implicitly destroyed, so no need to cleanup

    std::vector<VkCommandBuffer> shadowCommandBuffers;

    // Needed to interface with the GLFW window
    VkSurfaceKHR surface;

    // present queue will present to the surface/window
    VkQueue presentQueue;

    // Tells vulkan about what attachments to use (Like color, stencil etc)
    VkRenderPass renderPass;
    VkRenderPass offscreenRenderPass;






    VkSampleCountFlagBits msaaSamples = VK_SAMPLE_COUNT_1_BIT; // Multisampling. 1 bit is equivillant to nothing

    // So we can apply MSAA to the final image (Smooth out jagged lines). This color image is what gets sent to the GPU
    VkImage colorImage;
    VkDeviceMemory colorImageMemory;
    VkImageView colorImageView;


    VkDescriptorSetLayout descriptorSetLayout;


    std::vector<VkDescriptorPool> descriptorPool;


    // Pipeline layouts can be used to send uniform data to shaders. This can be anything from a transformation matrix or textures for a fragment shader.
    VkPipelineLayout pipelineLayout;


    std::vector<Object> objects;


    // This is stored as an array so if one buffer is in-flight, we don't modify it by mistake.
    std::vector<VkBuffer> uniformBuffers;
    std::vector<VkDeviceMemory> uniformBuffersMemory;

    std::vector<VkBuffer> shadowBuffers;
    std::vector<VkDeviceMemory> shadowBuffersMemory;




    // We must create our own depth image to attach to the graphics pipeline
    VkImage depthImage;
    VkDeviceMemory depthImageMemory;
    VkImageView depthImageView;



    VkPipeline graphicsPipeline;
    VkPipeline offscreenPipeline;

    VkImage shadowImage;
    VkDeviceMemory shadowImageMemory;
    VkImageView shadowImageView;
    VkSampler shadowSampler;

    uint32_t mipLevels;


    GLFWwindow* window;

    VkDebugUtilsMessengerEXT debugMessenger;

    // Used to asyncronously render images (as we need to safely read it from the GPU). In list form so we can submit frames in flight
    std::vector<VkSemaphore> imageAvailableSemaphores;
    std::vector<VkSemaphore> renderFinishedSemaphores;
    std::vector<VkFence> inFlightFences; // Fences are CPU-GPU synchronization, where the GPU waits on the CPU
    std::vector<VkFence> imagesInFlight; // This makes sure if we push a frame in flight, we don't render it twice when its already in flight.
    size_t currentFrame = 0; // current frame we are on
    bool framebufferResized = false; // was the frame resized

    // Camera settings

    // Camera rotation
    glm::vec3 cameraPos = glm::vec3(0.0f, 1.0f, 5.0f);
    glm::vec3 cameraFront = glm::vec3(0.0f, 0.0f, -1.0f);
    glm::vec3 cameraUp = glm::vec3(0.0f, 1.0f, 0.0f);

    float yaw = YAW;
    float pitch = PITCH;

    float deltaTime = 0.0f;

    std::chrono::steady_clock::time_point lastFrameTime = std::chrono::high_resolution_clock::now();


    int cube_index = 0;


    // Depth bias (and slope) are used to avoid shadowing artifacts
    // Constant depth bias factor (always applied)
    float depthBiasConstant = 1.25f;
    // Slope depth bias factor, applied depending on polygon's slope
    float depthBiasSlope = 1.75f;

    
    uboOffscreenVS depthMVP;
    // (Z, X, Y)

    //Light, eg sun
    Light light;

    // This is stored as an array so if one buffer is in-flight, we don't modify it by mistake.
    std::vector<VkBuffer> offscreenBuffers;
    std::vector<VkDeviceMemory> offscreenBuffersMemory;


    bool recreateDescriptorSets = true;

    struct QueueFamilyIndices {
        std::optional<uint32_t> graphicsFamily;
        std::optional<uint32_t> presentFamily; // Make sure the device can actually present

        bool isComplete() {
            return graphicsFamily.has_value() && presentFamily.has_value();
        }
    };

    // Needed to check what capabilities it has - as it may not support what we need (Eg, swapchain num and color formats)
    struct SwapChainSupportDetails {
        VkSurfaceCapabilitiesKHR capabilities;
        std::vector<VkSurfaceFormatKHR> formats;
        std::vector<VkPresentModeKHR> presentModes;
    };


    void initWindow();
    static void framebufferResizeCallback(GLFWwindow* window, int width, int height);
    static void mouse_callback(GLFWwindow* window, double xpos, double ypos);
    static void key_callback(GLFWwindow* window, int key, int scancode, int action, int mods);
    void createSurface();

    SwapChainSupportDetails querySwapChainSupport(VkPhysicalDevice device);
    VkSurfaceFormatKHR chooseSwapSurfaceFormat(const std::vector<VkSurfaceFormatKHR>& availableFormats);
    VkPresentModeKHR chooseSwapPresentMode(const std::vector<VkPresentModeKHR>& availablePresentModes);
    VkExtent2D chooseSwapExtent(const VkSurfaceCapabilitiesKHR& capabilities);

    void initVulkan();
    void recreateSwapChain();

    void createInstance();
    void pickPhysicalDevice();
    void createLogicalDevice();
    void createSwapChain();

    void createImageViews();

    void createRenderPass();
    void prepareShadowRenderpass();
    void prepareShadowFramebuffer();
    void createDescriptorPool();
    void createOffscreenBuffer();
    void createDescriptorSets(Object *obj);
    void createDescriptorSetLayout();
    void createGraphicsPipeline();
    void prepareShadowGraphicsPipeline();
    void createFramebuffers();
    void createCommandPool();
    void createDepthResources();
    VkFormat findSupportedFormat(const std::vector<VkFormat>& candidates, VkImageTiling tiling, VkFormatFeatureFlags features);
    VkFormat findDepthFormat();
    bool hasStencilComponent(VkFormat format);
    void createColorResources();
    

    
    void createBuffer(VkDeviceSize size, VkBufferUsageFlags usage, VkMemoryPropertyFlags properties, VkBuffer& buffer, VkDeviceMemory& bufferMemory);
    void copyBuffer(VkBuffer srcBuffer, VkBuffer dstBuffer, VkDeviceSize size);
    VkCommandBuffer beginSingleTimeCommands();
    void endSingleTimeCommands(VkCommandBuffer commandBuffer);

    
   
    

    void createVertexBuffer(Object *obj);
    void createIndexBuffer(Object *obj);
    void createMaterialBuffers(Object* obj);
    void createTransformBuffers(Object* obj);
    void createUniformBuffers();

    void createImage(uint32_t width, uint32_t height, uint32_t mipLevels, VkSampleCountFlagBits numSamples, VkFormat format, VkImageTiling tiling, VkImageUsageFlags usage, VkMemoryPropertyFlags properties, VkImage& image, VkDeviceMemory& imageMemory);
    VkImageView createImageView(VkImage image, VkFormat format, VkImageAspectFlags aspectFlags, uint32_t mipLevels);
    void createTextureImage(std::string path, Object *obj);
    void createTextureImageView(Object *obj);
    void createTextureSampler(Object *obj);
    void transitionImageLayout(VkImage image, VkFormat format, VkImageLayout oldLayout, VkImageLayout newLayout, uint32_t mipLevels);
    void copyBufferToImage(VkBuffer buffer, VkImage image, uint32_t width, uint32_t height);
    void generateMipmaps(VkImage image, VkFormat imageFormat, int32_t texWidth, int32_t texHeight, uint32_t mipLevels);

    uint32_t findMemoryType(uint32_t typeFilter, VkMemoryPropertyFlags properties);
    void createCommandBuffers();
    void createSyncObjects();
    

    void mainLoop();

    void calculateDeltaTime();

    void drawFrame();

    void updateOffscreenBuffer(uint32_t currentImage);
    void updateUniformBuffer(uint32_t currentImage);
    void updateMaterialBuffer(uint32_t currentImage);
    void updateTransforms(uint32_t currentImage);

    void loadModel(Object *obj);

    void createObject(std::string model_path, std::string texture_path, glm::vec3 position, glm::vec3 rotation, glm::vec3 scale);
    void createLight(lightType type, lightUpdate update, glm::vec3 pos, glm::vec3 rot);
    void recreateObjects();

    void cleanup();
    void cleanupSwapChain();

    VkShaderModule createShaderModule(const std::vector<char>& code);
    bool isDeviceSuitable(VkPhysicalDevice device);
    QueueFamilyIndices findQueueFamilies(VkPhysicalDevice device);
    VkSampleCountFlagBits getMaxUsableSampleCount();
    bool checkValidationLayerSupport();
    std::vector<const char*> getRequiredExtensions();
    static VKAPI_ATTR VkBool32 VKAPI_CALL debugCallback(VkDebugUtilsMessageSeverityFlagBitsEXT messageSeverity, VkDebugUtilsMessageTypeFlagsEXT messageType, const VkDebugUtilsMessengerCallbackDataEXT* pCallbackData, void* pUserData);
    void setupDebugMessenger();
    void populateDebugMessengerCreateInfo(VkDebugUtilsMessengerCreateInfoEXT& createInfo);
    bool checkDeviceExtensionSupport(VkPhysicalDevice device);
};

