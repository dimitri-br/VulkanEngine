#define GLFW_INCLUDE_VULKAN // GLFW vulkan support (Vulkan is an API not window)
#define GLM_FORCE_RADIANS // Use radians instead of degrees
#define STB_IMAGE_IMPLEMENTATION
#define GLM_FORCE_DEPTH_ZERO_TO_ONE
#define TINYOBJLOADER_IMPLEMENTATION
#define GLM_ENABLE_EXPERIMENTAL


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
#include <stb_image.h> // Used to import images to be used as textures
#include <chrono> // Time
#include <fstream> // Needed to load shaders
#include <array> // array stuff
#include <tiny_obj_loader.h> // load obj
#include <unordered_map> // used to check we aren't adding unnecessary verticies
#include <glm/gtx/hash.hpp> // used to hash

const uint32_t WIDTH = 1920;
const uint32_t HEIGHT = 1080;

const bool FULLSCREEN = true;

// base names to help customize later
const std::string MODEL_PATH = "models/cube.obj";
const std::string TEXTURE_PATH = "textures/texture.png";

// (Z, X, Y)
const glm::vec3 eyePos(3.0f, 0.0f, 2.0f); // eye pos, eg camera
const glm::vec3 lightPos(-5.0f, 2.0f, 2.5f); // light pos, eg sun



// Validation layers can be used to debug the app, as vulkan comes with minimal API overhead for speed and effiency. This can help find leaks, errors and more!!!!
const std::vector<const char*> validationLayers = { // What layers do we want
    "VK_LAYER_KHRONOS_validation"
};

// extensions required by the app
const std::vector<const char*> deviceExtensions = {
    VK_KHR_SWAPCHAIN_EXTENSION_NAME
};

// max frames allowed to be submitted to the GPU at the same time
const int MAX_FRAMES_IN_FLIGHT = 2;

// Helper function to read shaders
static std::vector<char> readFile(const std::string& filename) {
    std::ifstream file(filename, std::ios::ate | std::ios::binary);

    

    if (!file.is_open()) {
        throw std::runtime_error("failed to open file!");
    }

    // Start at end to get length of file, allocate a buffer
    size_t fileSize = (size_t)file.tellg();
    std::vector<char> buffer(fileSize);

    // Go to start and read
    file.seekg(0);
    file.read(buffer.data(), fileSize);


    file.close();

    return buffer;
}
// If we're in debug mode, then activate the validation layers
#ifdef NDEBUG
const bool enableValidationLayers = false;
#else
const bool enableValidationLayers = true;
#endif

// Create the messenger
VkResult CreateDebugUtilsMessengerEXT(VkInstance instance, const VkDebugUtilsMessengerCreateInfoEXT* pCreateInfo, const VkAllocationCallbacks* pAllocator, VkDebugUtilsMessengerEXT* pDebugMessenger) {
    auto func = (PFN_vkCreateDebugUtilsMessengerEXT)vkGetInstanceProcAddr(instance, "vkCreateDebugUtilsMessengerEXT");
    if (func != nullptr) {
        return func(instance, pCreateInfo, pAllocator, pDebugMessenger);
    }
    else {
        return VK_ERROR_EXTENSION_NOT_PRESENT;
    }
}

// Destroy the messenger
void DestroyDebugUtilsMessengerEXT(VkInstance instance, VkDebugUtilsMessengerEXT debugMessenger, const VkAllocationCallbacks* pAllocator) {
    auto func = (PFN_vkDestroyDebugUtilsMessengerEXT)vkGetInstanceProcAddr(instance, "vkDestroyDebugUtilsMessengerEXT");
    if (func != nullptr) {
        func(instance, debugMessenger, pAllocator);
    }
}

// This holds vertex data - namely the position and color of the vertex
struct Vertex {
    glm::vec3 pos; // pos of vertex
    glm::vec3 color; // color
    glm::vec2 texCoord; // Texture coordinate mapping
    glm::vec3 normal; // normals (lighting)

    // This function is needed to tell vulkan about how to use this struct
    static VkVertexInputBindingDescription getBindingDescription() {
        VkVertexInputBindingDescription bindingDescription{};

        bindingDescription.binding = 0; // what binding the per-vertex data comes.
        bindingDescription.stride = sizeof(Vertex); // number of bytes between each entry
        bindingDescription.inputRate = VK_VERTEX_INPUT_RATE_VERTEX; // is it per-vertex or per-instance

        return bindingDescription;
    }
    // Tells vulkan what is in the struct (For this, Pos and Color, texcoords and normals)
    static std::array<VkVertexInputAttributeDescription, 4> getAttributeDescriptions() {

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

    // This is required to make the Vertex struct work in a map
    bool operator==(const Vertex& other) const {
        return pos == other.pos && color == other.color && texCoord == other.texCoord && normal == other.normal;
    }
};

// Hash function for the vertex struct
namespace std {
    template<> struct hash<Vertex> {
        size_t operator()(Vertex const& vertex) const {
            return ((hash<glm::vec3>()(vertex.pos) ^ (hash<glm::vec3>()(vertex.color) << 1)) ^ (hash<glm::vec2>()(vertex.texCoord) << 1) >> 1) ^ (hash<glm::vec3>()(vertex.normal) << 1);
        }
    };
}

// This holds UBO data, like the model pos, view point and projection
struct UniformBufferObject {
    alignas(16) glm::mat4 model;
    alignas(16) glm::mat4 view;
    alignas(16) glm::mat4 proj;
    alignas(16) glm::vec3 lightPos;
    alignas(16) glm::vec3 camPos;
};

// Material information - holds info for the shader
struct Material {
    alignas(16) float shininess;
    alignas(16) glm::vec3 color;
};

// Holds descriptor set, material data and vertex data. Will be extended
struct Object {

};

// Pos, then Color
/*
const std::vector<Vertex> vertices = {
    {{-0.5f, -0.5f, 0.0f}, {1.0f, 0.0f, 0.0f}, {0.0f, 0.0f}},
    {{0.5f, -0.5f, 0.0f}, {0.0f, 1.0f, 0.0f}, {1.0f, 0.0f}},
    {{0.5f, 0.5f, 0.0f}, {0.0f, 0.0f, 1.0f}, {1.0f, 1.0f}},
    {{-0.5f, 0.5f, 0.0f}, {1.0f, 1.0f, 1.0f}, {0.0f, 1.0f}},

    {{-0.5f, -0.5f, -1.0f}, {1.0f, 0.0f, 0.0f}, {0.0f, 0.0f}},
    {{0.5f, -0.5f, -1.0f}, {0.0f, 1.0f, 0.0f}, {1.0f, 0.0f}},
    {{0.5f, 0.5f, -1.0f}, {0.0f, 0.0f, 1.0f}, {1.0f, 1.0f}},
    {{-0.5f, 0.5f, -1.0f}, {1.0f, 1.0f, 1.0f}, {0.0f, 1.0f}}
};
// Index buffer - reuse verticies
const std::vector<uint16_t> indices = {
    0, 1, 2, 2, 3, 0,
    4, 5, 6, 6, 7, 4
};
*/

class HelloTriangleApplication {
public:

    void run() {
        initWindow();
        initVulkan();
        mainLoop();
        cleanup();
    }

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

    std::vector<VkCommandBuffer> commandBuffers; // Stores every command. They're implicitly destroyed, so no need to cleanup

    // Needed to interface with the GLFW window
    VkSurfaceKHR surface;

    // present queue will present to the surface/window
    VkQueue presentQueue;

    // Tells vulkan about what attachments to use (Like color, stencil etc)
    VkRenderPass renderPass;

    VkSampleCountFlagBits msaaSamples = VK_SAMPLE_COUNT_1_BIT; // Multisampling. 1 bit is equivillant to nothing

    VkImage colorImage;
    VkDeviceMemory colorImageMemory;
    VkImageView colorImageView;


    VkDescriptorSetLayout descriptorSetLayout;

    VkDescriptorPool descriptorPool;
    std::vector<VkDescriptorSet> descriptorSets;

    // Pipeline layouts can be used to send uniform data to shaders. This can be anything from a transformation matrix or textures for a fragment shader.
    VkPipelineLayout pipelineLayout;

    // Vertex and index buffers must be stored as buffers on the GPU, as they need to be read by the device.
    std::vector<Vertex> vertices;
    std::vector<uint32_t> indices;
    VkBuffer vertexBuffer;
    VkDeviceMemory vertexBufferMemory;
    VkBuffer indexBuffer;
    VkDeviceMemory indexBufferMemory;
    // This is stored as an array so if one buffer is in-flight, we don't modify it by mistake.
    std::vector<VkBuffer> uniformBuffers;
    std::vector<VkDeviceMemory> uniformBuffersMemory;

    std::vector<VkBuffer> materialBuffers;
    std::vector<VkDeviceMemory> materialBuffersMemory;

    VkImage textureImage;
    VkDeviceMemory textureImageMemory;
    VkImageView textureImageView; // in order to access the texture images, we need to access them through views.
    VkSampler textureSampler;

    // We must create our own depth image to attach to the graphics pipeline
    VkImage depthImage;
    VkDeviceMemory depthImageMemory;
    VkImageView depthImageView;

    VkPipeline graphicsPipeline;

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


    //Initialize window - all required parameters!
    void initWindow() {
        glfwInit(); // Initialize glfw (The thing we will use to render the window)

        glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API); // Default is OpenGL - Lets make it use no API (so we can setup vulkan)
        glfwWindowHint(GLFW_RESIZABLE, GLFW_TRUE); //  resizable

        window = glfwCreateWindow(WIDTH, HEIGHT, "Vulkan", glfwGetPrimaryMonitor(), nullptr); // Create the window object
        glfwSetWindowUserPointer(window, this); // set a pointer to this class so we can use it in static functions where we pass a window
        glfwSetFramebufferSizeCallback(window, framebufferResizeCallback);
    }

    // This function will be called whenever the screen is resized, along with the new width and height
    static void framebufferResizeCallback(GLFWwindow* window, int width, int height) {
        auto app = reinterpret_cast<HelloTriangleApplication*>(glfwGetWindowUserPointer(window));
        app->framebufferResized = true;
    }
    // Create a surface - this is how vulkan interfaces with the window for renderering. Can affect Device selection, so we need to do it first! Is optional.
    void createSurface() {
        if (glfwCreateWindowSurface(instance, window, nullptr, &surface) != VK_SUCCESS) {
            throw std::runtime_error("failed to create window surface!");
        }
    }

    SwapChainSupportDetails querySwapChainSupport(VkPhysicalDevice device) {
        SwapChainSupportDetails details;

        vkGetPhysicalDeviceSurfaceCapabilitiesKHR(device, surface, &details.capabilities);

        uint32_t formatCount;
        vkGetPhysicalDeviceSurfaceFormatsKHR(device, surface, &formatCount, nullptr);

        if (formatCount != 0) {
            details.formats.resize(formatCount);
            vkGetPhysicalDeviceSurfaceFormatsKHR(device, surface, &formatCount, details.formats.data());
        }

        uint32_t presentModeCount;
        vkGetPhysicalDeviceSurfacePresentModesKHR(device, surface, &presentModeCount, nullptr);

        if (presentModeCount != 0) {
            details.presentModes.resize(presentModeCount);
            vkGetPhysicalDeviceSurfacePresentModesKHR(device, surface, &presentModeCount, details.presentModes.data());
        }

        return details;
    }


    // Pick best possible format (SRGB is good)
    VkSurfaceFormatKHR chooseSwapSurfaceFormat(const std::vector<VkSurfaceFormatKHR>& availableFormats) {
        for (const auto& availableFormat : availableFormats) {
            if (availableFormat.format == VK_FORMAT_B8G8R8A8_SRGB && availableFormat.colorSpace == VK_COLOR_SPACE_SRGB_NONLINEAR_KHR) {
                return availableFormat;
            }
        }

        return availableFormats[0];
    }

    // Choose best present mode
    VkPresentModeKHR chooseSwapPresentMode(const std::vector<VkPresentModeKHR>& availablePresentModes) {
        for (const auto& availablePresentMode : availablePresentModes) {
            if (availablePresentMode == VK_PRESENT_MODE_MAILBOX_KHR) {
                return availablePresentMode;
            }
        }

        return VK_PRESENT_MODE_FIFO_KHR;
    }

    // Resolution of swapchain images (Think of swapchain = buffer). Almost always equal to res of window
    VkExtent2D chooseSwapExtent(const VkSurfaceCapabilitiesKHR& capabilities) {
        if (capabilities.currentExtent.width != UINT32_MAX) {
            return capabilities.currentExtent;
        }
        else {
            int width, height;
            glfwGetFramebufferSize(window, &width, &height);

            VkExtent2D actualExtent = {
                static_cast<uint32_t>(width),
                static_cast<uint32_t>(height)
            };

            actualExtent.width = std::max(capabilities.minImageExtent.width, std::min(capabilities.maxImageExtent.width, actualExtent.width));
            actualExtent.height = std::max(capabilities.minImageExtent.height, std::min(capabilities.maxImageExtent.height, actualExtent.height));

            return actualExtent;
        }
    }


    // Initialize vulkan - run all the boilerplate code needed to setup
    void initVulkan() {
        createInstance(); // See the comment at createInstance
        setupDebugMessenger(); // See comment
        createSurface(); // see comment
        pickPhysicalDevice(); // See comment

        createLogicalDevice(); // See comment

        createSwapChain(); // *sigh*

        createImageViews(); // Get the idea?

        createRenderPass();

        createDescriptorSetLayout();

        createGraphicsPipeline(); // Come on!

        createCommandPool();

        createColorResources();

        createDepthResources();

        createFramebuffers();

        createTextureImage(TEXTURE_PATH);
        createTextureImageView();
        createTextureSampler();

        loadModel();

        createVertexBuffer();
        createIndexBuffer();
        createUniformBuffers();
        createMaterialBuffers();

        createDescriptorPool();
        createDescriptorSets();

        createCommandBuffers();

        createSyncObjects();
    }

    //Main loop
    void mainLoop() {
        while (!glfwWindowShouldClose(window)) { // While the window shouldn't close
            glfwPollEvents(); // Loop!
            drawFrame();
        }

        vkDeviceWaitIdle(device); // make sure we're finished rendering before destroying the window
    }

    // drawFrame will do the following operations - Aqiuire an image from the swapchain, execute a command buffer then return the image to the swapchain for presentation
    void drawFrame() {
        // We wait for fences to be called, then we reset the fence to its uninitialized state. This means we won't draw anything until we are ready to.
        vkWaitForFences(device, 1, &inFlightFences[currentFrame], VK_TRUE, UINT64_MAX);


        uint32_t imageIndex; // Current image index
        // As swapchains are an extension, we must use the KHR. This function gets the next image from the swapchain to be rendered.
        // the device to use -> the swapchain to get images from -> timeout in nanoseconds -> our sephamore -> fence (both can be used) -> the index of the image we got
        VkResult result = vkAcquireNextImageKHR(device, swapChain, UINT64_MAX, imageAvailableSemaphores[currentFrame], VK_NULL_HANDLE, &imageIndex);

        if (result == VK_ERROR_OUT_OF_DATE_KHR) { // If the surface and swapchain are incompatiable, recreate the swapchain
            recreateSwapChain();
            return;
        }
        else if (result != VK_SUCCESS && result != VK_SUBOPTIMAL_KHR) { // if it failed, and the result can still present but isnt optimial, throw an error
            throw std::runtime_error("failed to acquire swap chain image!");
        }

        // Check if a previous frame is using this image (i.e. there is its fence to wait on)
        if (imagesInFlight[imageIndex] != VK_NULL_HANDLE) {
            vkWaitForFences(device, 1, &imagesInFlight[imageIndex], VK_TRUE, UINT64_MAX); // wait until it is no longer being used
        }
        // Mark the image as now being in use by this frame
        imagesInFlight[imageIndex] = inFlightFences[currentFrame];

        updateUniformBuffer(imageIndex);
        updateMaterialBuffer(imageIndex);

        // To submit to the command buffer, we need to create submit info
        VkSubmitInfo submitInfo{};
        submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;

        VkSemaphore waitSemaphores[] = { imageAvailableSemaphores[currentFrame] }; // the sephamore so we don't mess it up
        VkPipelineStageFlags waitStages[] = { VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT }; // what can be done while we wait for the image to arrive
        submitInfo.waitSemaphoreCount = 1;
        submitInfo.pWaitSemaphores = waitSemaphores;
        submitInfo.pWaitDstStageMask = waitStages;

        submitInfo.commandBufferCount = 1;
        submitInfo.pCommandBuffers = &commandBuffers[imageIndex]; // get the command buffer with the corresponding image index

        VkSemaphore signalSemaphores[] = { renderFinishedSemaphores[currentFrame] }; // must be an array as we can use multiple
        submitInfo.signalSemaphoreCount = 1;
        submitInfo.pSignalSemaphores = signalSemaphores; // what sephamore should be signalled when its done.

        // Best to reset the fence until we're about to use it, so when we wait it works properly (So if it gets set, we don't accidentally reset it)
        vkResetFences(device, 1, &inFlightFences[currentFrame]);
        // Submit the queue to the GPU. We will also wait on the CPU to finish its thing before submitting.
        if (vkQueueSubmit(graphicsQueue, 1, &submitInfo, inFlightFences[currentFrame]) != VK_SUCCESS) {
            throw std::runtime_error("failed to submit draw command buffer!");
        }

        // To present, we need to wait for the render pass to finish
        VkPresentInfoKHR presentInfo{};
        presentInfo.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;

        presentInfo.waitSemaphoreCount = 1;
        presentInfo.pWaitSemaphores = signalSemaphores; // reuse the signal
        
        VkSwapchainKHR swapChains[] = { swapChain }; // The swapchain to send it back to
        presentInfo.swapchainCount = 1;
        presentInfo.pSwapchains = swapChains;
        presentInfo.pImageIndices = &imageIndex; // what index the image was, that we got when we first read the image

        presentInfo.pResults = nullptr; // Optional - can be used to check the results of the swapchain

        result = vkQueuePresentKHR(presentQueue, &presentInfo); // sends the rendered frame back to the swapchain.

        if (result == VK_ERROR_OUT_OF_DATE_KHR || result == VK_SUBOPTIMAL_KHR || framebufferResized) { // Check again if we should recreate the swapchain
            recreateSwapChain();
            framebufferResized = false; // reset the resize
        }
        else if (result != VK_SUCCESS) {
            throw std::runtime_error("failed to present swap chain image!");
        } 

        vkQueueWaitIdle(presentQueue); // wait for the work to finish to prevent memory usage going up - make sure the GPU can keep up!

        currentFrame = (currentFrame + 1) % MAX_FRAMES_IN_FLIGHT; // the modulo makes sure the current frame never goes over 2, so we either render frame 0 or frame 1
    }

    // Updates the uniform buffer depending on the current swapchain image
    void updateUniformBuffer(uint32_t currentImage) {
        std::cout << "Update ubo!!!!\n\n\n\n" << std::endl;
        static auto startTime = std::chrono::high_resolution_clock::now(); // Get the time

        auto currentTime = std::chrono::high_resolution_clock::now(); // Get the time
        float time = std::chrono::duration<float, std::chrono::seconds::period>(currentTime - startTime).count(); // Check how much time has passed

        // Create a ubo, then rotate the model Z by 5 radians every second
        UniformBufferObject ubo{};
        

        ubo.model = glm::translate(glm::mat4(1.0f), glm::vec3(0.0f, 0.0f, 0.0f)); // translate the model. vec3 specifies translation. Seems to be (z, x, y)
        ubo.model *= glm::rotate(glm::mat4(1.0f), time * glm::radians(5.0f), glm::vec3(0.0f, 0.0f, 1.0f));
        
        ubo.lightPos = lightPos;
        ubo.camPos = eyePos;


        // How we look at the world (Useful for 3d)
        // Eye pos, center pos and up axis
        ubo.view = glm::lookAt(eyePos, glm::vec3(0.0f, 0.0f, 0.0f), glm::vec3(0.0f, 0.0f, 1.0f));
        // Perspective - 45 degre vertical FOV, aspect ratio and near and far planes.
        ubo.proj = glm::perspective(glm::radians(45.0f), swapChainExtent.width / (float)swapChainExtent.height, 0.1f, 10.0f);

        // Don't forget to flip Y axis as it was designed for opengl!!!!!
        ubo.proj[1][1] *= -1;

        // Now all the transformations are done, write it to memory.
        void* data;
        vkMapMemory(device, uniformBuffersMemory[currentImage], 0, sizeof(ubo), 0, &data);
        memcpy(data, &ubo, sizeof(ubo));
        vkUnmapMemory(device, uniformBuffersMemory[currentImage]);
    }

    void updateMaterialBuffer(uint32_t currentImage) {
        std::cout << "Update mat!!!!\n\n\n\n" << std::endl;
      
        Material mat{};

        mat.shininess = 1.0f;
        mat.color = { 1.0f, 1.0f, 1.0f };


        // write material to memory
        void* data;
        vkMapMemory(device, materialBuffersMemory[currentImage], 0, sizeof(mat), 0, &data);
        memcpy(data, &mat, sizeof(mat));
        vkUnmapMemory(device, materialBuffersMemory[currentImage]);
    }

    void loadModel() {
        tinyobj::attrib_t attrib; // holds the position, normals and texture coordinates
        std::vector<tinyobj::shape_t> shapes; // contains all seperate objects + faces
        std::vector<tinyobj::material_t> materials; //ignore
        std::string warn, err; // warnings and errors while loading the file

        if (!tinyobj::LoadObj(&attrib, &shapes, &materials, &warn, &err, MODEL_PATH.c_str())) {
            throw std::runtime_error(warn + err);
        }

        std::unordered_map<Vertex, uint32_t> uniqueVertices{}; // a map of verticies to make sure we aren't duplicating unnecessary verticies.

        // we're gonna combine all the objects and faces into one
        for (const auto& shape : shapes) {
            for (const auto& index : shape.mesh.indices) {
                // loop through all the index and vertex of the object and dump it into the buffers.
                Vertex vertex{};
                // get pos from the attrib using the index to get the verticies we need
                vertex.pos = {
                    attrib.vertices[3 * index.vertex_index + 0],
                    attrib.vertices[3 * index.vertex_index + 1],
                    attrib.vertices[3 * index.vertex_index + 2]
                };

                // get the texture coordinates for this vertex
                vertex.texCoord = {
                    attrib.texcoords[2 * index.texcoord_index + 0],
                    1.0 - attrib.texcoords[2 * index.texcoord_index + 1] // flip the tex-coord as vulkan uses the DX system, rather than opengl
                };
                // just white
                vertex.color = { 1.0f, 1.0f, 1.0f };

                vertex.normal = {
                    attrib.normals[3 * index.normal_index + 0],
                    attrib.normals[3 * index.normal_index + 1],
                    attrib.normals[3 * index.normal_index + 2]
                };
                
                // add to the buffers - but only add the vertex if it is not present already within the array
                if (uniqueVertices.count(vertex) == 0) {
                    uniqueVertices[vertex] = static_cast<uint32_t>(vertices.size());
                    vertices.push_back(vertex);
                }
                // add the vertex to the index buffer
                indices.push_back(uniqueVertices[vertex]);
                
            }
        }
        std::cout << "Finished loading the model" << std::endl;
    }

    // Needed to stay memory-safe. Cleans up on exit
    void cleanup() {
        cleanupSwapChain();

        // Destroy the sampler
        vkDestroySampler(device, textureSampler, nullptr);

        // Destroy the image view
        vkDestroyImageView(device, textureImageView, nullptr);

        // Destroy the image and the imageMemory
        vkDestroyImage(device, textureImage, nullptr);
        vkFreeMemory(device, textureImageMemory, nullptr);

        // destroy our descriptor set
        vkDestroyDescriptorSetLayout(device, descriptorSetLayout, nullptr);

        // Destroy the buffer and buffer memory
        vkDestroyBuffer(device, vertexBuffer, nullptr);
        vkFreeMemory(device, vertexBufferMemory, nullptr);


        for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
            vkDestroySemaphore(device, renderFinishedSemaphores[i], nullptr);
            vkDestroySemaphore(device, imageAvailableSemaphores[i], nullptr);
            vkDestroyFence(device, inFlightFences[i], nullptr);
        }

        vkDestroyCommandPool(device, commandPool, nullptr);

        vkDestroyDevice(device, nullptr);

        if (enableValidationLayers) {
            DestroyDebugUtilsMessengerEXT(instance, debugMessenger, nullptr);
        }

        vkDestroySurfaceKHR(instance, surface, nullptr);
        vkDestroyInstance(instance, nullptr);

        glfwDestroyWindow(window);

        glfwTerminate();
    }

    // Create a vulkan instance - This tells vulkan about the app information and connects the app to vulkan
    void createInstance() { 
        if (enableValidationLayers && !checkValidationLayerSupport()) { // Check if we have the validation layers, or panic and leave
            throw std::runtime_error("validation layers requested, but not available!");
        }
        VkApplicationInfo appInfo{};
        appInfo.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
        appInfo.pApplicationName = "Hello Triangle";
        appInfo.applicationVersion = VK_MAKE_VERSION(1, 0, 0);
        appInfo.pEngineName = "VulkanEngine";
        appInfo.engineVersion = VK_MAKE_VERSION(1, 0, 0);
        appInfo.apiVersion = VK_API_VERSION_1_2;

        VkInstanceCreateInfo createInfo{};
        createInfo.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
        createInfo.pApplicationInfo = &appInfo;


        // Extensions - needed to interface with the window & platform, as vulkan is platform agnostic
        
        auto extensions = getRequiredExtensions();
        createInfo.enabledExtensionCount = static_cast<uint32_t>(extensions.size());
        createInfo.ppEnabledExtensionNames = extensions.data();


        // If we are using valid layers, enable them or just continue as normal)
        VkDebugUtilsMessengerCreateInfoEXT debugCreateInfo;
        if (enableValidationLayers) {
            createInfo.enabledLayerCount = static_cast<uint32_t>(validationLayers.size());
            createInfo.ppEnabledLayerNames = validationLayers.data();
            populateDebugMessengerCreateInfo(debugCreateInfo);
            createInfo.pNext = (VkDebugUtilsMessengerCreateInfoEXT*)&debugCreateInfo;
        }
        else {
            createInfo.enabledLayerCount = 0;
            createInfo.pNext = nullptr;
        }

        // Error checking - If something went wrong, let us know!
        if (vkCreateInstance(&createInfo, nullptr, &instance) != VK_SUCCESS) {
            throw std::runtime_error("failed to create instance!");
        }
    }

    // Select a GPU to run on. Can select as many as needed, but this will stick to one.
    void pickPhysicalDevice() {
        //  Very similar to listing extensions - just a count.
        uint32_t deviceCount = 0;
        vkEnumeratePhysicalDevices(instance, &deviceCount, nullptr);

        if (deviceCount == 0) {
            throw std::runtime_error("failed to find GPUs with Vulkan support!");
        }

        // Go through all the device count and return a device.
        std::vector<VkPhysicalDevice> devices(deviceCount);
        vkEnumeratePhysicalDevices(instance, &deviceCount, devices.data());

        // Check if the devices are suitable
        for (const auto& device : devices) {
            if (isDeviceSuitable(device)) {
                physicalDevice = device;
                msaaSamples = getMaxUsableSampleCount();
                break;
            }
        }

        if (physicalDevice == VK_NULL_HANDLE) {
            throw std::runtime_error("failed to find a suitable GPU!");
        }
    }

    // Interfaces with the Physical Device
    void createLogicalDevice() {
        QueueFamilyIndices indices = findQueueFamilies(physicalDevice);

        std::vector<VkDeviceQueueCreateInfo> queueCreateInfos;
        std::set<uint32_t> uniqueQueueFamilies = { indices.graphicsFamily.value(), indices.presentFamily.value() };

        float queuePriority = 1.0f;
        for (uint32_t queueFamily : uniqueQueueFamilies) {
            VkDeviceQueueCreateInfo queueCreateInfo{};
            queueCreateInfo.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
            queueCreateInfo.queueFamilyIndex = queueFamily;
            queueCreateInfo.queueCount = 1;
            queueCreateInfo.pQueuePriorities = &queuePriority;
            queueCreateInfos.push_back(queueCreateInfo);
        }

        VkPhysicalDeviceFeatures deviceFeatures{}; // Will be updated. Contains things like geometry shaders, and device features.
        deviceFeatures.samplerAnisotropy = VK_TRUE; // enable anisotropic filtering


        // Create the logical device

        VkDeviceCreateInfo createInfo{};
        createInfo.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;

        createInfo.queueCreateInfoCount = static_cast<uint32_t>(queueCreateInfos.size());
        createInfo.pQueueCreateInfos = queueCreateInfos.data();
        createInfo.queueCreateInfoCount = 1;

        createInfo.pEnabledFeatures = &deviceFeatures; // Device features


        // Very similar to Instance creation. We need to get the layers needed, eg for features and so on. Also for validation for debugging
        createInfo.enabledExtensionCount = static_cast<uint32_t>(deviceExtensions.size());
        createInfo.ppEnabledExtensionNames = deviceExtensions.data();

        if (enableValidationLayers) {
            createInfo.enabledLayerCount = static_cast<uint32_t>(validationLayers.size());
            createInfo.ppEnabledLayerNames = validationLayers.data();
        }
        else {
            createInfo.enabledLayerCount = 0;
        }

        // Create the logical device and check for errors.
        if (vkCreateDevice(physicalDevice, &createInfo, nullptr, &device) != VK_SUCCESS) {
            throw std::runtime_error("failed to create logical device!");
        }

        // Get the required queues
        vkGetDeviceQueue(device, indices.graphicsFamily.value(), 0, &graphicsQueue);
        vkGetDeviceQueue(device, indices.presentFamily.value(), 0, &presentQueue);
    }

    // Create the swapchain. Swapchain is buffer of frames for speed.
    void createSwapChain() {
        // This just checks what is availiable to the user, and if it is compatiable. It also selects the best options.
        SwapChainSupportDetails swapChainSupport = querySwapChainSupport(physicalDevice); // Check for a swapchain

        VkSurfaceFormatKHR surfaceFormat = chooseSwapSurfaceFormat(swapChainSupport.formats); // Choose the format of the swapcahin
        VkPresentModeKHR presentMode = chooseSwapPresentMode(swapChainSupport.presentModes); // Choose how to present
        VkExtent2D extent = chooseSwapExtent(swapChainSupport.capabilities); // Choose the extent


        // Needed for other parts of vulkan, so we save it into the class members.
        swapChainImageFormat = surfaceFormat.format;
        swapChainExtent = extent;


        // How many images do we want. Recommended to always put one more to boost driver speed.
        uint32_t imageCount = swapChainSupport.capabilities.minImageCount + 1;

        // Check we do not exceed the max - 0 means unlimited.
        if (swapChainSupport.capabilities.maxImageCount > 0 && imageCount > swapChainSupport.capabilities.maxImageCount) {
            imageCount = swapChainSupport.capabilities.maxImageCount;
        }

        // Create the swapchain
        VkSwapchainCreateInfoKHR createInfo{};
        createInfo.sType = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR;
        createInfo.surface = surface;

        createInfo.minImageCount = imageCount;
        createInfo.imageFormat = surfaceFormat.format;
        createInfo.imageColorSpace = surfaceFormat.colorSpace;
        createInfo.imageExtent = extent;
        createInfo.imageArrayLayers = 1;
        createInfo.imageUsage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT; // This can be used for post-processing. Here, this renders directly

        QueueFamilyIndices indices = findQueueFamilies(physicalDevice);
        uint32_t queueFamilyIndices[] = { indices.graphicsFamily.value(), indices.presentFamily.value() };

        if (indices.graphicsFamily != indices.presentFamily) {
            createInfo.imageSharingMode = VK_SHARING_MODE_CONCURRENT; // ownership is concurrent - can be shared
            createInfo.queueFamilyIndexCount = 2;
            createInfo.pQueueFamilyIndices = queueFamilyIndices;
        }
        else {
            createInfo.imageSharingMode = VK_SHARING_MODE_EXCLUSIVE; // ownership is exclusive - must be asked for. Best performance, at loss of ease of use
            createInfo.queueFamilyIndexCount = 0; // Optional
            createInfo.pQueueFamilyIndices = nullptr; // Optional
        }

        if (vkCreateSwapchainKHR(device, &createInfo, nullptr, &swapChain) != VK_SUCCESS) {
            throw std::runtime_error("failed to create swap chain!");
        }

        vkGetSwapchainImagesKHR(device, swapChain, &imageCount, nullptr);
        swapChainImages.resize(imageCount);
        vkGetSwapchainImagesKHR(device, swapChain, &imageCount, swapChainImages.data());
    }

    // Creates a basic image view for every image in the swapchain
    void createImageViews() {
        swapChainImageViews.resize(swapChainImages.size()); // Resize all the image views to fit the swapchain images

        for (size_t i = 0; i < swapChainImages.size(); i++) { // Iterate over all the swapchain images
            swapChainImageViews[i] = createImageView(swapChainImages[i], swapChainImageFormat, VK_IMAGE_ASPECT_COLOR_BIT, 1); // See function
        }
    }

    // Render passes define what attachments will be used by the framebuffer. This can define color and depth buffers, samples and how their contents will be handled.
    void createRenderPass() {
        VkAttachmentDescription colorAttachment{};
        colorAttachment.format = swapChainImageFormat; // Should match our swapchain
        colorAttachment.samples = msaaSamples; //  sample count (MSAA)
        // This part can be used to do stencil stuff
        colorAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR; // What to do before rendering - in this case, clear
        colorAttachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE; // What to do after rendering - in this case, store
        // Textures and frame buffers are represented by VK_IMAGE. This can change how the pixels are stored in memory
        colorAttachment.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED; // This means the initial image is undefined - this can mean the image beforehand is not guaranteed to be preserved.
        colorAttachment.finalLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL; // This means the final layout is a swapchain image - anything run through MSAA cannot be rendered directly!

        // attachement for depth. Similar to color, except we have to use the same layout as the depth image.
        VkAttachmentDescription depthAttachment{};
        depthAttachment.format = findDepthFormat();
        depthAttachment.samples = msaaSamples;
        depthAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
        depthAttachment.storeOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
        depthAttachment.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
        depthAttachment.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
        depthAttachment.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED; // we don't care about the old contents
        depthAttachment.finalLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;

        // As things run through MSAA cannot be rendered directly, we must run it through an attachement resolve, which will convert it into a drawable format
        VkAttachmentDescription colorAttachmentResolve{};
        colorAttachmentResolve.format = swapChainImageFormat;
        colorAttachmentResolve.samples = VK_SAMPLE_COUNT_1_BIT;
        colorAttachmentResolve.loadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
        colorAttachmentResolve.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
        colorAttachmentResolve.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
        colorAttachmentResolve.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
        colorAttachmentResolve.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        colorAttachmentResolve.finalLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;


        // Render passes can be made up of smaller subpasses. This can be used for post-processing effects. This can significantly boost performance.
        VkAttachmentReference colorAttachmentRef{};
        colorAttachmentRef.attachment = 0; // Specifies which attachement to reference in the attachment array index.
        colorAttachmentRef.layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL; // The layout of the attachment when we begin the subpass

        // reference to the attachment
        VkAttachmentReference depthAttachmentRef{};
        depthAttachmentRef.attachment = 1;
        depthAttachmentRef.layout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;

        // For the resolve
        VkAttachmentReference colorAttachmentResolveRef{};
        colorAttachmentResolveRef.attachment = 2;
        colorAttachmentResolveRef.layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

        // Make a subpass
        VkSubpassDescription subpass{};
        subpass.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS; // Must be explicit that is it a graphics pass
        // Pass our attachment to the subpass
        subpass.colorAttachmentCount = 1;
        subpass.pColorAttachments = &colorAttachmentRef; // attach the color reference
        subpass.pDepthStencilAttachment = &depthAttachmentRef; // attach the depth reference
        subpass.pResolveAttachments = &colorAttachmentResolveRef; // attach the resolve reference

        // This can be used to prevent the subpass writing to the image before its ready to use.
        VkSubpassDependency dependency{};
        dependency.srcSubpass = VK_SUBPASS_EXTERNAL;
        dependency.dstSubpass = 0;
        // We will wait on the COLOR_ATTACHMENT_OUTPUT_BIT to do its thing, and won't draw until necessary.
        dependency.srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT; 
        dependency.srcAccessMask = 0;
        dependency.dstStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
        dependency.dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;

        std::array<VkAttachmentDescription, 3> attachments = { colorAttachment, depthAttachment, colorAttachmentResolve };
        // Build the renderpass. Reference our attachments and our subpass
        VkRenderPassCreateInfo renderPassInfo{};
        renderPassInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
        renderPassInfo.attachmentCount = static_cast<uint32_t>(attachments.size());
        renderPassInfo.pAttachments = attachments.data();
        renderPassInfo.subpassCount = 1;
        renderPassInfo.pSubpasses = &subpass;
        // This prevents us writing to the image and adding color too early
        renderPassInfo.dependencyCount = 1;
        renderPassInfo.pDependencies = &dependency;

        if (vkCreateRenderPass(device, &renderPassInfo, nullptr, &renderPass) != VK_SUCCESS) {
            throw std::runtime_error("failed to create render pass!");
        }
    }

    // Descriptor sets cannot be created directly, they must be allocated in descriptor pools.
    void createDescriptorPool() {
        // we have to allocate one pool for every frame, to prevent conflicts while in-flight. Also include all the descriptors we will use
        std::array<VkDescriptorPoolSize, 3> poolSizes{};
        poolSizes[0].type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        poolSizes[0].descriptorCount = static_cast<uint32_t>(swapChainImages.size());


        poolSizes[1].type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        poolSizes[1].descriptorCount = static_cast<uint32_t>(swapChainImages.size());

        poolSizes[2].type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        poolSizes[2].descriptorCount = static_cast<uint32_t>(swapChainImages.size());

        // we must specify the max amount of descriptor sets that can be allocated
        VkDescriptorPoolCreateInfo poolInfo{};
        poolInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
        poolInfo.poolSizeCount = static_cast<uint32_t>(poolSizes.size()); // size of the pool (How many descriptors to use)
        poolInfo.pPoolSizes = poolSizes.data(); // the pool size within the array
        poolInfo.maxSets = static_cast<uint32_t>(swapChainImages.size());

        poolInfo.flags = 0; // optional - this can be used to define whether individual descriptor sets can be freed or not.

        if (vkCreateDescriptorPool(device, &poolInfo, nullptr, &descriptorPool) != VK_SUCCESS) {
            throw std::runtime_error("failed to create descriptor pool!");
        }
    }

    // Create descriptor sets, which can be used by shaders to access buffer data (Like the UBO) or image data.
    void createDescriptorSets() {
        std::vector<VkDescriptorSetLayout> layouts(swapChainImages.size(), descriptorSetLayout);

        VkDescriptorSetAllocateInfo allocInfo{};
        allocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
        allocInfo.descriptorPool = descriptorPool;
        allocInfo.descriptorSetCount = static_cast<uint32_t>(swapChainImages.size());
        allocInfo.pSetLayouts = layouts.data();

        descriptorSets.resize(swapChainImages.size());
        if (vkAllocateDescriptorSets(device, &allocInfo, descriptorSets.data()) != VK_SUCCESS) {
            throw std::runtime_error("failed to allocate descriptor sets!");
        }

        for (size_t i = 0; i < swapChainImages.size(); i++) {
            VkDescriptorBufferInfo uboBufferInfo{};
            uboBufferInfo.buffer = uniformBuffers[i];
            uboBufferInfo.offset = 0;
            uboBufferInfo.range = sizeof(UniformBufferObject);

            VkDescriptorImageInfo imageInfo{};
            imageInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
            imageInfo.imageView = textureImageView;
            imageInfo.sampler = textureSampler;

            VkDescriptorBufferInfo matBufferInfo{};
            matBufferInfo.buffer = materialBuffers[i];
            matBufferInfo.offset = 0;
            matBufferInfo.range = sizeof(Material);

            // create an array to store our descriptor sets
            std::array<VkWriteDescriptorSet, 3> descriptorWrites{};

            descriptorWrites[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            descriptorWrites[0].dstSet = descriptorSets[i]; // the set to write to ( we jave one for each frame in the swapchain)
            descriptorWrites[0].dstBinding = 0;// the binding in the shader
            descriptorWrites[0].dstArrayElement = 0;
            descriptorWrites[0].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
            descriptorWrites[0].descriptorCount = 1;
            descriptorWrites[0].pBufferInfo = &uboBufferInfo; // the buffer to put in the set

            descriptorWrites[1].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            descriptorWrites[1].dstSet = descriptorSets[i];
            descriptorWrites[1].dstBinding = 1; // the binding in the shader
            descriptorWrites[1].dstArrayElement = 0;
            descriptorWrites[1].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
            descriptorWrites[1].descriptorCount = 1;
            descriptorWrites[1].pImageInfo = &imageInfo; // the image (the image sampler) to put in the set

            descriptorWrites[2].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            descriptorWrites[2].dstSet = descriptorSets[i]; // the set to write to ( we jave one for each frame in the swapchain)
            descriptorWrites[2].dstBinding = 2;// the binding in the shader
            descriptorWrites[2].dstArrayElement = 0;
            descriptorWrites[2].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
            descriptorWrites[2].descriptorCount = 1;
            descriptorWrites[2].pBufferInfo = &matBufferInfo; // the buffer to put in the set
            
            // Update descriptor sets on the device
            vkUpdateDescriptorSets(device, static_cast<uint32_t>(descriptorWrites.size()), descriptorWrites.data(), 0, nullptr);
        }
    }

    // Automatically creates the descriptor sets.
    void createDescriptorSetLayout() {
        VkDescriptorSetLayoutBinding uboLayoutBinding{};
        uboLayoutBinding.binding = 0;
        uboLayoutBinding.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER; // define it is a uniform buffer
        uboLayoutBinding.descriptorCount = 1;
        uboLayoutBinding.stageFlags = VK_SHADER_STAGE_VERTEX_BIT; // what stage it will be used on
        uboLayoutBinding.pImmutableSamplers = nullptr; // Optional - useful for images


        // Create a binding to a sampler, so the shader can access the image from the sampler
        VkDescriptorSetLayoutBinding samplerLayoutBinding{};
        samplerLayoutBinding.binding = 1; // will be binding 1
        samplerLayoutBinding.descriptorCount = 1;
        samplerLayoutBinding.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER; // the type of descriptor
        samplerLayoutBinding.pImmutableSamplers = nullptr;
        samplerLayoutBinding.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT; // When we will use it - the frag shader

        VkDescriptorSetLayoutBinding matLayoutBinding{};
        matLayoutBinding.binding = 2;
        matLayoutBinding.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER; // define it is a uniform buffer
        matLayoutBinding.descriptorCount = 1;
        matLayoutBinding.stageFlags = VK_SHADER_STAGE_VERTEX_BIT; // what stage it will be used on
        matLayoutBinding.pImmutableSamplers = nullptr; // Optional - useful for images

        std::array<VkDescriptorSetLayoutBinding, 3> bindings = { uboLayoutBinding, samplerLayoutBinding, matLayoutBinding }; // create an array of our bindings

        // create the descriptor set, take the bindings needed.
        VkDescriptorSetLayoutCreateInfo layoutInfo{};
        layoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
        layoutInfo.bindingCount = static_cast<uint32_t>(bindings.size()); // length of bindings array
        layoutInfo.pBindings = bindings.data(); // data stored in array

        if (vkCreateDescriptorSetLayout(device, &layoutInfo, nullptr, &descriptorSetLayout) != VK_SUCCESS) {
            throw std::runtime_error("failed to create descriptor set layout!");
        }
    }

    // Create a graphics pipeline (This controls shaders, shader inputs, etc). Must be remade for every instance - it is immutable. This loses flexibility for performance.
    void createGraphicsPipeline() {
        // Load the shaders :D
        auto vertShaderCode = readFile("./shaders/vert.spv");
        auto fragShaderCode = readFile("./shaders/frag.spv");

        // Needed to send the vertex information to vulkan so it knows how to use it, and accept it
        auto bindingDescription = Vertex::getBindingDescription();
        auto attributeDescriptions = Vertex::getAttributeDescriptions();

        // Create the shader modules
        VkShaderModule vertShaderModule = createShaderModule(vertShaderCode);
        VkShaderModule fragShaderModule = createShaderModule(fragShaderCode);

        // Create the shader stage - Vertex
        VkPipelineShaderStageCreateInfo vertShaderStageInfo{};
        vertShaderStageInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO; // standard *sigh&
        vertShaderStageInfo.stage = VK_SHADER_STAGE_VERTEX_BIT; // tells vulkan what stage of the pipeline this shader will be used in (Vertex)

        vertShaderStageInfo.module = vertShaderModule; // Load the shader into the object
        vertShaderStageInfo.pName = "main"; // Entry point. This means we can have multiple fragment shaders in one file.
        vertShaderStageInfo.pSpecializationInfo = nullptr; // allows you to define shader constants - more efficient than setting them at render time.



        // Same as vertex, but fragment
        VkPipelineShaderStageCreateInfo fragShaderStageInfo{};
        fragShaderStageInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        fragShaderStageInfo.stage = VK_SHADER_STAGE_FRAGMENT_BIT;

        fragShaderStageInfo.module = fragShaderModule;
        fragShaderStageInfo.pName = "main";
        vertShaderStageInfo.pSpecializationInfo = nullptr;

        VkPipelineShaderStageCreateInfo shaderStages[] = { vertShaderStageInfo, fragShaderStageInfo }; // put the shaders into an array for later use


        // This part lets you define what inputs to send to the vertex shader. This can be very useful!
        VkPipelineVertexInputStateCreateInfo vertexInputInfo{};
        vertexInputInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
        vertexInputInfo.vertexBindingDescriptionCount = 1;
        vertexInputInfo.vertexAttributeDescriptionCount = static_cast<uint32_t>(attributeDescriptions.size());
        vertexInputInfo.pVertexBindingDescriptions = &bindingDescription; // binding descriptions from the vertex struct
        vertexInputInfo.pVertexAttributeDescriptions = attributeDescriptions.data(); // the descriptions from the vertex struct



        // Topology lets you optimize the shaders by reusing verticies, thus saving storage and render time.
        VkPipelineInputAssemblyStateCreateInfo inputAssembly{};
        inputAssembly.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
        inputAssembly.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
        inputAssembly.primitiveRestartEnable = VK_FALSE;


        // The viewport defines what area of the framebuffer to render to. This is almost always (0, 0) and (width, height). We should use the swapchain values in case its different to the screen.
        VkViewport viewport{};
        viewport.x = 0.0f;
        viewport.y = 0.0f;
        viewport.width = (float)swapChainExtent.width;
        viewport.height = (float)swapChainExtent.height;
        viewport.minDepth = 0.0f;
        viewport.maxDepth = 1.0f;

        // Defines what region pixels will be stored. Any pixels outside of the extent will be discarded.
        VkRect2D scissor{};
        scissor.offset = { 0, 0 };
        scissor.extent = swapChainExtent;

        // Build the viewport and scissor. Some GPUS can support multiple, (See logical device creation) so we pass it as an array.
        VkPipelineViewportStateCreateInfo viewportState{};
        viewportState.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
        viewportState.viewportCount = 1;
        viewportState.pViewports = &viewport;
        viewportState.scissorCount = 1;
        viewportState.pScissors = &scissor;


        // The rasterizer takes info from the vertex shader and creates fragments. It can also do depth testing, face culling and does the scissor.
        // It can also be configured to do wireframe rendering, or fill polygons.
        VkPipelineRasterizationStateCreateInfo rasterizer{};
        rasterizer.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
        rasterizer.depthClampEnable = VK_FALSE; // anything beyond the near and far will be clamped, not discarded if true. Useful for shadow maps. Requires a feature


        rasterizer.rasterizerDiscardEnable = VK_FALSE; // If its set to true, nothing goes through the rasterizer, so nothing gets renderered

        rasterizer.polygonMode = VK_POLYGON_MODE_FILL; // See the main comment. Any other mode required a GPU feature enabled.
        rasterizer.lineWidth = 1.0f; // any value larger than one requires a feature

        rasterizer.cullMode = VK_CULL_MODE_NONE; // How to cull (Eg, front, back, both, none)
        rasterizer.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE; // What vertex order is front (Clockwise/Counter clockwise)


        // This can be used for shadow mapping, and typically is, but we dont need it. This can alter the depth values
        rasterizer.depthBiasEnable = VK_FALSE;
        rasterizer.depthBiasConstantFactor = 0.0f; // Optional
        rasterizer.depthBiasClamp = 0.0f; // Optional
        rasterizer.depthBiasSlopeFactor = 0.0f; // Optional

        // Multisampling - works by combining fragment shader result of multiple polygons on the same pixel to smooth lines. Requires feature
        VkPipelineMultisampleStateCreateInfo multisampling{};
        multisampling.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
        multisampling.sampleShadingEnable = VK_FALSE;
        multisampling.rasterizationSamples = msaaSamples;
        multisampling.minSampleShading = 1.0f; // Optional
        multisampling.pSampleMask = nullptr; // Optional
        multisampling.alphaToCoverageEnable = VK_FALSE; // Optional
        multisampling.alphaToOneEnable = VK_FALSE; // Optional

        // Once the fragment shader has done its thing, we must blend the colors of pixels together called Color Blending. We can do this by simply mixing, or mixing old and new with bitwise
        // Very useful if using Alpha (Like transparency)
        VkPipelineColorBlendAttachmentState colorBlendAttachment{}; // Two options for this - One for per framebuffer, and one for global options
        colorBlendAttachment.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
        colorBlendAttachment.blendEnable = VK_FALSE; // Don't use color blending
        colorBlendAttachment.srcColorBlendFactor = VK_BLEND_FACTOR_ONE; // Optional
        colorBlendAttachment.dstColorBlendFactor = VK_BLEND_FACTOR_ZERO; // Optional
        colorBlendAttachment.colorBlendOp = VK_BLEND_OP_ADD; // Optional
        colorBlendAttachment.srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE; // Optional
        colorBlendAttachment.dstAlphaBlendFactor = VK_BLEND_FACTOR_ZERO; // Optional
        colorBlendAttachment.alphaBlendOp = VK_BLEND_OP_ADD; // Optional

        VkPipelineColorBlendStateCreateInfo colorBlending{};
        colorBlending.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
        colorBlending.logicOpEnable = VK_FALSE; // Enable the second option (bitwise) blend. This will disable the above meth
        colorBlending.logicOp = VK_LOGIC_OP_COPY; // Optional
        colorBlending.attachmentCount = 1;
        colorBlending.pAttachments = &colorBlendAttachment;
        colorBlending.blendConstants[0] = 0.0f; // Optional
        colorBlending.blendConstants[1] = 0.0f; // Optional
        colorBlending.blendConstants[2] = 0.0f; // Optional
        colorBlending.blendConstants[3] = 0.0f; // Optional


        // Depth stencil creation
        VkPipelineDepthStencilStateCreateInfo depthStencil{};
        depthStencil.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
        depthStencil.depthTestEnable = VK_TRUE; // enable depth test
        depthStencil.depthWriteEnable = VK_TRUE; // enable deph write
        // Optional depth bound test
        depthStencil.depthBoundsTestEnable = VK_FALSE;
        depthStencil.minDepthBounds = 0.0f; // Optional
        depthStencil.maxDepthBounds = 1.0f; // Optional
        // Stencil buffer  operations
        depthStencil.stencilTestEnable = VK_FALSE;
        depthStencil.front = {}; // Optional
        depthStencil.back = {}; // Optional

        depthStencil.depthCompareOp = VK_COMPARE_OP_LESS; // keep closer objects, discard further away


        // Dynamic states can be used to modify some aspects of the pipeline
        VkDynamicState dynamicStates[] = {
            VK_DYNAMIC_STATE_VIEWPORT,
            VK_DYNAMIC_STATE_LINE_WIDTH
        };

        // Create info for the dynamic state
        VkPipelineDynamicStateCreateInfo dynamicState{};
        dynamicState.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
        dynamicState.dynamicStateCount = 2;
        dynamicState.pDynamicStates = dynamicStates;

        

        // Create the pipeline layout
        VkPipelineLayoutCreateInfo pipelineLayoutInfo{};
        pipelineLayoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
        pipelineLayoutInfo.setLayoutCount = 1;
        pipelineLayoutInfo.pSetLayouts = &descriptorSetLayout; // use our descriptor set layout

        // Push constants is another way of sending data to shaders
        pipelineLayoutInfo.pushConstantRangeCount = 0; // Optional
        pipelineLayoutInfo.pPushConstantRanges = nullptr; // Optional

        if (vkCreatePipelineLayout(device, &pipelineLayoutInfo, nullptr, &pipelineLayout) != VK_SUCCESS) {
            throw std::runtime_error("failed to create pipeline layout!");
        }

        // Create the pipeline info.
        VkGraphicsPipelineCreateInfo pipelineInfo{};
        pipelineInfo.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
        pipelineInfo.stageCount = 2; // We have two shader stages, vert and frag
        pipelineInfo.pStages = shaderStages; // pass the shader stages we made in this function

        // Reference everything in the fixed function stage
        pipelineInfo.pVertexInputState = &vertexInputInfo; // pass the vertex input info (Constant values for the shader)
        pipelineInfo.pInputAssemblyState = &inputAssembly; // pass the input assembly
        pipelineInfo.pViewportState = &viewportState; // pass the viewport state 
        pipelineInfo.pRasterizationState = &rasterizer; // pass the rasterizer
        pipelineInfo.pMultisampleState = &multisampling; // pass the multisampling settings
        pipelineInfo.pDepthStencilState = &depthStencil; // Optional - depth stencil settings
        pipelineInfo.pColorBlendState = &colorBlending; // pass our color blending settings
        pipelineInfo.pDynamicState = nullptr; // Optional

        // Pass the pipeline layout - It is a vulkan handle rather than a pointer.
        pipelineInfo.layout = pipelineLayout;

        // finally, pass the render pass and the subpass that will be used (In our case, 0 which is the output of the frag shader!!!)
        pipelineInfo.renderPass = renderPass;
        pipelineInfo.subpass = 0;

        // You can use this to create a pipeline based off another one already made, as it would be less expensive. As we only have one, lets not use it
        pipelineInfo.basePipelineHandle = VK_NULL_HANDLE; // Optional
        pipelineInfo.basePipelineIndex = -1; // Optional


        // Build it! The VK_NULL_HANDLE can be used to store the pipeline in cache, which can speedup creation times
        if (vkCreateGraphicsPipelines(device, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &graphicsPipeline) != VK_SUCCESS) {
            throw std::runtime_error("failed to create graphics pipeline!");
        }

        // We can destroy the shader modules once we've made the pipeline, as we no longer need it.
        vkDestroyShaderModule(device, fragShaderModule, nullptr);
        vkDestroyShaderModule(device, vertShaderModule, nullptr);
    }
    
    // Create framebuffers, which are the result to send to the render pass, then the GPU
    void createFramebuffers() {
        swapChainFramebuffers.resize(swapChainImageViews.size()); // Resize it to fit the swapchain image views.

        // Loop through the swapchain image views
        for (size_t i = 0; i < swapChainImageViews.size(); i++) {
            std::array<VkImageView, 3> attachments = {
                
            colorImageView,
            depthImageView,
            swapChainImageViews[i],
            
            }; // Check what attachments we need 

            VkFramebufferCreateInfo framebufferInfo{}; // Framebuffer info
            framebufferInfo.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
            framebufferInfo.renderPass = renderPass; // What render pass do we need to be compatiable with
            framebufferInfo.attachmentCount = static_cast<uint32_t>(attachments.size());
            framebufferInfo.pAttachments = attachments.data(); // Attachements from image views
            framebufferInfo.width = swapChainExtent.width; // Width of image - matches swapchain
            framebufferInfo.height = swapChainExtent.height; // Height of image - matches swapchain
            framebufferInfo.layers = 1; // num of layers in image arrays

            if (vkCreateFramebuffer(device, &framebufferInfo, nullptr, &swapChainFramebuffers[i]) != VK_SUCCESS) { // Create the framebuffer images.
                throw std::runtime_error("failed to create framebuffer!");
            }
        }
    }

    // Create the command pool, which takes command buffers. Command buffers hold the commands to be sent to the GPU. It is faster to send 100 commands at once, than 100 commands one at a time.
    void createCommandPool() {
        // Commands are created by submitting them through the device queue. We can get this through the findQueueFamilies function, referencing the physicalDevice.
        QueueFamilyIndices queueFamilyIndices = findQueueFamilies(physicalDevice);

        VkCommandPoolCreateInfo poolInfo{};
        poolInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
        poolInfo.queueFamilyIndex = queueFamilyIndices.graphicsFamily.value();
        poolInfo.flags = 0; // Optional - can be used to change commands when needed. Flags can be either all change, or individual changes.

        if (vkCreateCommandPool(device, &poolInfo, nullptr, &commandPool) != VK_SUCCESS) {
            throw std::runtime_error("failed to create command pool!");
        }
    }

    // Create the depth image, buffer and allocate the memory required. Setup the resources.
    void createDepthResources() {
        VkFormat depthFormat = findDepthFormat();

        // As the depth buffer is an image, we have to create an image with the right format, width and height as the swapchain, and properties we need. We can directly allocate this to the device local bit (fastest memory) as we won't write anything from the CPU
        createImage(swapChainExtent.width, swapChainExtent.height, 1, msaaSamples, depthFormat, VK_IMAGE_TILING_OPTIMAL, VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, depthImage, depthImageMemory);
        // We need the view to interface the image with vulkan, as vulkan cannot interface with VkImage (needs to be a view)
        depthImageView = createImageView(depthImage, depthFormat, VK_IMAGE_ASPECT_DEPTH_BIT, 1);

        transitionImageLayout(depthImage, depthFormat, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL, 1);
    }

    // Find a format that our device supports for images.
    VkFormat findSupportedFormat(const std::vector<VkFormat>& candidates, VkImageTiling tiling, VkFormatFeatureFlags features) {
        for (VkFormat format : candidates) {

            VkFormatProperties props;
            vkGetPhysicalDeviceFormatProperties(physicalDevice, format, &props);

            if (tiling == VK_IMAGE_TILING_LINEAR && (props.linearTilingFeatures & features) == features) {
                return format;
            }
            else if (tiling == VK_IMAGE_TILING_OPTIMAL && (props.optimalTilingFeatures & features) == features) {
                return format;
            }
        }

        throw std::runtime_error("failed to find supported format!");
    }

    // find the best format for our depth that supports what our application requires.
    VkFormat findDepthFormat() {
        return findSupportedFormat(
            { VK_FORMAT_D32_SFLOAT, VK_FORMAT_D32_SFLOAT_S8_UINT, VK_FORMAT_D24_UNORM_S8_UINT },
            VK_IMAGE_TILING_OPTIMAL,
            VK_FORMAT_FEATURE_DEPTH_STENCIL_ATTACHMENT_BIT
        );
    }

    // Lets us know if our selected format supports stencil
    bool hasStencilComponent(VkFormat format) {
        return format == VK_FORMAT_D32_SFLOAT_S8_UINT || format == VK_FORMAT_D24_UNORM_S8_UINT;
    }

    // Create color resources for MSAA
    void createColorResources() {
        VkFormat colorFormat = swapChainImageFormat;

        createImage(swapChainExtent.width, swapChainExtent.height, 1, msaaSamples, colorFormat, VK_IMAGE_TILING_OPTIMAL, VK_IMAGE_USAGE_TRANSIENT_ATTACHMENT_BIT | VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, colorImage, colorImageMemory);
        colorImageView = createImageView(colorImage, colorFormat, VK_IMAGE_ASPECT_COLOR_BIT, 1);
    }

    // Create a texture from an image. This can be used for well, textures.
    void createTextureImage(std::string path) {
        // Load an image. Store the width, height and channels, and the pixels
        int texWidth, texHeight, texChannels;
        stbi_uc* pixels = stbi_load(path.c_str(), &texWidth, &texHeight, &texChannels, STBI_rgb_alpha); // Last input forces an alpha channel, even if the image doesn't have one (4 channel)

        mipLevels = static_cast<uint32_t>(std::floor(std::log2(std::max(texWidth, texHeight)))) + 1;
        // Get the image size so we can make buffers
        VkDeviceSize imageSize = texWidth * texHeight * 4;

        // If the image is empty, then panic!
        if (!pixels) {
            throw std::runtime_error("failed to load texture image!");
        }

        // Create a staging buffer to load the image into the GPU memory. We use a staging buffer so we can move the image texture data to faster, but cpu-inaccessable memory
        VkBuffer stagingBuffer;
        VkDeviceMemory stagingBufferMemory;

        // Create a buffer, where we will use it to transfer bits, and the properties define what properties we want from the memory - something CPU accessable. We also pass the buffer and the memory for the buffer
        createBuffer(imageSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, stagingBuffer, stagingBufferMemory);

        // Next, we map the memory of the memory buffer to the image size, and some empty data
        void* data;
        vkMapMemory(device, stagingBufferMemory, 0, imageSize, 0, &data);
        // We copy pixels to data, so its now in the memory
        memcpy(data, pixels, static_cast<size_t>(imageSize));
        // Unmap the memory so we remain memory safe.
        vkUnmapMemory(device, stagingBufferMemory);

        // Unload the image.
        stbi_image_free(pixels);

        // Although we can pass the raw pixel data to the shader, it is much better and faster to write it to a texture image.
        // It allows us to use 2d coordinates. Pixels within an image are known as *texels*

        // Create the image using the parameters we want
        createImage(texWidth, texHeight, mipLevels, VK_SAMPLE_COUNT_1_BIT, VK_FORMAT_R8G8B8A8_SRGB, VK_IMAGE_TILING_OPTIMAL, VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, textureImage, textureImageMemory);
        // Change the layout to the most optimal layout
        transitionImageLayout(textureImage, VK_FORMAT_R8G8B8A8_SRGB, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, mipLevels);
        // Copy our staging buffer (With the image data) to the image.
        copyBufferToImage(stagingBuffer, textureImage, static_cast<uint32_t>(texWidth), static_cast<uint32_t>(texHeight));

        // Make it shader readable and generate mipmaps
        generateMipmaps(textureImage, VK_FORMAT_R8G8B8A8_SRGB, texWidth, texHeight, mipLevels);

        // Destroy our staging buffer
        vkDestroyBuffer(device, stagingBuffer, nullptr);
        vkFreeMemory(device, stagingBufferMemory, nullptr);
    }

    // This function generates mipmaps. Mipmaps help save vram by rendering a lower quality texture when a model is further away
    void generateMipmaps(VkImage image, VkFormat imageFormat, int32_t texWidth, int32_t texHeight, uint32_t mipLevels) {
        // Check the properties of the device to make sure we can support mipmapping (linear blitting)
        VkFormatProperties formatProperties;
        vkGetPhysicalDeviceFormatProperties(physicalDevice, imageFormat, &formatProperties);
        // Error :(
        if (!(formatProperties.optimalTilingFeatures & VK_FORMAT_FEATURE_SAMPLED_IMAGE_FILTER_LINEAR_BIT)) {
            throw std::runtime_error("texture image format does not support linear blitting!");
        }

        // Run the helper function to start recording commands to a command buffer
        VkCommandBuffer commandBuffer = beginSingleTimeCommands();

        // Create a memory barrier. This will be global for every mipmap level.
        VkImageMemoryBarrier barrier{};
        barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
        barrier.image = image;
        barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED; // don't care about what queue it comes from
        barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED; // same
        barrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT; // it is a color bit, so define it
        barrier.subresourceRange.baseArrayLayer = 0; // only needed if using a texture array (like 3d images)
        barrier.subresourceRange.layerCount = 1;
        barrier.subresourceRange.levelCount = 1;

        // Values for current mip width and height
        int32_t mipWidth = texWidth;
        int32_t mipHeight = texHeight;

        // iterate through every mipLevel
        for (uint32_t i = 1; i < mipLevels; i++) {
            // Using the global barrier, define new parameters. Set the base mip level for our new texture, and move it from a write to a read access mask
            barrier.subresourceRange.baseMipLevel = i - 1;
            barrier.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
            barrier.newLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
            barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
            barrier.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;

            // Add the barrier as a command, and set the required values
            vkCmdPipelineBarrier(commandBuffer,
                VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0,
                0, nullptr,
                0, nullptr,
                1, &barrier);

            // This will wait on the command above (barrier) to finish
            VkImageBlit blit{};
            blit.srcOffsets[0] = { 0, 0, 0 }; // offsets for the source image.
            blit.srcOffsets[1] = { mipWidth, mipHeight, 1 }; // the width of the source image
            blit.srcSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
            blit.srcSubresource.mipLevel = i - 1; // what mip level the source image is
            blit.srcSubresource.baseArrayLayer = 0;
            blit.srcSubresource.layerCount = 1;

            blit.dstOffsets[0] = { 0, 0, 0 }; // can be used to offset the image
            blit.dstOffsets[1] = { mipWidth > 1 ? mipWidth / 2 : 1, mipHeight > 1 ? mipHeight / 2 : 1, 1 }; // keep halving the mip levels until the mipWidth (width of the mip mapped image) is less or equal to 1
            blit.dstSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT; // has to match the image aspect mask
            blit.dstSubresource.mipLevel = i; // what mip level the destination image is
            blit.dstSubresource.baseArrayLayer = 0;
            blit.dstSubresource.layerCount = 1;

            // Submit the blit command. Image is the same for both source and dest images as we're blitting different levels of the same image
            vkCmdBlitImage(commandBuffer,
                image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                1, &blit,
                VK_FILTER_LINEAR);
            
            // We need the mipmap to be in a shader readable format, so redefine the barrier layouts to match.
            barrier.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
            barrier.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
            barrier.srcAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
            barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;

            // Submit this to the command buffer using the barrier.
            vkCmdPipelineBarrier(commandBuffer,
                VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, 0,
                0, nullptr,
                0, nullptr,
                1, &barrier);

            // Divide the mip by half if its bigger than 1
            if (mipWidth > 1) mipWidth /= 2;
            if (mipHeight > 1) mipHeight /= 2;

        }

        // Before we end the recording, quickly change the mip levels of the last mip to be a SHADER_READ_ONLY_OPTIMAL. This is because the loop never converts the last mip,
        // so we have to do it,
        barrier.subresourceRange.baseMipLevel = mipLevels - 1;
        barrier.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
        barrier.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;

        // Submit it
        vkCmdPipelineBarrier(commandBuffer,
            VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, 0,
            0, nullptr,
            0, nullptr,
            1, &barrier);

        // End the recording and submit the buffer to be run on the device.
        endSingleTimeCommands(commandBuffer);
    }

    // Create a VkImage, and assign all the values we need and memory
    void createImage(uint32_t width, uint32_t height, uint32_t mipLevels, VkSampleCountFlagBits numSamples, VkFormat format, VkImageTiling tiling, VkImageUsageFlags usage, VkMemoryPropertyFlags properties, VkImage& image, VkDeviceMemory& imageMemory) {
        // Creation info for images
        VkImageCreateInfo imageInfo{};
        imageInfo.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
        imageInfo.imageType = VK_IMAGE_TYPE_2D; // What coordinate system to use for the image. You can have 1D, 2D and 3D images.
        // Extent is the dimensions of the image - eg width, height and depth
        imageInfo.extent.width = static_cast<uint32_t>(width);
        imageInfo.extent.height = static_cast<uint32_t>(height);
        imageInfo.extent.depth = 1; // must be 1
        imageInfo.mipLevels = mipLevels; // mipmapping is when you use lower quality textures to save VRAM and render speed.
        imageInfo.arrayLayers = 1; // we are only using one image, not an array

        imageInfo.format = format; // format of the image, should match the format we loaded with otherwise the copy operation will fail.

        imageInfo.tiling = tiling; // Linear means you can directly access it from memory, whereas optimal optimizes it for the shader

        imageInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED; // just whether to keep it or discard it

        // Similar to buffer creation. here, we want to transfer from the staging buffer.
        imageInfo.usage = usage;

        // As we're using it for graphics only, and we only have one queue, make it exclusive.
        imageInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

        imageInfo.samples = numSamples; // for multisampling - only used for attachments
        imageInfo.flags = 0; // Optional - useful for 3d images and saving memory

        // Create the image. One of the errors may be the format, however R8G8B8A8 is so widespread this is very unlikely
        if (vkCreateImage(device, &imageInfo, nullptr, &image) != VK_SUCCESS) {
            throw std::runtime_error("failed to create image!");
        }

        // Get the memory requirements for the image
        VkMemoryRequirements memRequirements;
        vkGetImageMemoryRequirements(device, image, &memRequirements);

        // allocation information for the image. Similar to buffer creation. Select local device for best perfomance
        VkMemoryAllocateInfo allocInfo{};
        allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
        allocInfo.allocationSize = memRequirements.size;
        allocInfo.memoryTypeIndex = findMemoryType(memRequirements.memoryTypeBits, properties);

        // allocate the memory
        if (vkAllocateMemory(device, &allocInfo, nullptr, &imageMemory) != VK_SUCCESS) {
            throw std::runtime_error("failed to allocate image memory!");
        }

        // bind the image memory to the image.
        vkBindImageMemory(device, image, imageMemory, 0);
    }

    // Abstraction to create a buffer. Takes in the various things needed, then will setup a buffer and allocate the memory automatically
    void createBuffer(VkDeviceSize size, VkBufferUsageFlags usage, VkMemoryPropertyFlags properties, VkBuffer& buffer, VkDeviceMemory& bufferMemory) {
        VkBufferCreateInfo bufferInfo{};
        bufferInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
        bufferInfo.size = size; // size of the buffer
        bufferInfo.usage = usage; // usage of buffer - ours is for verticies

        bufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE; // this will only be used by the graphics queue, so we can say its exclusive

        if (vkCreateBuffer(device, &bufferInfo, nullptr, &buffer) != VK_SUCCESS) {
            throw std::runtime_error("failed to create vertex buffer!");
        }

        VkMemoryRequirements memRequirements;
        vkGetBufferMemoryRequirements(device, buffer, &memRequirements); // get the memory requirements we need

        // Allocation info so we can allocate the memory needed
        VkMemoryAllocateInfo allocInfo{};
        allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
        allocInfo.allocationSize = memRequirements.size;
        allocInfo.memoryTypeIndex = findMemoryType(memRequirements.memoryTypeBits, properties);

        // Allocate the memory based off the allocInfo
        if (vkAllocateMemory(device, &allocInfo, nullptr, &bufferMemory) != VK_SUCCESS) {
            throw std::runtime_error("failed to allocate buffer memory!");
        }

        vkBindBufferMemory(device, buffer, bufferMemory, 0); // Bind to the buffer
    }

    // helpful abstraction to copy one buffer to another
    void copyBuffer(VkBuffer srcBuffer, VkBuffer dstBuffer, VkDeviceSize size) {
        // This is a GPU command, so we need to do something similar to creating commandBuffers, where we send a command through a queue to move this data around :D
        VkCommandBuffer commandBuffer = beginSingleTimeCommands();

        // add the copy command to the command buffer
        VkBufferCopy copyRegion{};
        copyRegion.size = size;
        vkCmdCopyBuffer(commandBuffer, srcBuffer, dstBuffer, 1, &copyRegion);

        endSingleTimeCommands(commandBuffer);

    }

    // Builds a command buffer for single-time usage, like copying
    VkCommandBuffer beginSingleTimeCommands() {
        // Allocate a command buffer. make it primary (as we want to run it directly) and attach the command pool
        VkCommandBufferAllocateInfo allocInfo{};
        allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
        allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        allocInfo.commandPool = commandPool;
        allocInfo.commandBufferCount = 1;

        // allocate the buffers
        VkCommandBuffer commandBuffer;
        vkAllocateCommandBuffers(device, &allocInfo, &commandBuffer);

        // We only want to submit it once, and use it once, so flag that
        VkCommandBufferBeginInfo beginInfo{};
        beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;

        // begin recording
        vkBeginCommandBuffer(commandBuffer, &beginInfo);

        return commandBuffer;
    }

    // Submits the command buffer for single-time usage buffers.
    void endSingleTimeCommands(VkCommandBuffer commandBuffer) {
        // end the recording
        vkEndCommandBuffer(commandBuffer);

        // submit info - attach the buffers
        VkSubmitInfo submitInfo{};
        submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
        submitInfo.commandBufferCount = 1;
        submitInfo.pCommandBuffers = &commandBuffer;

        // submit to the graphics queue that we got from the device, no need for fences
        vkQueueSubmit(graphicsQueue, 1, &submitInfo, VK_NULL_HANDLE);
        // wait for the command to execute
        vkQueueWaitIdle(graphicsQueue);
        // free up memory
        vkFreeCommandBuffers(device, commandPool, 1, &commandBuffer);
    }

    // This will change the image layout for usage with the shader. This is done with a pipeline barrier. This can be used to synchronise resources, but can also be used to change image layouts and change queue ownership. 
    // There is an equivalent for buffers.
    void transitionImageLayout(VkImage image, VkFormat format, VkImageLayout oldLayout, VkImageLayout newLayout, uint32_t mipLevels) {
        VkCommandBuffer commandBuffer = beginSingleTimeCommands();

        // create the barrier
        VkImageMemoryBarrier barrier{};
        barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
        barrier.oldLayout = oldLayout; // what was the old layout
        barrier.newLayout = newLayout; // layout to change to

        // Must be hardset! these use the queue indices to transfer queue ownership. Currently, we are only using a graphics queue.
        barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;

        // Specify what image is affected
        barrier.image = image;
        // What parts of the image are going to be used.
        if (newLayout == VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL) {
            barrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT;

            if (hasStencilComponent(format)) {
                barrier.subresourceRange.aspectMask |= VK_IMAGE_ASPECT_STENCIL_BIT;
            }
        }
        else {
            barrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        }
        barrier.subresourceRange.baseMipLevel = 0; // no mipmapping
        barrier.subresourceRange.levelCount = mipLevels;
        barrier.subresourceRange.baseArrayLayer = 0;
        barrier.subresourceRange.layerCount = 1; // not an array

        VkPipelineStageFlags sourceStage;
        VkPipelineStageFlags destinationStage;

        // Check if we're changing layout from valid options
        if (oldLayout == VK_IMAGE_LAYOUT_UNDEFINED && newLayout == VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL) {
            barrier.srcAccessMask = 0; // set the source mast
            barrier.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT; // desired mask

            sourceStage = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT; // source stage
            destinationStage = VK_PIPELINE_STAGE_TRANSFER_BIT; // desired stage
        }
        else if (oldLayout == VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL && newLayout == VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL) {
            barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
            barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;

            sourceStage = VK_PIPELINE_STAGE_TRANSFER_BIT;
            destinationStage = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
        }
        else if (oldLayout == VK_IMAGE_LAYOUT_UNDEFINED && newLayout == VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL) {
            barrier.srcAccessMask = 0;
            barrier.dstAccessMask = VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_READ_BIT | VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;

            sourceStage = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;
            destinationStage = VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT;
        }
        else {
            throw std::invalid_argument("unsupported layout transition!");
        }

        vkCmdPipelineBarrier(
            commandBuffer,
            sourceStage, destinationStage,
            0,
            0, nullptr,
            0, nullptr,
            1, &barrier
        );

        endSingleTimeCommands(commandBuffer);
    }

    // Copies a buffer to an image.
    void copyBufferToImage(VkBuffer buffer, VkImage image, uint32_t width, uint32_t height) {
        VkCommandBuffer commandBuffer = beginSingleTimeCommands();


        VkBufferImageCopy region{};
        region.bufferOffset = 0; // offset for the buffer
        region.bufferRowLength = 0; // Used for padding in the image (Which we don't have)
        region.bufferImageHeight = 0; // same as row length

        
        region.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        region.imageSubresource.mipLevel = 0;
        region.imageSubresource.baseArrayLayer = 0;
        region.imageSubresource.layerCount = 1;

        // which part of the image to copy to, eg offsets and extents
        region.imageOffset = { 0, 0, 0 };
        region.imageExtent = {
            width,
            height,
            1
        };

        // Create the command. We can define what layout to use, however here it is hardcoded as OPTIMAL
        vkCmdCopyBufferToImage(
            commandBuffer,
            buffer,
            image,
            VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
            1,
            &region
        );

        endSingleTimeCommands(commandBuffer);
    }

    // create an image view. Graphics pipelines cannot access images directly, and must access an image through a view.
    void createTextureImageView() {
        // Very similar to createImageViews. Just change the image and format to match. No need to define components, as it is set to 0 by default
        textureImageView = createImageView(textureImage, VK_FORMAT_R8G8B8A8_SRGB, VK_IMAGE_ASPECT_COLOR_BIT, mipLevels);
    }

    // Abstraction to create image views
    VkImageView createImageView(VkImage image, VkFormat format, VkImageAspectFlags aspectFlags, uint32_t mipLevels) {
        VkImageViewCreateInfo createInfo{}; // Image view creation - another struct
        createInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
        createInfo.image = image;

        createInfo.viewType = VK_IMAGE_VIEW_TYPE_2D; // how to treat the image (eg, 1d, 2d, 3d etc)
        createInfo.format = format; // defined during swapchain creation (srgb)

        // This part lets you change color channels around (Eg, monochrome -> red)
        createInfo.components.r = VK_COMPONENT_SWIZZLE_IDENTITY;
        createInfo.components.g = VK_COMPONENT_SWIZZLE_IDENTITY;
        createInfo.components.b = VK_COMPONENT_SWIZZLE_IDENTITY;
        createInfo.components.a = VK_COMPONENT_SWIZZLE_IDENTITY;

        // This part defines what the image will be used for. This will be used for colors only, so we do not use mipmapping (See google), and define it as a color bit.
        createInfo.subresourceRange.aspectMask = aspectFlags;
        createInfo.subresourceRange.baseMipLevel = 0;
        createInfo.subresourceRange.levelCount = mipLevels;
        createInfo.subresourceRange.baseArrayLayer = 0;
        createInfo.subresourceRange.layerCount = 1;

        VkImageView imageView;
        // Finally, create the view.
        if (vkCreateImageView(device, &createInfo, nullptr, &imageView) != VK_SUCCESS) {
            throw std::runtime_error("failed to create image views!");
        }

        return imageView;
    }

    // Samplers can apply fliters to the image to reduce artifacts and create a smoother, nicer image (like bilinear sampling), and they can also apply transformations to a texture, like repeating and clamping
    void createTextureSampler() {
        // Create a sampler
        VkSamplerCreateInfo samplerInfo{};
        samplerInfo.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
        samplerInfo.magFilter = VK_FILTER_LINEAR; // magFilter solves oversampling
        samplerInfo.minFilter = VK_FILTER_LINEAR; // minFilter solves undersampling
        // U, V, W are conventional for textures. This defines what to do when going beyond the texture coordinates
        samplerInfo.addressModeU = VK_SAMPLER_ADDRESS_MODE_REPEAT;
        samplerInfo.addressModeV = VK_SAMPLER_ADDRESS_MODE_REPEAT;
        samplerInfo.addressModeW = VK_SAMPLER_ADDRESS_MODE_REPEAT; 
       

        // Makes the image sharper at angles
        samplerInfo.anisotropyEnable = VK_TRUE;
        samplerInfo.maxAnisotropy = 16.0f;

        samplerInfo.borderColor = VK_BORDER_COLOR_INT_OPAQUE_BLACK; // what color to render if we go beyond the address mode, and its set to clamp

        // Useful for percentage-close filtering, on shadow maps. Texels will be compared to a value which will be used for filtering.
        samplerInfo.compareEnable = VK_FALSE;
        samplerInfo.compareOp = VK_COMPARE_OP_ALWAYS;

        // Mipmapping is another filter we can apply to Textures
        samplerInfo.mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR;
        samplerInfo.minLod = 0.0f; // Optional
        samplerInfo.maxLod = static_cast<float>(mipLevels);
        samplerInfo.mipLodBias = 0.0f; // Optional

        if (vkCreateSampler(device, &samplerInfo, nullptr, &textureSampler) != VK_SUCCESS) {
            throw std::runtime_error("failed to create texture sampler!");
        }
    }

    // Vertex buffers contain the verticies we will send to the GPU
    void createVertexBuffer() {
        VkDeviceSize bufferSize = sizeof(vertices[0]) * vertices.size();

        // The staging buffer is used as a temporary buffer to move values from the CPU to device local memory, then we will move the values to a more optimized type of memory as the CPU cannot access this memory on dGPU
        VkBuffer stagingBuffer;
        VkDeviceMemory stagingBufferMemory;
        createBuffer(bufferSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, stagingBuffer, stagingBufferMemory);

        // we can now add data to our buffer
        void* data;
        vkMapMemory(device, stagingBufferMemory, 0, bufferSize, 0, &data);
        memcpy(data, vertices.data(), (size_t)bufferSize);
        vkUnmapMemory(device, stagingBufferMemory);

        // device-local buffer, that is more optimized.
        createBuffer(bufferSize, VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_VERTEX_BUFFER_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, vertexBuffer, vertexBufferMemory);

        // Copy the staging buffer to the vertex buffer
        copyBuffer(stagingBuffer, vertexBuffer, bufferSize);

        // Make sure to free up the staging buffer afterwards
        vkDestroyBuffer(device, stagingBuffer, nullptr);
        vkFreeMemory(device, stagingBufferMemory, nullptr);
    }

    // Nearly identical to createVertexBuffer, except we're using the index buffer. Also the usage of this buffer is index instead of vertex
    void createIndexBuffer() {
        VkDeviceSize bufferSize = sizeof(indices[0]) * indices.size();

        VkBuffer stagingBuffer;
        VkDeviceMemory stagingBufferMemory;
        createBuffer(bufferSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, stagingBuffer, stagingBufferMemory);

        void* data;
        vkMapMemory(device, stagingBufferMemory, 0, bufferSize, 0, &data);
        memcpy(data, indices.data(), (size_t)bufferSize);
        vkUnmapMemory(device, stagingBufferMemory);

        createBuffer(bufferSize, VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_INDEX_BUFFER_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, indexBuffer, indexBufferMemory);

        copyBuffer(stagingBuffer, indexBuffer, bufferSize);

        vkDestroyBuffer(device, stagingBuffer, nullptr);
        vkFreeMemory(device, stagingBufferMemory, nullptr);
    }

    // Create the uniform buffers. Very similar to vertex and index, except its an array (As these values can be changed during runtime quite a bit)
    void createUniformBuffers() {
        VkDeviceSize bufferSize = sizeof(UniformBufferObject);

        uniformBuffers.resize(swapChainImages.size());
        uniformBuffersMemory.resize(swapChainImages.size());

        for (size_t i = 0; i < swapChainImages.size(); i++) {
            // Create a buffer. We don't use staging as we want to be able to modify this buffer. We will also send data to the memory when we modify it in the drawFrame, so no need to memcpy
            createBuffer(bufferSize, VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, uniformBuffers[i], uniformBuffersMemory[i]);
        }
    }

    void createMaterialBuffers() {
       VkDeviceSize bufferSize = sizeof(Material);

        materialBuffers.resize(swapChainImages.size());
        materialBuffersMemory.resize(swapChainImages.size());

        for (size_t i = 0; i < swapChainImages.size(); i++) {
            // Create a buffer. We don't use staging as we want to be able to modify this buffer. We will also send data to the memory when we modify it in the drawFrame, so no need to memcpy
            createBuffer(bufferSize, VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, materialBuffers[i], materialBuffersMemory[i]);
        }
        
    }
    
    // Finds what kind of memory we need to use for the buffer. The different types can change allowed operations/performance.
    uint32_t findMemoryType(uint32_t typeFilter, VkMemoryPropertyFlags properties) {
        VkPhysicalDeviceMemoryProperties memProperties;
        vkGetPhysicalDeviceMemoryProperties(physicalDevice, &memProperties); // Ger physical properties about memory and store it

        for (uint32_t i = 0; i < memProperties.memoryTypeCount; i++) {
            if ((typeFilter & (1 << i)) && (memProperties.memoryTypes[i].propertyFlags & properties) == properties) { // if the type matches, return i. We need to check it can support our required properties, can be written to etc
                return i;
            }
        }

        throw std::runtime_error("failed to find suitable memory type!"); // nothing matches

    }

    // Creates command buffers, which hold commands to go to the GPU
    void createCommandBuffers() {
        commandBuffers.resize(swapChainFramebuffers.size()); // We need a buffer for each framebuffer, as we have to bind the exact VkFramebuffer

        // Buffers are made with the following struct.
        VkCommandBufferAllocateInfo allocInfo{};
        allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
        allocInfo.commandPool = commandPool; // command pool, which we made
        allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY; // Primary can be submitted directly, but cannot be called from other buffers. Secondary cannot be submitted directly but can be called by primary buffers. Useful for reusing buffers
        allocInfo.commandBufferCount = (uint32_t)commandBuffers.size(); // how many buffers to allocate

        // Allocate them!
        if (vkAllocateCommandBuffers(device, &allocInfo, commandBuffers.data()) != VK_SUCCESS) {
            throw std::runtime_error("failed to allocate command buffers!");
        }

        // Begin recording command buffers. This means submitting data to them
        for (size_t i = 0; i < commandBuffers.size(); i++) { // loop through them
            VkCommandBufferBeginInfo beginInfo{};
            beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
            beginInfo.flags = 0; // Optional - defines how we are going to use the buffer - eg rerecording it once we've executed it, or rerecording it while its being executed.
            beginInfo.pInheritanceInfo = nullptr; // Optional - for secondary command buffers to see what state to inherit from primary command buffers.
            // Create them
            if (vkBeginCommandBuffer(commandBuffers[i], &beginInfo) != VK_SUCCESS) {
                throw std::runtime_error("failed to begin recording command buffer!");
            }

            // Begin the render pass
            VkRenderPassBeginInfo renderPassInfo{};
            renderPassInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
            renderPassInfo.renderPass = renderPass; // pass the renderpass
            renderPassInfo.framebuffer = swapChainFramebuffers[i]; // pass the current framebuffer
            // This part defines the size of the render area, like a viewport.
            renderPassInfo.renderArea.offset = { 0, 0 };
            renderPassInfo.renderArea.extent = swapChainExtent;
            // This defines what values to clear with as we defined in the render pass to clear the screen.
            std::array<VkClearValue, 2> clearValues{}; // clear values
            clearValues[0].color = { 0.0f, 0.0f, 0.0f, 1.0f }; // clear the screen black
            clearValues[1].depthStencil = { 1.0f, 0 }; // clear the depth
            // add it to the built render pass
            renderPassInfo.clearValueCount = static_cast<uint32_t>(clearValues.size());
            renderPassInfo.pClearValues = clearValues.data();

            // We now begin the render pass, passing the command buffer and the render pass we just did. The last value can control what buffer executes it, whether it is secondary or primary
            vkCmdBeginRenderPass(commandBuffers[i], &renderPassInfo, VK_SUBPASS_CONTENTS_INLINE);

            // We can now bind the pipeline to the buffer. The second option defines what type of pipeline it is - in our case, graphics. Alternatives include compute.
            vkCmdBindPipeline(commandBuffers[i], VK_PIPELINE_BIND_POINT_GRAPHICS, graphicsPipeline);

            // Send the vertexBuffer through the queue along with the offsets (Which we do not offset). This part binds the buffer to the bindings we setup in the vertex struct.
            VkBuffer vertexBuffers[] = { vertexBuffer };
            VkDeviceSize offsets[] = { 0 };
            vkCmdBindVertexBuffers(commandBuffers[i], 0, 1, vertexBuffers, offsets);

            // bind the index buffer to the command. Similar to vertex, we need the index type (which is controlled by the type up in the const). Can be 16 bit or 32 bit.
            vkCmdBindIndexBuffer(commandBuffers[i], indexBuffer, 0, VK_INDEX_TYPE_UINT32);

            // Bind the descriptor sets to the command buffer
            vkCmdBindDescriptorSets(commandBuffers[i], VK_PIPELINE_BIND_POINT_GRAPHICS, pipelineLayout, 0, 1, &descriptorSets[i], 0, nullptr);
            


            // Tell vulkan to draw a triangle. We pass:
            // Command buffer
            // Index count
            // Instance count - 1 to disable
            // First index - can be used to offset
            // first vertex - can be used to offset
            // first instance - can also be used to offset
            vkCmdDrawIndexed(commandBuffers[i], static_cast<uint32_t>(indices.size()), 1, 0, 0, 0);

            // We can now end the render pass
            vkCmdEndRenderPass(commandBuffers[i]);

            // And end the recording of the command buffer
            if (vkEndCommandBuffer(commandBuffers[i]) != VK_SUCCESS) {
                throw std::runtime_error("failed to record command buffer!");
            }
        }
    }

    // We need semaphores to tell us when we've got the image from the swapchain and is ready to render, and once we've rendered the image and its ready to be stored
    void createSyncObjects() {
        // Creates semaphores and fences, which can be used to wait on the GPU and CPU to finish rendering or whatever theyre doing :D
        // Prevents misuse of memory and allocated things, eg frames. Helps when you need to get a frame and you want to pass it to the render pass
        // but don't have it yet. This stops that from breaking everything. We make one for each frame we can have in flight.
        imageAvailableSemaphores.resize(MAX_FRAMES_IN_FLIGHT);
        renderFinishedSemaphores.resize(MAX_FRAMES_IN_FLIGHT);
        inFlightFences.resize(MAX_FRAMES_IN_FLIGHT);
        imagesInFlight.resize(swapChainImages.size(), VK_NULL_HANDLE);


        VkSemaphoreCreateInfo semaphoreInfo{};
        semaphoreInfo.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;

        VkFenceCreateInfo fenceInfo{};
        fenceInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
        fenceInfo.flags = VK_FENCE_CREATE_SIGNALED_BIT; // set it to signalled so when we wait for a signal, it won't cause a memory leak.

        for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
            if (vkCreateSemaphore(device, &semaphoreInfo, nullptr, &imageAvailableSemaphores[i]) != VK_SUCCESS ||
                vkCreateSemaphore(device, &semaphoreInfo, nullptr, &renderFinishedSemaphores[i]) != VK_SUCCESS ||
                vkCreateFence(device, &fenceInfo, nullptr, &inFlightFences[i]) != VK_SUCCESS) {

                throw std::runtime_error("failed to create synchronization objects for a frame!");
            }
        }
    }

    // Cleans up the swapchain so we can make a new one. Only destroys things we need for the swapchain
    void cleanupSwapChain() {
        vkDestroyImageView(device, colorImageView, nullptr);
        vkDestroyImage(device, colorImage, nullptr);
        vkFreeMemory(device, colorImageMemory, nullptr);


        vkDestroyImageView(device, depthImageView, nullptr);
        vkDestroyImage(device, depthImage, nullptr);
        vkFreeMemory(device, depthImageMemory, nullptr);

        for (size_t i = 0; i < swapChainFramebuffers.size(); i++) {
            vkDestroyFramebuffer(device, swapChainFramebuffers[i], nullptr);
        }

        vkFreeCommandBuffers(device, commandPool, static_cast<uint32_t>(commandBuffers.size()), commandBuffers.data());

        vkDestroyPipeline(device, graphicsPipeline, nullptr);
        vkDestroyPipelineLayout(device, pipelineLayout, nullptr);
        vkDestroyRenderPass(device, renderPass, nullptr);

        for (size_t i = 0; i < swapChainImageViews.size(); i++) {
            vkDestroyImageView(device, swapChainImageViews[i], nullptr);
        }

        vkDestroySwapchainKHR(device, swapChain, nullptr);

        for (size_t i = 0; i < swapChainImages.size(); i++) {
            vkDestroyBuffer(device, materialBuffers[i], nullptr);
            vkFreeMemory(device, materialBuffersMemory[i], nullptr);
        }

       for (size_t i = 0; i < swapChainImages.size(); i++) {
            vkDestroyBuffer(device, uniformBuffers[i], nullptr);
            vkFreeMemory(device, uniformBuffersMemory[i], nullptr);
        }
        
       vkDestroyDescriptorPool(device, descriptorPool, nullptr);
    }

    // Recreates the swapchain. This can happen for many reasons, like a window getting resized
    void recreateSwapChain() {
        // Check the width and height of the screen, so if we're minimized we don't run anything and we wait
        int width = 0, height = 0;
        glfwGetFramebufferSize(window, &width, &height);
        while (width == 0 || height == 0) {
            glfwGetFramebufferSize(window, &width, &height);
            glfwWaitEvents();
        }

        vkDeviceWaitIdle(device); // Wait for the GPU/CPU to finish, so we don't touch things we shouldn't

        cleanupSwapChain();

        createSwapChain();
        createImageViews();
        createRenderPass();
        createGraphicsPipeline();
        createColorResources();
        createDepthResources();
        createFramebuffers();
        createUniformBuffers();
        createMaterialBuffers();
        createDescriptorPool();
        createDescriptorSets();
        createCommandBuffers();
    }

    // Before we can use the shaders, we must wrap them in a VkShaderModule object.
    VkShaderModule createShaderModule(const std::vector<char>& code) {
        VkShaderModuleCreateInfo createInfo{};
        createInfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
        createInfo.codeSize = code.size();
        createInfo.pCode = reinterpret_cast<const uint32_t*>(code.data()); // pCode takes uint32, so we change it with reinterpret cast. It also needs a pointer to the buffer

        VkShaderModule shaderModule;
        if (vkCreateShaderModule(device, &createInfo, nullptr, &shaderModule) != VK_SUCCESS) {
            throw std::runtime_error("failed to create shader module!");
        }

        return shaderModule;
    }

    // Create a function to evaluate if a function is suitable. Useful for VR or other intense apps 
    bool isDeviceSuitable(VkPhysicalDevice device) {
        QueueFamilyIndices indices = findQueueFamilies(device);

        bool extensionsSupported = checkDeviceExtensionSupport(device);

        bool swapChainAdequate = false;
        if (extensionsSupported) {
            SwapChainSupportDetails swapChainSupport = querySwapChainSupport(device);
            swapChainAdequate = !swapChainSupport.formats.empty() && !swapChainSupport.presentModes.empty();
        }

        VkPhysicalDeviceFeatures supportedFeatures;
        vkGetPhysicalDeviceFeatures(device, &supportedFeatures);

        return indices.isComplete() && supportedFeatures.samplerAnisotropy;;
    }

    QueueFamilyIndices findQueueFamilies(VkPhysicalDevice device) {
        QueueFamilyIndices indices;
        // Assign index to queue families that could be found

        // Similar to extension and device
        uint32_t queueFamilyCount = 0;
        vkGetPhysicalDeviceQueueFamilyProperties(device, &queueFamilyCount, nullptr);

        std::vector<VkQueueFamilyProperties> queueFamilies(queueFamilyCount);
        vkGetPhysicalDeviceQueueFamilyProperties(device, &queueFamilyCount, queueFamilies.data());

        // We need one that supports VK_QUEUE_GRAPHICS_BIT
        int i = 0;
        for (const auto& queueFamily : queueFamilies) {
            if (queueFamily.queueFlags & VK_QUEUE_GRAPHICS_BIT) {
                indices.graphicsFamily = i; // Set the value to this queue

                VkBool32 presentSupport = false;
                vkGetPhysicalDeviceSurfaceSupportKHR(device, i, surface, &presentSupport); // check this device can present to the screen

                if (presentSupport) {
                    indices.presentFamily = i;
                }
            }
            if (indices.isComplete()) { // Check we have features needed
                break;
            }
            i++;
        }
        return indices;
    }

    // Helper function to get the maximum number of samples the device can support (eg, MSAA)
    VkSampleCountFlagBits getMaxUsableSampleCount() {
        VkPhysicalDeviceProperties physicalDeviceProperties;
        vkGetPhysicalDeviceProperties(physicalDevice, &physicalDeviceProperties);

        VkSampleCountFlags counts = physicalDeviceProperties.limits.framebufferColorSampleCounts & physicalDeviceProperties.limits.framebufferDepthSampleCounts;
        if (counts & VK_SAMPLE_COUNT_64_BIT) { return VK_SAMPLE_COUNT_64_BIT; }
        if (counts & VK_SAMPLE_COUNT_32_BIT) { return VK_SAMPLE_COUNT_32_BIT; }
        if (counts & VK_SAMPLE_COUNT_16_BIT) { return VK_SAMPLE_COUNT_16_BIT; }
        if (counts & VK_SAMPLE_COUNT_8_BIT) { return VK_SAMPLE_COUNT_8_BIT; }
        if (counts & VK_SAMPLE_COUNT_4_BIT) { return VK_SAMPLE_COUNT_4_BIT; }
        if (counts & VK_SAMPLE_COUNT_2_BIT) { return VK_SAMPLE_COUNT_2_BIT; }

        return VK_SAMPLE_COUNT_1_BIT;
    }

    //Check the layers to make sure we have the required validation layers.
    bool checkValidationLayerSupport() {
        // Similar to checking extensions
        uint32_t layerCount;
        vkEnumerateInstanceLayerProperties(&layerCount, nullptr);

        std::vector<VkLayerProperties> availableLayers(layerCount);
        vkEnumerateInstanceLayerProperties(&layerCount, availableLayers.data());

        // Loop through every layer name that is a validation layer
        for (const char* layerName : validationLayers) {
            bool layerFound = false;
            // Check each layer availiable for the properties
            for (const auto& layerProperties : availableLayers) {
                if (strcmp(layerName, layerProperties.layerName) == 0) { // if strings match
                    layerFound = true;
                    break;
                }
            }

            // exit if it isnt there
            if (!layerFound) {
                return false;
            }
        }

        return true;
    }

    // Helper function to get the required extenstions we need. This can be helpful for setting up custom error messages, as not all messages are fatal.
    std::vector<const char*> getRequiredExtensions() {
        uint32_t glfwExtensionCount = 0;
        const char** glfwExtensions;
        glfwExtensions = glfwGetRequiredInstanceExtensions(&glfwExtensionCount);

        std::vector<const char*> extensions(glfwExtensions, glfwExtensions + glfwExtensionCount);

        if (enableValidationLayers) {
            extensions.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
        }

        return extensions;
    }

    // Build a debug callback function that is static. VKAPI_ATTR and VKAPI_CALL ensure it is called correctly.
    static VKAPI_ATTR VkBool32 VKAPI_CALL debugCallback(VkDebugUtilsMessageSeverityFlagBitsEXT messageSeverity, VkDebugUtilsMessageTypeFlagsEXT messageType, const VkDebugUtilsMessengerCallbackDataEXT* pCallbackData, void* pUserData) {
        std::cout << "\n";
        std::cerr << "validation layer: " << pCallbackData->pMessage << std::endl;

        return VK_FALSE;
    }

    // Setup the debug messenger (Debug - as release shouldn't have this!)
    void setupDebugMessenger() {
        if (!enableValidationLayers) return;

        VkDebugUtilsMessengerCreateInfoEXT createInfo;
        populateDebugMessengerCreateInfo(createInfo);

        if (CreateDebugUtilsMessengerEXT(instance, &createInfo, nullptr, &debugMessenger) != VK_SUCCESS) {
            throw std::runtime_error("failed to set up debug messenger!");
        }
    }

    // setupDebugManager helper func
    void populateDebugMessengerCreateInfo(VkDebugUtilsMessengerCreateInfoEXT& createInfo) {
        createInfo = {};
        createInfo.sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT;
        createInfo.messageSeverity = VK_DEBUG_UTILS_MESSAGE_SEVERITY_VERBOSE_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT;
        createInfo.messageType = VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT;
        createInfo.pfnUserCallback = debugCallback;
    }


    // Check if we have all the extensions we need by matching availiable extensions and the ones we defined in the const
    bool checkDeviceExtensionSupport(VkPhysicalDevice device) {
        uint32_t extensionCount;
        vkEnumerateDeviceExtensionProperties(device, nullptr, &extensionCount, nullptr);

        std::vector<VkExtensionProperties> availableExtensions(extensionCount);
        vkEnumerateDeviceExtensionProperties(device, nullptr, &extensionCount, availableExtensions.data());

        std::set<std::string> requiredExtensions(deviceExtensions.begin(), deviceExtensions.end());

        for (const auto& extension : availableExtensions) {
            requiredExtensions.erase(extension.extensionName);
        }

        return requiredExtensions.empty();
    }
};

int main() {
    HelloTriangleApplication app; // Create our app

    try {
        app.run(); // Run the app
    }
    catch (const std::exception& e) {
        std::cerr << e.what() << std::endl;
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}