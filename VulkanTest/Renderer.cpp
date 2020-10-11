#define GLFW_INCLUDE_VULKAN // GLFW vulkan support (Vulkan is an API not window)
#define GLM_FORCE_RADIANS // Use radians instead of degrees
#define GLM_FORCE_DEPTH_ZERO_TO_ONE
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
#include <glm/gtx/rotate_vector.hpp>
#include <glm/gtx/quaternion.hpp>
#include <glm/gtc/quaternion.hpp>
#include <stb_image.h> // Used to import images to be used as textures
#include <chrono> // Time
#include <fstream> // Needed to load shaders
#include <array> // array stuff
#include <tiny_obj_loader.h> // load obj
#include <unordered_map> // used to check we aren't adding unnecessary verticies
#include <glm/gtx/hash.hpp> // used to hash

#include "Physics.h"
#include "object.h" // holds local files
#include "transform.h"
#include "material.h"
#include "Vertex.h"
#include "ubo.h"
#include "lightTypes.h"
#include "Light.h"
#include "Renderer.h"
#include "btBulletDynamicsCommon.h"


#define DEPTH_FORMAT VK_FORMAT_D16_UNORM
#define DEFAULT_SHADOWMAP_FILTER VK_FILTER_LINEAR

bool firstMouse = true; 
float lastX = WIDTH / 2.0f;
float lastY = HEIGHT / 2.0f;

// Hash function for the vertex struct
namespace std {
    template<> struct hash<Vertex> {
        size_t operator()(Vertex const& vertex) const {
            return ((hash<glm::vec3>()(vertex.pos) ^ (hash<glm::vec3>()(vertex.color) << 1)) ^ (hash<glm::vec2>()(vertex.texCoord) << 1) >> 1) ^ (hash<glm::vec3>()(vertex.normal) << 1);
        }
    };
}


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




// Run!
void Renderer::run() {
    initWindow();
    initVulkan();
    mainLoop();
    cleanup();
}


//Initialize window - all required parameters!
void Renderer::initWindow() {
    glfwInit(); // Initialize glfw (The thing we will use to render the window)

    glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API); // Default is OpenGL - Lets make it use no API (so we can setup vulkan)
    glfwWindowHint(GLFW_RESIZABLE, GLFW_TRUE); //  resizable
    if (FULLSCREEN) {
        window = glfwCreateWindow(WIDTH, HEIGHT, "Vulkan", glfwGetPrimaryMonitor(), nullptr); // Create the window object

    }
    else {
        window = glfwCreateWindow(WIDTH, HEIGHT, "Vulkan", nullptr, nullptr); // Create the window object

    }
    glfwSetWindowUserPointer(window, this); // set a pointer to this class so we can use it in static functions where we pass a window
    glfwSetFramebufferSizeCallback(window, framebufferResizeCallback);
    glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_DISABLED);
    glfwSetCursorPosCallback(window, mouse_callback);
    glfwSetKeyCallback(window, key_callback);

    glfwSwapInterval(1);
}

// This function will be called whenever the screen is resized, along with the new width and height
void Renderer::framebufferResizeCallback(GLFWwindow* window, int width, int height) {
    auto app = reinterpret_cast<Renderer*>(glfwGetWindowUserPointer(window));
    app->framebufferResized = true;
}
// get mouse input
void Renderer::mouse_callback(GLFWwindow* window, double xpos, double ypos)
{
    auto app = reinterpret_cast<Renderer*>(glfwGetWindowUserPointer(window));
    float yaw = app->yaw;
    float pitch = app->pitch;

    if (firstMouse)
    {
        lastX = xpos;
        lastY = ypos;
        firstMouse = false;
    }

    float xoffset = xpos - lastX;
    float yoffset = lastY - ypos;
    lastX = xpos;
    lastY = ypos;

    float sensitivity = 0.1f;
    xoffset *= sensitivity;
    yoffset *= sensitivity;

    yaw += xoffset;
    pitch += yoffset;

    if (pitch > 89.0f)
        pitch = 89.0f;
    if (pitch < -89.0f)
        pitch = -89.0f;

    glm::vec3 direction;
    direction.x = cos(glm::radians(yaw)) * cos(glm::radians(pitch));
    direction.y = sin(glm::radians(pitch));
    direction.z = sin(glm::radians(yaw)) * cos(glm::radians(pitch));
    app->cameraFront = glm::normalize(direction);
    app->yaw = yaw;
    app->pitch = pitch;
}
// get keyboard input
void Renderer::key_callback(GLFWwindow* window, int key, int scancode, int action, int mods)
{
    auto app = reinterpret_cast<Renderer*>(glfwGetWindowUserPointer(window));



    glm::vec3 front = app->cameraFront;
    glm::vec3 pos = app->cameraPos;
    glm::vec3 right = glm::normalize(glm::cross(front, glm::vec3(0.0f, 1.0f, 0.0f)));
    float deltaTime = app->deltaTime;
    float velocity = 10.0f * deltaTime;

    if (key == GLFW_KEY_W && (action == GLFW_REPEAT || action == GLFW_PRESS)) {
        pos += front * velocity;
    }
    if (key == GLFW_KEY_S && (action == GLFW_REPEAT || action == GLFW_PRESS)) {
        pos -= front * velocity;
    }
    if (key == GLFW_KEY_A && (action == GLFW_REPEAT || action == GLFW_PRESS)) {
        pos -= right * velocity;
    }
    if (key == GLFW_KEY_D && (action == GLFW_REPEAT || action == GLFW_PRESS)) {
        pos += right * velocity;
    }

    app->cameraPos = pos;
}

// Create a surface - this is how vulkan interfaces with the window for renderering. Can affect Device selection, so we need to do it first! Is optional.
void Renderer::createSurface() {
    if (glfwCreateWindowSurface(instance, window, nullptr, &surface) != VK_SUCCESS) {
        throw std::runtime_error("failed to create window surface!");
    }
}

Renderer::SwapChainSupportDetails Renderer::querySwapChainSupport(VkPhysicalDevice device) {
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
VkSurfaceFormatKHR Renderer::chooseSwapSurfaceFormat(const std::vector<VkSurfaceFormatKHR>& availableFormats) {
    for (const auto& availableFormat : availableFormats) {
        if (availableFormat.format == VK_FORMAT_B8G8R8A8_SRGB && availableFormat.colorSpace == VK_COLOR_SPACE_SRGB_NONLINEAR_KHR) {
            return availableFormat;
        }
    }

    return availableFormats[0];
}

// Choose best present mode
VkPresentModeKHR Renderer::chooseSwapPresentMode(const std::vector<VkPresentModeKHR>& availablePresentModes) {
    for (const auto& availablePresentMode : availablePresentModes) {
        if (availablePresentMode == VK_PRESENT_MODE_MAILBOX_KHR) {
            return availablePresentMode;
        }
    }

    return VK_PRESENT_MODE_FIFO_KHR;
}

// Resolution of swapchain images (Think of swapchain = buffer). Almost always equal to res of window
VkExtent2D Renderer::chooseSwapExtent(const VkSurfaceCapabilitiesKHR& capabilities) {
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
void Renderer::initVulkan() {
    createPhysics();


    createInstance(); // See the comment at createInstance
    setupDebugMessenger(); // See comment
    createSurface(); // see comment
    pickPhysicalDevice(); // See comment

    createLogicalDevice(); // See comment

    createSwapChain(); // *sigh*


    createImageViews(); // Get the idea?

    createRenderPass();


    createDescriptorSetLayout();



    //createLight(lightType::Directional, lightUpdate::Realtime, glm::vec3(50.0f, 20.0f, 50.0f), glm::vec3(1.0f, 1.0f, 1.0f));
    createLight(lightType::Spot, lightUpdate::Realtime, glm::vec3(15.0f, 2.0f, 15.0f), glm::vec3(1.0f, 1.0f, 1.0f));

    createCommandPool();

    createColorResources();

    createDepthResources();

    createFramebuffers();



    prepareShadowFramebuffer(&lights[0]);
    
    
    


    createGraphicsPipeline(); // Come on!
    prepareShadowGraphicsPipeline();



    createOffscreenBuffer();
    createUniformBuffers();

    //
    

    createObjects();
    

    createCommandBuffers();

    createSyncObjects();
}

void Renderer::createObjects() {

    createObject("./models/cube.obj", "./textures/texture.png", "./textures/texture.png", glm::vec3(0, 7, 0), glm::vec3(0.785398f, 0.0f, 0.785398f), glm::vec3(1, 1, 1), true);

    createObject("./models/cube.obj", "./textures/texture.png", "./textures/texture.png", glm::vec3(2, 7, 0), glm::vec3(0.0f, 0.0f, 0.0f), glm::vec3(1, 1, 1), true);

    createObject("./models/cube.obj", "./textures/texture.png", "./textures/texture.png", glm::vec3(-2, 5, 0), glm::vec3(0.785398f, 0.0f, 0.0f), glm::vec3(1, 1, 1), true);

    createObject("./models/cube.obj", "./textures/texture.png", "./textures/texture.png", glm::vec3(-2, 8, 0), glm::vec3(0.785398f, 0.0f, 0.785398f), glm::vec3(1, 1, 1), true);

    createObject("./models/cube.obj", "./textures/texture.png", "./textures/texture.png", glm::vec3(0, 9, 0), glm::vec3(0.0f, 0.0f, 0.785398f), glm::vec3(1, 1, 1), true);

    createObject("./models/cube.obj", "./textures/texture.png", "./textures/texture.png", glm::vec3(2, 10, 0), glm::vec3(0.0f, 0.0f, 0.785398f), glm::vec3(1, 1, 1), true);

    createObject("./models/cube.obj", "./textures/texture.png", "./textures/texture.png", glm::vec3(-2, 2, 0), glm::vec3(0.785398f, 0.0f, 0.785398f), glm::vec3(1, 1, 1), true);

    createObject("./models/cube.obj", "./textures/texture.png", "./textures/texture.png", glm::vec3(-2, 15, 0), glm::vec3(0.0f, 0.0f, 0.0f), glm::vec3(1, 1, 1), true);

    createObject("./models/cube.obj", "./textures/texture.png", "./textures/texture.png", glm::vec3(-2.5, 4, 0), glm::vec3(0.0f, 0.0f, 0.0f), glm::vec3(1, 1, 1), true);

    createObject("./models/cube.obj", "./textures/texture.png", "./textures/texture.png", glm::vec3(0, 0, 0), glm::vec3(0.0f, 0.0f, 0.0f), glm::vec3(5, 0.25f, 5), false);
}
void Renderer::createPhysics() {
    physics = Physics(); // create the physics object
    physics.init(btVector3(0, -9.81f, 0)); // initialize it, set gravity
}
//Main loop
void Renderer::mainLoop() {

    while (!glfwWindowShouldClose(window)) { // While the window shouldn't close
        glfwPollEvents(); // Loop!
            // Get the time

        physics.dynamicsWorld->stepSimulation(1.f / 60.f, 10);
        //print positions of all objects. Helps to debug if there are any issues.
        /*for (int j= physics.dynamicsWorld ->getNumCollisionObjects() -1; j>=0 ;j--)  {
            btCollisionObject* obj = physics.dynamicsWorld ->getCollisionObjectArray()[j];  // get all the collision objects
            btRigidBody* body = btRigidBody::upcast(obj);  // get the rigidbody for the object
            btTransform trans; // define a transform
            if (body && body ->getMotionState()){ 
                body ->getMotionState()->getWorldTransform(trans); // get transform from the body
            } 
            else {
                trans = obj->getWorldTransform(); // get transform from the object
            } 
            printf("world pos object %d = %f,%f,%f\n",j,float(trans.getOrigin().getX()),float( trans.getOrigin().getY()),float(trans.getOrigin().getZ())); //print world pos
        } */
        drawFrame();
        calculateDeltaTime();
    }

    vkDeviceWaitIdle(device); // make sure we're finished rendering before destroying the window
}

void Renderer::calculateDeltaTime() {
    auto currentTime = std::chrono::high_resolution_clock::now(); // Get the time
    float time = std::chrono::duration<float, std::chrono::seconds::period>(currentTime - lastFrameTime).count(); // Check how much time has passed
    lastFrameTime = currentTime;
    deltaTime = time;
}

// drawFrame will do the following operations - Aqiuire an image from the swapchain, execute a command buffer then return the image to the swapchain for presentation
void Renderer::drawFrame() {
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

    updateTransforms(imageIndex);
    updateOffscreenBuffer(imageIndex);
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


void Renderer::updateOffscreenBuffer(uint32_t currentImage) {
    // All values set on light creation. TODO: add dynamic light movement
    void* sData;
    vkMapMemory(device, offscreenBuffersMemory[currentImage], 0, sizeof(depthMVP), 0, &sData);
    memcpy(sData, &depthMVP, sizeof(depthMVP));
    vkUnmapMemory(device, offscreenBuffersMemory[currentImage]);
}

// Updates the uniform buffer depending on the current swapchain image
void Renderer::updateUniformBuffer(uint32_t currentImage) {
    static auto startTime = std::chrono::high_resolution_clock::now(); // Get the time

    auto currentTime = std::chrono::high_resolution_clock::now(); // Get the time
    float time = std::chrono::duration<float, std::chrono::seconds::period>(currentTime - startTime).count() / 7; // Check how much time has passed



   // lightFOV = 70.0f * sin(time) + 1;


    // Create a ubo, then rotate the model Z by 5 radians every second
    UniformBufferObject ubo{};


    ubo.model = glm::translate(glm::mat4(1.0f), glm::vec3(0.0f, 0.0f, 0.0f)); // translate the model. vec3 specifies translation. Seems to be (z, x, y)
    //ubo.model *= glm::rotate(glm::mat4(1.0f), time * glm::radians(5.0f), glm::vec3(0.0f, 0.0f, 1.0f));
    for (Light l : lights) {
        ubo.lightPos = l.position;
    }
    ubo.camPos = cameraPos;

    ubo.lightSpace = depthMVP.MVP;



    // How we look at the world (Useful for 3d)
    // Eye pos, center pos and up axis
    ubo.view = glm::lookAt(cameraPos, cameraPos + cameraFront, cameraUp);

    // Perspective - 45 degree vertical FOV, aspect ratio and near and far planes.
    ubo.proj = glm::perspective(glm::radians(45.0f), swapChainExtent.width / (float)swapChainExtent.height, 0.1f, 100.0f);

    // Don't forget to flip Y axis as it was designed for opengl!!!!!
    ubo.proj[1][1] *= -1;

    // Now all the transformations are done, write it to memory.
    void* data;
    vkMapMemory(device, uniformBuffersMemory[currentImage], 0, sizeof(ubo), 0, &data);
    memcpy(data, &ubo, sizeof(ubo));
    vkUnmapMemory(device, uniformBuffersMemory[currentImage]);
}

// Update the material settings per model
void Renderer::updateMaterialBuffer(uint32_t currentImage) {
    for (Object obj : objects) {

        // write material to memory
        void* data;
        vkMapMemory(device, obj.materialBuffersMemory[currentImage], 0, sizeof(obj.mat), 0, &data);
        memcpy(data, &obj.mat, sizeof(obj.mat));
        vkUnmapMemory(device, obj.materialBuffersMemory[currentImage]);
    }
}

// Update the transform settings per model
void Renderer::updateTransforms(uint32_t currentImage) {
    static auto startTime = std::chrono::high_resolution_clock::now(); // Get the time

    auto currentTime = std::chrono::high_resolution_clock::now(); // Get the time
    float time = std::chrono::duration<float, std::chrono::seconds::period>(currentTime - startTime).count(); // Check how much time has passed
    int i = 0;
    for (Object obj : objects) {

        
        obj.transform = obj.getTransform(); // set the object transform

        // write material to memory
        void* data;
        vkMapMemory(device, obj.transformBuffersMemory[currentImage], 0, sizeof(obj.transform), 0, &data);
        memcpy(data, &obj.transform, sizeof(obj.transform));
        vkUnmapMemory(device, obj.transformBuffersMemory[currentImage]);

        i++;
    }
}

void Renderer::loadModel(Object *obj) {
    tinyobj::attrib_t attrib; // holds the position, normals and texture coordinates
    std::vector<tinyobj::shape_t> shapes; // contains all seperate objects + faces
    std::vector<tinyobj::material_t> materials; //ignore
    std::string warn, err; // warnings and errors while loading the file

    if (!tinyobj::LoadObj(&attrib, &shapes, &materials, &warn, &err, obj->model_path.c_str())) {
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
                uniqueVertices[vertex] = static_cast<uint32_t>(obj->vertices.size());
                obj->vertices.push_back(vertex);
            }
            // add the vertex to the index buffer
            obj->indices.push_back(uniqueVertices[vertex]);

        }
    }
    std::cout << "Finished loading the model" << std::endl;


}

// Helper function to create a light
void Renderer::createLight(lightType type, lightUpdate update, glm::vec3 pos, glm::vec3 rot) {
    // Create a light, calculate views, set MVP and send it to the lights vector
    Light l = Light();
    l.init(type, update, pos, rot);
    l.calculateView();
    lights.push_back(l);

    depthMVP.MVP = l.lightView;
}

// Creates an object and populates it
void Renderer::createObject(std::string model_path, std::string texture_path, std::string normal_path, glm::vec3 position, glm::vec3 rotation, glm::vec3 scale, bool rigidbody) {
    // Define a new object and material
    Object obj{};

    Material mat{};

    // material colors. The below randomizes colors, but we default white
    float r = static_cast <float> (rand()) / static_cast <float> (RAND_MAX);
    float g = static_cast <float> (rand()) / static_cast <float> (RAND_MAX);
    float b = static_cast <float> (rand()) / static_cast <float> (RAND_MAX);
    mat.color = { 1.0f, 1.0f, 1.0f };

    // shininess for specular (OBSOLETE for now)
    mat.shininess = 1.0f;

    // Transformation info for the object
    Transform trans{};
    trans.transform = glm::translate(glm::mat4(1.0f), position); // translate the model. vec3 specifies translation. Seems to be (z, x, y)
    trans.transform *= glm::rotate(glm::mat4(1.0f), rotation.x, glm::vec3(1, 0, 0)); // rotate the object X
    trans.transform *= glm::rotate(glm::mat4(1.0f), rotation.y, glm::vec3(0, 1, 0)); // rotate the object Y
    trans.transform *= glm::rotate(glm::mat4(1.0f), rotation.z, glm::vec3(0, 0, 1)); // rotate the object Z
    trans.transform = glm::scale(trans.transform, scale); // scale the object

    


    //Set default values
    obj.position = position;
    obj.rotation = rotation;
    obj.scale = scale;

    obj.init(model_path, texture_path, normal_path, mat, trans); // initialize our object

    
    btCollisionShape* collider = new btBoxShape(btVector3(obj.scale.x, obj.scale.y, obj.scale.z));
    if (rigidbody)
        obj.setupPhysics(1.0f, collider);
    else
        obj.setupPhysics(0.0f, collider);
    
        
    physics.dynamicsWorld->addRigidBody(obj.rigidBody);

    obj.instance = objects.size();

    // Create texture image for the object
    createTextureImage(obj.texture_path, &obj);
    createTextureImageView(&obj);
    createTextureSampler(&obj);

    // Create normal image for the object
    createNormalImage(obj.normal_path, &obj);
    createNormalImageView(&obj);
    createNormalSampler(&obj);

    // Begin loading and creating the buffers for the object
    loadModel(&obj);
    std::cout << "Model loaded to obj! \n\n\n";

    createVertexBuffer(&obj);
    std::cout << "Vertex buffer created! \n\n\n";

    createIndexBuffer(&obj);
    std::cout << "Index buffer created! \n\n\n";

    createMaterialBuffers(&obj);
    std::cout << "Material buffer created! \n\n\n";

    createTransformBuffers(&obj);
    std::cout << "Transform buffer created! \n\n\n";

    // Resize descriptor sets (Make this more dynamic!)
    if (recreateDescriptorSets) {
        descriptorPool.resize(descriptorSetSize);
        std::cout << "resized pool! \n\n\n";
        createDescriptorPool();
        recreateDescriptorSets = false;
    }

    // Create them!
    createDescriptorSets(&obj, &lights[0]);
    std::cout << "sets created! \n\n\n";
    objects.push_back(obj);


    std::cout << "Created object! \n\n\n";
}

void Renderer::recreateObjects() {
    if (recreateDescriptorSets) {
        descriptorPool.resize(descriptorSetSize);
        std::cout << "resized pool! \n\n\n";
        createDescriptorPool();
        recreateDescriptorSets = false;
    }
    for (Object obj : objects) {

        createMaterialBuffers(&obj);
        createTransformBuffers(&obj);
        createDescriptorSets(&obj, &lights[0]);
        objects[obj.instance] = obj;

    }
}

// Needed to stay memory-safe. Cleans up on exit
void Renderer::cleanup() {
    cleanupSwapChain();



    // destroy our descriptor set
    vkDestroyDescriptorSetLayout(device, descriptorSetLayout, nullptr);
    for (Light l : lights) {
        vkDestroySampler(device, l.shadowSampler, nullptr);
    }
    
    // Destroy the buffer and buffer memory
    for (Object obj : objects) {
        vkDestroyBuffer(device, obj.vertexBuffer, nullptr);
        vkFreeMemory(device, obj.vertexBufferMemory, nullptr);        // Destroy the sampler
        vkDestroySampler(device, obj.textureSampler, nullptr);


        // Destroy the image view
        vkDestroyImageView(device, obj.textureImageView, nullptr);

        // Destroy the image and the imageMemory
        vkDestroyImage(device, obj.textureImage, nullptr);
        vkFreeMemory(device, obj.textureImageMemory, nullptr);

        vkDestroySampler(device, obj.normalSampler, nullptr);


        // Destroy the image view
        vkDestroyImageView(device, obj.normalImageView, nullptr);

        // Destroy the image and the imageMemory
        vkDestroyImage(device, obj.normalImage, nullptr);
        vkFreeMemory(device, obj.normalImageMemory, nullptr);

        vkDestroyBuffer(device, obj.indexBuffer, nullptr);
        vkFreeMemory(device, obj.indexBufferMemory, nullptr);

        
        physics.dynamicsWorld->removeCollisionObject(obj.rigidBody);
        delete obj.collisionShape;
        delete obj.rigidBody;
    }


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

    //cleanup physics objects
    physics.cleanup();


    glfwDestroyWindow(window);

    glfwTerminate();
}

// Create a vulkan instance - This tells vulkan about the app information and connects the app to vulkan
void Renderer::createInstance() {
    if (enableValidationLayers && !checkValidationLayerSupport()) { // Check if we have the validation layers, or panic and leave
        throw std::runtime_error("validation layers requested, but not available!");
    }
    VkApplicationInfo appInfo{};
    appInfo.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
    appInfo.pApplicationName = "VulkanEngine";
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
void Renderer::pickPhysicalDevice() {
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
void Renderer::createLogicalDevice() {
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
void Renderer::createSwapChain() {
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

    createInfo.preTransform = VK_SURFACE_TRANSFORM_IDENTITY_BIT_KHR;
    createInfo.compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;

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
void Renderer::createImageViews() {
    swapChainImageViews.resize(swapChainImages.size()); // Resize all the image views to fit the swapchain images

    for (size_t i = 0; i < swapChainImages.size(); i++) { // Iterate over all the swapchain images
        swapChainImageViews[i] = createImageView(swapChainImages[i], swapChainImageFormat, VK_IMAGE_ASPECT_COLOR_BIT, 1); // See function
    }
}

// Render passes define what attachments will be used by the framebuffer. This can define color and depth buffers, samples and how their contents will be handled.
void Renderer::createRenderPass() {
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
    std::array<VkSubpassDescription, 1> subpass = {};
    subpass[0].pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS; // Must be explicit that is it a graphics pass
    // Pass our attachment to the subpass
    subpass[0].colorAttachmentCount = 1;
    subpass[0].pColorAttachments = &colorAttachmentRef; // attach the color reference
    subpass[0].pDepthStencilAttachment = &depthAttachmentRef; // attach the depth reference
    subpass[0].pResolveAttachments = &colorAttachmentResolveRef; // attach the resolve reference






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
    renderPassInfo.subpassCount = subpass.size();
    renderPassInfo.pSubpasses = subpass.data();
    // This prevents us writing to the image and adding color too early
    renderPassInfo.dependencyCount = 1;
    renderPassInfo.pDependencies = &dependency;

    if (vkCreateRenderPass(device, &renderPassInfo, nullptr, &renderPass) != VK_SUCCESS) {
        throw std::runtime_error("failed to create render pass!");
    }
}

// Set up a separate render pass for the offscreen frame buffer
// This is necessary as the offscreen frame buffer attachments use formats different to those from the example render pass
void Renderer::prepareShadowRenderpass()
{
    VkAttachmentDescription attachmentDescription{};
    attachmentDescription.format = DEPTH_FORMAT;
    attachmentDescription.samples = VK_SAMPLE_COUNT_1_BIT;
    attachmentDescription.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;							// Clear depth at beginning of the render pass
    attachmentDescription.storeOp = VK_ATTACHMENT_STORE_OP_STORE;						// We will read from depth, so it's important to store the depth attachment results
    attachmentDescription.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
    attachmentDescription.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    attachmentDescription.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;					// We don't care about initial layout of the attachment
    attachmentDescription.finalLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL;// Attachment will be transitioned to shader read at render pass end

    VkAttachmentReference depthReference = {};
    depthReference.attachment = 0;
    depthReference.layout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;			// Attachment will be used as depth/stencil during render pass

    VkSubpassDescription subpass = {};
    subpass.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
    subpass.colorAttachmentCount = 0;													// No color attachments
    subpass.pDepthStencilAttachment = &depthReference;									// Reference to our depth attachment

    // Use subpass dependencies for layout transitions
    std::array<VkSubpassDependency, 2> dependencies;

    dependencies[0].srcSubpass = VK_SUBPASS_EXTERNAL;
    dependencies[0].dstSubpass = 0;
    dependencies[0].srcStageMask = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
    dependencies[0].dstStageMask = VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT;
    dependencies[0].srcAccessMask = VK_ACCESS_SHADER_READ_BIT;
    dependencies[0].dstAccessMask = VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
    dependencies[0].dependencyFlags = VK_DEPENDENCY_BY_REGION_BIT;

    dependencies[1].srcSubpass = 0;
    dependencies[1].dstSubpass = VK_SUBPASS_EXTERNAL;
    dependencies[1].srcStageMask = VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT;
    dependencies[1].dstStageMask = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
    dependencies[1].srcAccessMask = VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
    dependencies[1].dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
    dependencies[1].dependencyFlags = VK_DEPENDENCY_BY_REGION_BIT;




    VkRenderPassCreateInfo renderPassCreateInfo{};
    renderPassCreateInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
    renderPassCreateInfo.attachmentCount = 1;
    renderPassCreateInfo.pAttachments = &attachmentDescription;
    renderPassCreateInfo.subpassCount = 1;
    renderPassCreateInfo.pSubpasses = &subpass;
    renderPassCreateInfo.dependencyCount = static_cast<uint32_t>(dependencies.size());
    renderPassCreateInfo.pDependencies = dependencies.data();


    if (vkCreateRenderPass(device, &renderPassCreateInfo, nullptr, &offscreenRenderPass) != VK_SUCCESS) {
        throw std::runtime_error("failed to create render pass!");
    }
}

// Setup an offscreen framebuffer
void Renderer::prepareShadowFramebuffer(Light * light)
{

    // For shadow mapping we only need a depth attachment
    VkImageCreateInfo image{};
    image.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
    image.imageType = VK_IMAGE_TYPE_2D;
    image.extent.width = SHADOWMAP_DIM;
    image.extent.height = SHADOWMAP_DIM;
    image.extent.depth = 1;
    image.mipLevels = 1;
    image.arrayLayers = 1;
    image.samples = VK_SAMPLE_COUNT_1_BIT;
    image.tiling = VK_IMAGE_TILING_OPTIMAL;
    image.format = DEPTH_FORMAT;		// Depth stencil attachment
    image.usage = VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;		// We will sample directly from the depth attachment for the shadow mapping

    if (vkCreateImage(device, &image, nullptr, &light->shadowImage) != VK_SUCCESS) {
        throw std::runtime_error("failed to create offscreen image!");
    }


    // Get the memory requirements for the image
    VkMemoryRequirements memRequirements;
    vkGetImageMemoryRequirements(device, light->shadowImage, &memRequirements);

    // allocation information for the image. Similar to buffer creation. Select local device for best perfomance
    VkMemoryAllocateInfo allocInfo{};
    allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    allocInfo.allocationSize = memRequirements.size;
    allocInfo.memoryTypeIndex = findMemoryType(memRequirements.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);

    if (vkAllocateMemory(device, &allocInfo, nullptr, &light->shadowImageMemory) != VK_SUCCESS) {
        throw std::runtime_error("failed to allocate offscreen image memory!");
    }

    if (vkBindImageMemory(device, light->shadowImage, light->shadowImageMemory, 0) != VK_SUCCESS) {
        throw std::runtime_error("failed to bind offscreen image memory!");
    }


    VkImageViewCreateInfo depthStencilView{};
    depthStencilView.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
    depthStencilView.viewType = VK_IMAGE_VIEW_TYPE_2D;
    depthStencilView.format = DEPTH_FORMAT;
    depthStencilView.subresourceRange = {};
    depthStencilView.subresourceRange.aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT;
    depthStencilView.subresourceRange.baseMipLevel = 0;
    depthStencilView.subresourceRange.levelCount = 1;
    depthStencilView.subresourceRange.baseArrayLayer = 0;
    depthStencilView.subresourceRange.layerCount = 1;
    depthStencilView.image = light->shadowImage;
    if (vkCreateImageView(device, &depthStencilView, nullptr, &light->shadowImageView) != VK_SUCCESS) {
        throw std::runtime_error("failed to create depth stencil image view!");
    }


    // Create sampler to sample from to depth attachment
    // Used to sample in the fragment shader for shadowed rendering



    // Create a sampler
    VkSamplerCreateInfo sampler{};
    sampler.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
    // U, V, W are conventional for textures. This defines what to do when going beyond the texture coordinates

    sampler.magFilter = DEFAULT_SHADOWMAP_FILTER;
    sampler.minFilter = DEFAULT_SHADOWMAP_FILTER;
    sampler.mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR;
    sampler.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    sampler.addressModeV = sampler.addressModeU;
    sampler.addressModeW = sampler.addressModeU;
    sampler.mipLodBias = 0.0f;
    sampler.maxAnisotropy = 1.0f;
    sampler.minLod = 0.0f;
    sampler.maxLod = 1.0f;
    sampler.borderColor = VK_BORDER_COLOR_FLOAT_OPAQUE_WHITE;
    if (vkCreateSampler(device, &sampler, nullptr, &light->shadowSampler) != VK_SUCCESS) {
        throw std::runtime_error("failed to create shadow sampler!");
    }

    prepareShadowRenderpass();

    // Create frame buffer
    VkFramebufferCreateInfo fbufCreateInfo{};
    fbufCreateInfo.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
    fbufCreateInfo.renderPass = offscreenRenderPass;
    fbufCreateInfo.attachmentCount = 1;
    fbufCreateInfo.pAttachments = &light->shadowImageView;
    fbufCreateInfo.width = SHADOWMAP_DIM;
    fbufCreateInfo.height = SHADOWMAP_DIM;
    fbufCreateInfo.layers = 1;

    if (vkCreateFramebuffer(device, &fbufCreateInfo, nullptr, &offscreenFramebuffer) != VK_SUCCESS) {
        throw std::runtime_error("failed to create offscreen framebuffer!");
    }
}


// Descriptor sets cannot be created directly, they must be allocated in descriptor pools.
void Renderer::createDescriptorPool() {
    std::cout << descriptorPool.size();
    for (int i = 0; i < descriptorPool.size(); i++) {
        // we have to allocate one pool for every frame, to prevent conflicts while in-flight. Also include all the descriptors we will use
        std::array<VkDescriptorPoolSize, 8> poolSizes{};
        poolSizes[0].type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        poolSizes[0].descriptorCount = static_cast<uint32_t>(swapChainImages.size());

        poolSizes[1].type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        poolSizes[1].descriptorCount = static_cast<uint32_t>(swapChainImages.size());

        poolSizes[2].type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        poolSizes[2].descriptorCount = static_cast<uint32_t>(swapChainImages.size());

        poolSizes[3].type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        poolSizes[3].descriptorCount = static_cast<uint32_t>(swapChainImages.size());

        poolSizes[4].type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        poolSizes[4].descriptorCount = static_cast<uint32_t>(swapChainImages.size());

        poolSizes[5].type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        poolSizes[5].descriptorCount = static_cast<uint32_t>(swapChainImages.size());

        poolSizes[6].type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        poolSizes[6].descriptorCount = static_cast<uint32_t>(swapChainImages.size());

        poolSizes[7].type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        poolSizes[7].descriptorCount = static_cast<uint32_t>(swapChainImages.size());

        // we must specify the max amount of descriptor sets that can be allocated
        VkDescriptorPoolCreateInfo poolInfo{};
        poolInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
        poolInfo.poolSizeCount = static_cast<uint32_t>(poolSizes.size()); // size of the pool (How many descriptors to use)
        poolInfo.pPoolSizes = poolSizes.data(); // the pool size within the array
        poolInfo.maxSets = static_cast<uint32_t>(swapChainImages.size());

        poolInfo.flags = 0; // optional - this can be used to define whether individual descriptor sets can be freed or not.

        if (vkCreateDescriptorPool(device, &poolInfo, nullptr, &descriptorPool[i]) != VK_SUCCESS) {
            throw std::runtime_error("failed to create descriptor pool!");
        }
    }
}

void Renderer::createOffscreenBuffer() {
    // Matrix from light's point of view

    VkDeviceSize bufferSize = sizeof(uboOffscreenVS);

    offscreenBuffers.resize(swapChainImages.size());
    offscreenBuffersMemory.resize(swapChainImages.size());

    for (size_t i = 0; i < swapChainImages.size(); i++) {
        // Create a buffer. We don't use staging as we want to be able to modify this buffer. We will also send data to the memory when we modify it in the drawFrame, so no need to memcpy
        createBuffer(bufferSize, VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, offscreenBuffers[i], offscreenBuffersMemory[i]);
    }
}

// Create descriptor sets, which can be used by shaders to access buffer data (Like the UBO) or image data.
void Renderer::createDescriptorSets(Object *obj, Light* light) {

    std::vector<VkDescriptorSetLayout> layouts(swapChainImages.size(), descriptorSetLayout);

    VkDescriptorSetAllocateInfo allocInfo{};
    allocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    allocInfo.descriptorPool = descriptorPool[obj->instance];
    allocInfo.descriptorSetCount = static_cast<uint32_t>(swapChainImages.size());
    allocInfo.pSetLayouts = layouts.data();

    obj->descriptorSets.resize(swapChainImages.size());


    if (vkAllocateDescriptorSets(device, &allocInfo, obj->descriptorSets.data()) != VK_SUCCESS) {
        throw std::runtime_error("failed to allocate descriptor sets!");
    }





    for (size_t i = 0; i < swapChainImages.size(); i++) {
        VkDescriptorBufferInfo uboBufferInfo{};
        uboBufferInfo.buffer = uniformBuffers[i];
        uboBufferInfo.offset = 0;
        uboBufferInfo.range = sizeof(UniformBufferObject);

        VkDescriptorImageInfo imageInfo{};
        imageInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        imageInfo.imageView = obj->textureImageView;
        imageInfo.sampler = obj->textureSampler;


        VkDescriptorBufferInfo matBufferInfo{};
        matBufferInfo.buffer = obj->materialBuffers[i];
        matBufferInfo.offset = 0;
        matBufferInfo.range = sizeof(Material);

        VkDescriptorBufferInfo transBufferInfo{};
        transBufferInfo.buffer = obj->transformBuffers[i];
        transBufferInfo.offset = 0;
        transBufferInfo.range = sizeof(Transform);

        VkDescriptorImageInfo offscreenInfo{};
        offscreenInfo.imageLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL;
        offscreenInfo.imageView = light->shadowImageView;
        offscreenInfo.sampler = light->shadowSampler;
        //offscreenInfo.sampler = shadowSampler;

        VkDescriptorBufferInfo shadowBufferInfo{};
        shadowBufferInfo.buffer = offscreenBuffers[i];
        shadowBufferInfo.offset = 0;
        shadowBufferInfo.range = sizeof(uboOffscreenVS);

        VkDescriptorBufferInfo shadowTransBufferInfo{};
        shadowTransBufferInfo.buffer = obj->transformBuffers[i];
        shadowTransBufferInfo.offset = 0;
        shadowTransBufferInfo.range = sizeof(Transform);


        VkDescriptorImageInfo imageNormalInfo{};
        imageNormalInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        imageNormalInfo.imageView = obj->normalImageView;
        imageNormalInfo.sampler = obj->normalSampler;

        // create an array to store our descriptor sets
        std::array<VkWriteDescriptorSet, 8> descriptorWrites{};

        descriptorWrites[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        descriptorWrites[0].dstSet = obj->descriptorSets[i]; // the set to write to ( we jave one for each frame in the swapchain)
        descriptorWrites[0].dstBinding = 0;// the binding in the shader
        descriptorWrites[0].dstArrayElement = 0;
        descriptorWrites[0].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        descriptorWrites[0].descriptorCount = 1;
        descriptorWrites[0].pBufferInfo = &uboBufferInfo; // the buffer to put in the set

        descriptorWrites[1].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        descriptorWrites[1].dstSet = obj->descriptorSets[i];
        descriptorWrites[1].dstBinding = 1; // the binding in the shader
        descriptorWrites[1].dstArrayElement = 0;
        descriptorWrites[1].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        descriptorWrites[1].descriptorCount = 1;
        descriptorWrites[1].pImageInfo = &imageInfo; // the image (the image sampler) to put in the set

        descriptorWrites[2].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        descriptorWrites[2].dstSet = obj->descriptorSets[i]; // the set to write to ( we jave one for each frame in the swapchain)
        descriptorWrites[2].dstBinding = 2;// the binding in the shader
        descriptorWrites[2].dstArrayElement = 0;
        descriptorWrites[2].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        descriptorWrites[2].descriptorCount = 1;
        descriptorWrites[2].pBufferInfo = &matBufferInfo; // the buffer to put in the set

        descriptorWrites[3].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        descriptorWrites[3].dstSet = obj->descriptorSets[i]; // the set to write to ( we jave one for each frame in the swapchain)
        descriptorWrites[3].dstBinding = 3;// the binding in the shader
        descriptorWrites[3].dstArrayElement = 0;
        descriptorWrites[3].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        descriptorWrites[3].descriptorCount = 1;
        descriptorWrites[3].pBufferInfo = &transBufferInfo; // the buffer to put in the set

        descriptorWrites[4].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        descriptorWrites[4].dstSet = obj->descriptorSets[i];
        descriptorWrites[4].dstBinding = 4; // the binding in the shader
        descriptorWrites[4].dstArrayElement = 0;
        descriptorWrites[4].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        descriptorWrites[4].descriptorCount = 1;
        descriptorWrites[4].pImageInfo = &offscreenInfo; // the image (the image sampler) to put in the set

        descriptorWrites[5].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        descriptorWrites[5].dstSet = obj->descriptorSets[i];
        descriptorWrites[5].dstBinding = 5; // the binding in the shader
        descriptorWrites[5].dstArrayElement = 0;
        descriptorWrites[5].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        descriptorWrites[5].descriptorCount = 1;
        descriptorWrites[5].pBufferInfo = &shadowBufferInfo; // the buffer to put in the set

        descriptorWrites[6].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        descriptorWrites[6].dstSet = obj->descriptorSets[i];
        descriptorWrites[6].dstBinding = 6; // the binding in the shader
        descriptorWrites[6].dstArrayElement = 0;
        descriptorWrites[6].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        descriptorWrites[6].descriptorCount = 1;
        descriptorWrites[6].pBufferInfo = &shadowTransBufferInfo; // the buffer to put in the set

        descriptorWrites[7].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        descriptorWrites[7].dstSet = obj->descriptorSets[i];
        descriptorWrites[7].dstBinding = 7; // the binding in the shader
        descriptorWrites[7].dstArrayElement = 0;
        descriptorWrites[7].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        descriptorWrites[7].descriptorCount = 1;
        descriptorWrites[7].pImageInfo = &imageNormalInfo; // the image (the image sampler) to put in the set

        // Update descriptor sets on the device
        vkUpdateDescriptorSets(device, static_cast<uint32_t>(descriptorWrites.size()), descriptorWrites.data(), 0, nullptr);




    }

}

// Automatically creates the descriptor sets.
void Renderer::createDescriptorSetLayout() {
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

    VkDescriptorSetLayoutBinding transLayoutBinding{};
    transLayoutBinding.binding = 3;
    transLayoutBinding.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER; // define it is a uniform buffer
    transLayoutBinding.descriptorCount = 1;
    transLayoutBinding.stageFlags = VK_SHADER_STAGE_VERTEX_BIT; // what stage it will be used on
    transLayoutBinding.pImmutableSamplers = nullptr; // Optional - useful for images

            // Create a binding to a sampler, so the shader can access the image from the sampler
    VkDescriptorSetLayoutBinding offscreenLayoutBinding{};
    offscreenLayoutBinding.binding = 4; // will be binding 4
    offscreenLayoutBinding.descriptorCount = 1;
    offscreenLayoutBinding.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER; // the type of descriptor
    offscreenLayoutBinding.pImmutableSamplers = nullptr;
    offscreenLayoutBinding.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT; // When we will use it - the frag shader

    // Holds light MVP info
    VkDescriptorSetLayoutBinding shadowmapLayoutBinding{};
    shadowmapLayoutBinding.binding = 5;
    shadowmapLayoutBinding.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER; // define it is a uniform buffer
    shadowmapLayoutBinding.descriptorCount = 1;
    shadowmapLayoutBinding.stageFlags = VK_SHADER_STAGE_VERTEX_BIT; // what stage it will be used on
    shadowmapLayoutBinding.pImmutableSamplers = nullptr; // Optional - useful for images

    // Holds transform info for objects for the shadow shader
    VkDescriptorSetLayoutBinding shadowTransLayoutBinding{};
    shadowTransLayoutBinding.binding = 6;
    shadowTransLayoutBinding.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER; // define it is a uniform buffer
    shadowTransLayoutBinding.descriptorCount = 1;
    shadowTransLayoutBinding.stageFlags = VK_SHADER_STAGE_VERTEX_BIT; // what stage it will be used on
    shadowTransLayoutBinding.pImmutableSamplers = nullptr; // Optional - useful for images

    VkDescriptorSetLayoutBinding normalImageBinding{};
    normalImageBinding.binding = 7; // will be binding 7
    normalImageBinding.descriptorCount = 1;
    normalImageBinding.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER; // the type of descriptor
    normalImageBinding.pImmutableSamplers = nullptr;
    normalImageBinding.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT; // When we will use it - the frag shader

    std::array<VkDescriptorSetLayoutBinding, 8> bindings = { uboLayoutBinding, samplerLayoutBinding, matLayoutBinding, transLayoutBinding, offscreenLayoutBinding, shadowmapLayoutBinding, shadowTransLayoutBinding, normalImageBinding }; // create an array of our bindings

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
void Renderer::createGraphicsPipeline() {
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

    depthStencil.depthCompareOp = VK_COMPARE_OP_LESS_OR_EQUAL; // keep closer objects, discard further away


    // Dynamic states can be used to modify some aspects of the pipeline
    std::vector<VkDynamicState> dynamicStates;

    dynamicStates.push_back(VK_DYNAMIC_STATE_VIEWPORT);
    dynamicStates.push_back(VK_DYNAMIC_STATE_LINE_WIDTH);



    // Create info for the dynamic state
    VkPipelineDynamicStateCreateInfo dynamicState{};
    dynamicState.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
    dynamicState.dynamicStateCount = dynamicStates.size();
    dynamicState.pDynamicStates = dynamicStates.data();



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
    pipelineInfo.pDynamicState = &dynamicState; // Optional

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

void Renderer::prepareShadowGraphicsPipeline() {
    // Load the shaders :D
    auto vertShaderCode = readFile("./shaders/shadowVert.spv");


    // Needed to send the vertex information to vulkan so it knows how to use it, and accept it
    auto bindingDescription = Vertex::getBindingDescription();
    auto attributeDescriptions = Vertex::getAttributeDescriptions();

    // Create the shader modules
    VkShaderModule vertShaderModule = createShaderModule(vertShaderCode);

    // Create the shader stage - Vertex
    VkPipelineShaderStageCreateInfo vertShaderStageInfo{};
    vertShaderStageInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO; // standard *sigh&
    vertShaderStageInfo.stage = VK_SHADER_STAGE_VERTEX_BIT; // tells vulkan what stage of the pipeline this shader will be used in (Vertex)

    vertShaderStageInfo.module = vertShaderModule; // Load the shader into the object
    vertShaderStageInfo.pName = "main"; // Entry point. This means we can have multiple fragment shaders in one file.
    vertShaderStageInfo.pSpecializationInfo = nullptr; // allows you to define shader constants - more efficient than setting them at render time.




    VkPipelineShaderStageCreateInfo shaderStages[] = { vertShaderStageInfo }; // put the shaders into an array for later use


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
    rasterizer.depthBiasEnable = VK_TRUE;
    rasterizer.depthBiasConstantFactor = depthBiasConstant; // Optional
    rasterizer.depthBiasSlopeFactor = depthBiasSlope; // Optional

    // Multisampling - works by combining fragment shader result of multiple polygons on the same pixel to smooth lines. Requires feature
    VkPipelineMultisampleStateCreateInfo multisampling{};
    multisampling.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
    multisampling.sampleShadingEnable = VK_FALSE;
    multisampling.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;
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
    colorBlending.attachmentCount = 0;
    colorBlending.pAttachments = nullptr;
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

    depthStencil.depthCompareOp = VK_COMPARE_OP_LESS_OR_EQUAL; // keep closer objects, discard further away


    // Dynamic states can be used to modify some aspects of the pipeline
    std::vector<VkDynamicState> dynamicStates;

    dynamicStates.push_back(VK_DYNAMIC_STATE_VIEWPORT);
    dynamicStates.push_back(VK_DYNAMIC_STATE_SCISSOR);
    dynamicStates.push_back(VK_DYNAMIC_STATE_LINE_WIDTH);
    dynamicStates.push_back(VK_DYNAMIC_STATE_DEPTH_BIAS);



    // Create info for the dynamic state
    VkPipelineDynamicStateCreateInfo dynamicState{};
    dynamicState.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
    dynamicState.dynamicStateCount = dynamicStates.size();
    dynamicState.pDynamicStates = dynamicStates.data();




    // Create the pipeline info.
    VkGraphicsPipelineCreateInfo pipelineInfo{};
    pipelineInfo.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
    pipelineInfo.stageCount = 1; // We have two shader stages, vert and frag
    pipelineInfo.pStages = shaderStages; // pass the shader stages we made in this function

    // Reference everything in the fixed function stage
    pipelineInfo.pVertexInputState = &vertexInputInfo; // pass the vertex input info (Constant values for the shader)
    pipelineInfo.pInputAssemblyState = &inputAssembly; // pass the input assembly
    pipelineInfo.pViewportState = &viewportState; // pass the viewport state 
    pipelineInfo.pRasterizationState = &rasterizer; // pass the rasterizer
    pipelineInfo.pMultisampleState = &multisampling; // pass the multisampling settings
    pipelineInfo.pDepthStencilState = &depthStencil; // Optional - depth stencil settings
    pipelineInfo.pColorBlendState = &colorBlending; // pass our color blending settings
    pipelineInfo.pDynamicState = &dynamicState; // Optional

    // Pass the pipeline layout - It is a vulkan handle rather than a pointer.
    pipelineInfo.layout = pipelineLayout;

    // finally, pass the render pass and the subpass that will be used (In our case, 0 which is the output of the frag shader!!!)
    pipelineInfo.renderPass = offscreenRenderPass;
    pipelineInfo.subpass = 0;

    // You can use this to create a pipeline based off another one already made, as it would be less expensive. As we only have one, lets not use it
    pipelineInfo.basePipelineHandle = VK_NULL_HANDLE; // Optional
    pipelineInfo.basePipelineIndex = -1; // Optional


    // Build it! The VK_NULL_HANDLE can be used to store the pipeline in cache, which can speedup creation times
    if (vkCreateGraphicsPipelines(device, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &offscreenPipeline) != VK_SUCCESS) {
        throw std::runtime_error("failed to create graphics pipeline!");
    }

    // We can destroy the shader modules once we've made the pipeline, as we no longer need it.
    vkDestroyShaderModule(device, vertShaderModule, nullptr);
}

// Create framebuffers, which are the result to send to the render pass, then the GPU
void Renderer::createFramebuffers() {
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
void Renderer::createCommandPool() {
    // Commands are created by submitting them through the device queue. We can get this through the findQueueFamilies function, referencing the physicalDevice.
    QueueFamilyIndices queueFamilyIndices = findQueueFamilies(physicalDevice);

    VkCommandPoolCreateInfo poolInfo{};
    poolInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    poolInfo.queueFamilyIndex = queueFamilyIndices.graphicsFamily.value();
    poolInfo.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT; // Optional - can be used to change commands when needed. Flags can be either all change, or individual changes.

    if (vkCreateCommandPool(device, &poolInfo, nullptr, &commandPool) != VK_SUCCESS) {
        throw std::runtime_error("failed to create command pool!");
    }
}

// Create the depth image, buffer and allocate the memory required. Setup the resources.
void Renderer::createDepthResources() {
    VkFormat depthFormat = findDepthFormat();

    // As the depth buffer is an image, we have to create an image with the right format, width and height as the swapchain, and properties we need. We can directly allocate this to the device local bit (fastest memory) as we won't write anything from the CPU
    createImage(swapChainExtent.width, swapChainExtent.height, 1, msaaSamples, depthFormat, VK_IMAGE_TILING_OPTIMAL, VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, depthImage, depthImageMemory);
    // We need the view to interface the image with vulkan, as vulkan cannot interface with VkImage (needs to be a view)
    depthImageView = createImageView(depthImage, depthFormat, VK_IMAGE_ASPECT_DEPTH_BIT, 1);

    transitionImageLayout(depthImage, depthFormat, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL, 1);
}

// Find a format that our device supports for images.
VkFormat Renderer::findSupportedFormat(const std::vector<VkFormat>& candidates, VkImageTiling tiling, VkFormatFeatureFlags features) {
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
VkFormat Renderer::findDepthFormat() {
    return findSupportedFormat(
        { VK_FORMAT_D32_SFLOAT, VK_FORMAT_D32_SFLOAT_S8_UINT, VK_FORMAT_D24_UNORM_S8_UINT },
        VK_IMAGE_TILING_OPTIMAL,
        VK_FORMAT_FEATURE_DEPTH_STENCIL_ATTACHMENT_BIT
    );
}

// Lets us know if our selected format supports stencil
bool Renderer::hasStencilComponent(VkFormat format) {
    return format == VK_FORMAT_D32_SFLOAT_S8_UINT || format == VK_FORMAT_D24_UNORM_S8_UINT;
}

// Create color resources for MSAA
void Renderer::createColorResources() {
    VkFormat colorFormat = swapChainImageFormat;

    createImage(swapChainExtent.width, swapChainExtent.height, 1, msaaSamples, colorFormat, VK_IMAGE_TILING_OPTIMAL, VK_IMAGE_USAGE_TRANSIENT_ATTACHMENT_BIT | VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, colorImage, colorImageMemory);
    colorImageView = createImageView(colorImage, colorFormat, VK_IMAGE_ASPECT_COLOR_BIT, 1);
}

// Create a texture from an image. This can be used for well, textures.
void Renderer::createTextureImage(std::string path, Object *obj) {
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
    createImage(texWidth, texHeight, mipLevels, VK_SAMPLE_COUNT_1_BIT, VK_FORMAT_R8G8B8A8_SRGB, VK_IMAGE_TILING_OPTIMAL, VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, obj->textureImage, obj->textureImageMemory);
    // Change the layout to the most optimal layout
    transitionImageLayout(obj->textureImage, VK_FORMAT_R8G8B8A8_SRGB, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, mipLevels);
    // Copy our staging buffer (With the image data) to the image.
    copyBufferToImage(stagingBuffer, obj->textureImage, static_cast<uint32_t>(texWidth), static_cast<uint32_t>(texHeight));

    // Make it shader readable and generate mipmaps
    generateMipmaps(obj->textureImage, VK_FORMAT_R8G8B8A8_SRGB, texWidth, texHeight, mipLevels);

    // Destroy our staging buffer
    vkDestroyBuffer(device, stagingBuffer, nullptr);
    vkFreeMemory(device, stagingBufferMemory, nullptr);

}

// Create a texture from an image. This can be used for well, textures.
void Renderer::createNormalImage(std::string path, Object* obj) {
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
    createImage(texWidth, texHeight, mipLevels, VK_SAMPLE_COUNT_1_BIT, VK_FORMAT_R8G8B8A8_SRGB, VK_IMAGE_TILING_OPTIMAL, VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, obj->normalImage, obj->normalImageMemory);
    // Change the layout to the most optimal layout
    transitionImageLayout(obj->normalImage, VK_FORMAT_R8G8B8A8_SRGB, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, mipLevels);
    // Copy our staging buffer (With the image data) to the image.
    copyBufferToImage(stagingBuffer, obj->textureImage, static_cast<uint32_t>(texWidth), static_cast<uint32_t>(texHeight));

    // Make it shader readable and generate mipmaps
    generateMipmaps(obj->normalImage, VK_FORMAT_R8G8B8A8_SRGB, texWidth, texHeight, mipLevels);

    // Destroy our staging buffer
    vkDestroyBuffer(device, stagingBuffer, nullptr);
    vkFreeMemory(device, stagingBufferMemory, nullptr);

}

// This function generates mipmaps. Mipmaps help save vram by rendering a lower quality texture when a model is further away
void Renderer::generateMipmaps(VkImage image, VkFormat imageFormat, int32_t texWidth, int32_t texHeight, uint32_t mipLevels) {
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
void Renderer::createImage(uint32_t width, uint32_t height, uint32_t mipLevels, VkSampleCountFlagBits numSamples, VkFormat format, VkImageTiling tiling, VkImageUsageFlags usage, VkMemoryPropertyFlags properties, VkImage& image, VkDeviceMemory& imageMemory) {
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
void Renderer::createBuffer(VkDeviceSize size, VkBufferUsageFlags usage, VkMemoryPropertyFlags properties, VkBuffer& buffer, VkDeviceMemory& bufferMemory) {
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
void Renderer::copyBuffer(VkBuffer srcBuffer, VkBuffer dstBuffer, VkDeviceSize size) {
    // This is a GPU command, so we need to do something similar to creating commandBuffers, where we send a command through a queue to move this data around :D
    VkCommandBuffer commandBuffer = beginSingleTimeCommands();

    // add the copy command to the command buffer
    VkBufferCopy copyRegion{};
    copyRegion.size = size;
    vkCmdCopyBuffer(commandBuffer, srcBuffer, dstBuffer, 1, &copyRegion);

    endSingleTimeCommands(commandBuffer);

}

// Builds a command buffer for single-time usage, like copying
VkCommandBuffer Renderer::beginSingleTimeCommands() {
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
void Renderer::endSingleTimeCommands(VkCommandBuffer commandBuffer) {
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
void Renderer::transitionImageLayout(VkImage image, VkFormat format, VkImageLayout oldLayout, VkImageLayout newLayout, uint32_t mipLevels) {
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
void Renderer::copyBufferToImage(VkBuffer buffer, VkImage image, uint32_t width, uint32_t height) {
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
void Renderer::createTextureImageView(Object *obj) {
    // Very similar to createImageViews. Just change the image and format to match. No need to define components, as it is set to 0 by default
    obj->textureImageView = createImageView(obj->textureImage, VK_FORMAT_R8G8B8A8_SRGB, VK_IMAGE_ASPECT_COLOR_BIT, mipLevels);

}
// create an image view. Graphics pipelines cannot access images directly, and must access an image through a view.
void Renderer::createNormalImageView(Object* obj) {
    // Very similar to createImageViews. Just change the image and format to match. No need to define components, as it is set to 0 by default
    obj->normalImageView = createImageView(obj->normalImage, VK_FORMAT_R8G8B8A8_SRGB, VK_IMAGE_ASPECT_COLOR_BIT, mipLevels);

}

// Abstraction to create image views
VkImageView Renderer::createImageView(VkImage image, VkFormat format, VkImageAspectFlags aspectFlags, uint32_t mipLevels) {
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
void Renderer::createTextureSampler(Object *obj) {
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

    if (vkCreateSampler(device, &samplerInfo, nullptr, &obj->textureSampler) != VK_SUCCESS) {
        throw std::runtime_error("failed to create texture sampler!");
    }

}

// Samplers can apply fliters to the image to reduce artifacts and create a smoother, nicer image (like bilinear sampling), and they can also apply transformations to a texture, like repeating and clamping
void Renderer::createNormalSampler(Object* obj) {
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

    if (vkCreateSampler(device, &samplerInfo, nullptr, &obj->normalSampler) != VK_SUCCESS) {
        throw std::runtime_error("failed to create texture sampler!");
    }

}

// Vertex buffers contain the verticies we will send to the GPU
void Renderer::createVertexBuffer(Object *obj) {
    VkDeviceSize bufferSize = sizeof(obj->vertices[0]) * obj->vertices.size();

    // The staging buffer is used as a temporary buffer to move values from the CPU to device local memory, then we will move the values to a more optimized type of memory as the CPU cannot access this memory on dGPU
    VkBuffer stagingBuffer;
    VkDeviceMemory stagingBufferMemory;
    createBuffer(bufferSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, stagingBuffer, stagingBufferMemory);

    // we can now add data to our buffer
    void* data;
    vkMapMemory(device, stagingBufferMemory, 0, bufferSize, 0, &data);
    memcpy(data, obj->vertices.data(), (size_t)bufferSize);
    vkUnmapMemory(device, stagingBufferMemory);

    // device-local buffer, that is more optimized.
    createBuffer(bufferSize, VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_VERTEX_BUFFER_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, obj->vertexBuffer, obj->vertexBufferMemory);

    // Copy the staging buffer to the vertex buffer
    copyBuffer(stagingBuffer, obj->vertexBuffer, bufferSize);

    // Make sure to free up the staging buffer afterwards
    vkDestroyBuffer(device, stagingBuffer, nullptr);
    vkFreeMemory(device, stagingBufferMemory, nullptr);
}

// Nearly identical to createVertexBuffer, except we're using the index buffer. Also the usage of this buffer is index instead of vertex
void Renderer::createIndexBuffer(Object *obj) {
    VkDeviceSize bufferSize = sizeof(obj->indices[0]) * obj->indices.size();

    VkBuffer stagingBuffer;
    VkDeviceMemory stagingBufferMemory;
    createBuffer(bufferSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, stagingBuffer, stagingBufferMemory);

    void* data;
    vkMapMemory(device, stagingBufferMemory, 0, bufferSize, 0, &data);
    memcpy(data, obj->indices.data(), (size_t)bufferSize);
    vkUnmapMemory(device, stagingBufferMemory);

    createBuffer(bufferSize, VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_INDEX_BUFFER_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, obj->indexBuffer, obj->indexBufferMemory);

    copyBuffer(stagingBuffer, obj->indexBuffer, bufferSize);

    vkDestroyBuffer(device, stagingBuffer, nullptr);
    vkFreeMemory(device, stagingBufferMemory, nullptr);
}

// Create the uniform buffers. Very similar to vertex and index, except its an array (As these values can be changed during runtime quite a bit)
void Renderer::createUniformBuffers() {
    VkDeviceSize bufferSize = sizeof(UniformBufferObject);

    uniformBuffers.resize(swapChainImages.size());
    uniformBuffersMemory.resize(swapChainImages.size());

    for (size_t i = 0; i < swapChainImages.size(); i++) {
        // Create a buffer. We don't use staging as we want to be able to modify this buffer. We will also send data to the memory when we modify it in the drawFrame, so no need to memcpy
        createBuffer(bufferSize, VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, uniformBuffers[i], uniformBuffersMemory[i]);
    }
}

// Create the material buffers for the object
void Renderer::createMaterialBuffers(Object *obj) {
    VkDeviceSize bufferSize = sizeof(Material);

    obj->materialBuffers.resize(swapChainImages.size());
    obj->materialBuffersMemory.resize(swapChainImages.size());

    for (size_t i = 0; i < swapChainImages.size(); i++) {
        // Create a buffer. We don't use staging as we want to be able to modify this buffer. We will also send data to the memory when we modify it in the drawFrame, so no need to memcpy
        createBuffer(bufferSize, VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, obj->materialBuffers[i], obj->materialBuffersMemory[i]);
    }

}

// Create the transform buffers for the object
void Renderer::createTransformBuffers(Object *obj) {
    VkDeviceSize bufferSize = sizeof(Transform);

    obj->transformBuffers.resize(swapChainImages.size());
    obj->transformBuffersMemory.resize(swapChainImages.size());

    for (size_t i = 0; i < swapChainImages.size(); i++) {
        // Create a buffer. We don't use staging as we want to be able to modify this buffer. We will also send data to the memory when we modify it in the drawFrame, so no need to memcpy
        createBuffer(bufferSize, VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, obj->transformBuffers[i], obj->transformBuffersMemory[i]);
    }

}

// Finds what kind of memory we need to use for the buffer. The different types can change allowed operations/performance.
uint32_t Renderer::findMemoryType(uint32_t typeFilter, VkMemoryPropertyFlags properties) {
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
void Renderer::createCommandBuffers() {
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


    VkViewport viewport;
    VkRect2D scissor;


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

        
        
        

        VkRenderPassBeginInfo renderPassInfo{};
        renderPassInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
        renderPassInfo.renderPass = offscreenRenderPass; // pass the renderpass
        renderPassInfo.framebuffer = offscreenFramebuffer; // pass the current framebuffer
        // This part defines the size of the render area, like a viewport.
        renderPassInfo.renderArea.offset = { 0, 0 };
        renderPassInfo.renderArea.extent.width = SHADOWMAP_DIM;
        renderPassInfo.renderArea.extent.height = SHADOWMAP_DIM;
        // This defines what values to clear with as we defined in the render pass to clear the screen.
        std::array<VkClearValue, 2> clearValues = {};
        clearValues[0].depthStencil = { 1.0f, 0 }; // clear the depth
        // add it to the built render pass
        renderPassInfo.clearValueCount = static_cast<uint32_t>(clearValues.size());
        renderPassInfo.pClearValues = clearValues.data();

        vkCmdBeginRenderPass(commandBuffers[i], &renderPassInfo, VK_SUBPASS_CONTENTS_INLINE);
        std::cout << "\n\n\nStarted offscreen render pass\n\n\n";

        viewport = {};
        viewport.height = SHADOWMAP_DIM;
        viewport.width = SHADOWMAP_DIM;
        viewport.minDepth = 0.0f;
        viewport.maxDepth = 1.0f;

        vkCmdSetViewport(commandBuffers[i], 0, 1, &viewport);

        scissor = {};
        scissor.extent.height = SHADOWMAP_DIM;
        scissor.extent.width = SHADOWMAP_DIM;
        scissor.offset.x = 0.0f;
        scissor.offset.y = 0.0f;

        vkCmdSetScissor(commandBuffers[i], 0, 1, &scissor);

        // Set depth bias (aka "Polygon offset")
        // Required to avoid shadow mapping artifacts
        vkCmdSetDepthBias(
            commandBuffers[i],
            depthBiasConstant,
            0.0f,
            depthBiasSlope);

        vkCmdBindPipeline(commandBuffers[i], VK_PIPELINE_BIND_POINT_GRAPHICS, offscreenPipeline);

        if (lights[0].update != lightUpdate::OnCreate) {
            for (Object obj : objects) {
                vkCmdBindDescriptorSets(commandBuffers[i], VK_PIPELINE_BIND_POINT_GRAPHICS, pipelineLayout, 0, 1, &obj.descriptorSets[i], 0, nullptr);
                // Send the vertexBuffer through the queue along with the offsets (Which we do not offset). This part binds the buffer to the bindings we setup in the vertex struct.
                VkBuffer vertexBuffers[] = { obj.vertexBuffer };
                VkDeviceSize offsets[] = { 0 };
                vkCmdBindVertexBuffers(commandBuffers[i], 0, 1, vertexBuffers, offsets);

                // bind the index buffer to the command. Similar to vertex, we need the index type (which is controlled by the type up in the const). Can be 16 bit or 32 bit.
                vkCmdBindIndexBuffer(commandBuffers[i], obj.indexBuffer, 0, VK_INDEX_TYPE_UINT32);

                // Tell vulkan to draw a triangle. We pass:
                // Command buffer
                // Index count
                // Instance count - 1 to disable
                // First index - can be used to offset
                // first vertex - can be used to offset
                // first instance - can also be used to offset
                vkCmdDrawIndexed(commandBuffers[i], static_cast<uint32_t>(obj.indices.size()), 1, 0, 0, 0);
            }
        }
        vkCmdEndRenderPass(commandBuffers[i]);
            
        lights[0].hasRendered = true;
        
        

        




        std::cout << "\n\n\nStarted onscreen render pass\n\n\n";
        // Begin the actual drawing render pass
        renderPassInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
        renderPassInfo.renderPass = renderPass; // pass the renderpass
        renderPassInfo.framebuffer = swapChainFramebuffers[i]; // pass the current framebuffer
        // This part defines the size of the render area, like a viewport.
        renderPassInfo.renderArea.offset = { 0, 0 };
        renderPassInfo.renderArea.extent = swapChainExtent;
        // This defines what values to clear with as we defined in the render pass to clear the screen.
        clearValues[0].color = { 0.0f, 0.0f, 0.0f, 1.0f }; // clear the screen black
        clearValues[1].depthStencil = { 1.0f, 0 }; // clear the depth
        // add it to the built render pass
        renderPassInfo.clearValueCount = static_cast<uint32_t>(clearValues.size());
        renderPassInfo.pClearValues = clearValues.data();


        // We now begin the render pass, passing the command buffer and the render pass we just did. The last value can control what buffer executes it, whether it is secondary or primary
        vkCmdBeginRenderPass(commandBuffers[i], &renderPassInfo, VK_SUBPASS_CONTENTS_INLINE);

        viewport = {};
        viewport.height = swapChainExtent.height;
        viewport.width = swapChainExtent.width;
        viewport.minDepth = 0.0f;
        viewport.maxDepth = 1.0f;

        vkCmdSetViewport(commandBuffers[i], 0, 1, &viewport);

        scissor = {};
        scissor.extent.height = swapChainExtent.height;
        scissor.extent.width = swapChainExtent.width;
        scissor.offset.x = 0.0f;
        scissor.offset.y = 0.0f;

        vkCmdSetScissor(commandBuffers[i], 0, 1, &scissor);

        // We can now bind the pipeline to the buffer. The second option defines what type of pipeline it is - in our case, graphics. Alternatives include compute.
        vkCmdBindPipeline(commandBuffers[i], VK_PIPELINE_BIND_POINT_GRAPHICS, graphicsPipeline);
        for (Object obj : objects) {
            // Send the vertexBuffer through the queue along with the offsets (Which we do not offset). This part binds the buffer to the bindings we setup in the vertex struct.
            VkBuffer vertexBuffers[] = { obj.vertexBuffer };
            VkDeviceSize offsets[] = { 0 };
            vkCmdBindVertexBuffers(commandBuffers[i], 0, 1, vertexBuffers, offsets);

            // bind the index buffer to the command. Similar to vertex, we need the index type (which is controlled by the type up in the const). Can be 16 bit or 32 bit.
            vkCmdBindIndexBuffer(commandBuffers[i], obj.indexBuffer, 0, VK_INDEX_TYPE_UINT32);

            // Bind the descriptor sets to the command buffer
            vkCmdBindDescriptorSets(commandBuffers[i], VK_PIPELINE_BIND_POINT_GRAPHICS, pipelineLayout, 0, 1, &obj.descriptorSets[i], 0, nullptr);



            // Tell vulkan to draw a triangle. We pass:
            // Command buffer
            // Index count
            // Instance count - 1 to disable
            // First index - can be used to offset
            // first vertex - can be used to offset
            // first instance - can also be used to offset
            vkCmdDrawIndexed(commandBuffers[i], static_cast<uint32_t>(obj.indices.size()), 1, 0, 0, 0);
        }
        // We can now end the render pass
        vkCmdEndRenderPass(commandBuffers[i]);
        std::cout << "\n\n\nFinished onscreen render pass\n\n\n";
        // And end the recording of the command buffer
        if (vkEndCommandBuffer(commandBuffers[i]) != VK_SUCCESS) {
            throw std::runtime_error("failed to record command buffer!");
        }
    }
}

// We need semaphores to tell us when we've got the image from the swapchain and is ready to render, and once we've rendered the image and its ready to be stored
void Renderer::createSyncObjects() {
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
void Renderer::cleanupSwapChain() {
    vkDestroyImageView(device, colorImageView, nullptr);
    vkDestroyImage(device, colorImage, nullptr);
    vkFreeMemory(device, colorImageMemory, nullptr);


    vkDestroyImageView(device, depthImageView, nullptr);
    vkDestroyImage(device, depthImage, nullptr);
    vkFreeMemory(device, depthImageMemory, nullptr);

    for (Light l : lights) {
        vkDestroyImageView(device, l.shadowImageView, nullptr);
        vkDestroyImage(device, l.shadowImage, nullptr);
        vkFreeMemory(device, l.shadowImageMemory, nullptr);
    }


    vkDestroyFramebuffer(device, offscreenFramebuffer, nullptr);

    for (size_t i = 0; i < swapChainFramebuffers.size(); i++) {
        vkDestroyFramebuffer(device, swapChainFramebuffers[i], nullptr);

    }


    vkFreeCommandBuffers(device, commandPool, static_cast<uint32_t>(commandBuffers.size()), commandBuffers.data());

    vkDestroyPipeline(device, graphicsPipeline, nullptr);
    vkDestroyPipeline(device, offscreenPipeline, nullptr);

    vkDestroyPipelineLayout(device, pipelineLayout, nullptr);
    vkDestroyRenderPass(device, renderPass, nullptr);

    vkDestroyRenderPass(device, offscreenRenderPass, nullptr);

    for (size_t i = 0; i < swapChainImageViews.size(); i++) {
        vkDestroyImageView(device, swapChainImageViews[i], nullptr);
    }



    vkDestroySwapchainKHR(device, swapChain, nullptr);

    for (Object obj : objects) {
        for (size_t i = 0; i < swapChainImages.size(); i++) {
            vkDestroyBuffer(device, obj.materialBuffers[i], nullptr);
            vkFreeMemory(device, obj.materialBuffersMemory[i], nullptr);

            vkDestroyBuffer(device, obj.transformBuffers[i], nullptr);
            vkFreeMemory(device, obj.transformBuffersMemory[i], nullptr);
        }
    }


    for (size_t i = 0; i < swapChainImages.size(); i++) {
        vkDestroyBuffer(device, uniformBuffers[i], nullptr);
        vkFreeMemory(device, uniformBuffersMemory[i], nullptr);
    }
    for (size_t i = 0; i < swapChainImages.size(); i++) {
        vkDestroyBuffer(device, offscreenBuffers[i], nullptr);
        vkFreeMemory(device, offscreenBuffersMemory[i], nullptr);
    }
    for (VkDescriptorPool pool : descriptorPool) {
        vkDestroyDescriptorPool(device, pool, nullptr);
    }

    recreateDescriptorSets = true;

}

// Recreates the swapchain. This can happen for many reasons, like a window getting resized
void Renderer::recreateSwapChain() {
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
    prepareShadowFramebuffer(&lights[0]);
    prepareShadowGraphicsPipeline();
    createFramebuffers();
    createOffscreenBuffer();
    createUniformBuffers();

    recreateObjects();

    createCommandBuffers();
}

// Before we can use the shaders, we must wrap them in a VkShaderModule object.
VkShaderModule Renderer::createShaderModule(const std::vector<char>& code) {
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
bool Renderer::isDeviceSuitable(VkPhysicalDevice device) {
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

Renderer::QueueFamilyIndices Renderer::findQueueFamilies(VkPhysicalDevice device) {
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
VkSampleCountFlagBits Renderer::getMaxUsableSampleCount() {
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
bool Renderer::checkValidationLayerSupport() {
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
std::vector<const char*> Renderer::getRequiredExtensions() {
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
VKAPI_ATTR VkBool32 VKAPI_CALL Renderer::debugCallback(VkDebugUtilsMessageSeverityFlagBitsEXT messageSeverity, VkDebugUtilsMessageTypeFlagsEXT messageType, const VkDebugUtilsMessengerCallbackDataEXT* pCallbackData, void* pUserData) {
    std::cout << "\n";
    std::cerr << "validation layer: " << pCallbackData->pMessage << std::endl;

    return VK_FALSE;
}

// Setup the debug messenger (Debug - as release shouldn't have this!)
void Renderer::setupDebugMessenger() {
    if (!enableValidationLayers) return;

    VkDebugUtilsMessengerCreateInfoEXT createInfo;
    populateDebugMessengerCreateInfo(createInfo);

    if (CreateDebugUtilsMessengerEXT(instance, &createInfo, nullptr, &debugMessenger) != VK_SUCCESS) {
        throw std::runtime_error("failed to set up debug messenger!");
    }
}

// setupDebugManager helper func
void Renderer::populateDebugMessengerCreateInfo(VkDebugUtilsMessengerCreateInfoEXT& createInfo) {
    createInfo = {};
    createInfo.sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT;
    createInfo.messageSeverity = VK_DEBUG_UTILS_MESSAGE_SEVERITY_VERBOSE_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT;
    createInfo.messageType = VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT;
    createInfo.pfnUserCallback = debugCallback;
}

// Check if we have all the extensions we need by matching availiable extensions and the ones we defined in the const
bool Renderer::checkDeviceExtensionSupport(VkPhysicalDevice device) {
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
