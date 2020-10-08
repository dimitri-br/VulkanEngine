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
#include <glm/gtx/rotate_vector.hpp>
#include <stb_image.h> // Used to import images to be used as textures
#include <chrono> // Time
#include <fstream> // Needed to load shaders
#include <array> // array stuff
#include <tiny_obj_loader.h> // load obj
#include <unordered_map> // used to check we aren't adding unnecessary verticies
#include <glm/gtx/hash.hpp> // used to hash


#include "Renderer.h"





int main() {
    Renderer app; // Create our app


    try {
        app.run(); // Run the app
    }
    catch (const std::exception& e) {
        std::cerr << e.what() << std::endl;
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}