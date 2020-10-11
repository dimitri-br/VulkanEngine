#define GLFW_INCLUDE_VULKAN // GLFW vulkan support (Vulkan is an API not window)
#include <GLFW/glfw3.h> // Windowing
#include "Light.h"
#include "transform.h"
#include <stdexcept>




void Light::init(lightType lighttype, lightUpdate updateRate, glm::vec3 pos, glm::vec3 rot)
{
    type = lighttype;
    update = updateRate;
    position = pos;
    rotation = rot;
}

void Light::calculateView()
{
    glm::mat4 depthProjectionMatrix;
    glm::mat4 depthViewMatrix = glm::lookAt(position, glm::vec3(0.0f, 0.0f, 0.0f), glm::vec3(0.0f, 1.0f, 0.0f));
    glm::mat4 depthModelMatrix = glm::mat4(1.0f);
    Transform trans{};


    switch (type) {
    case Directional:
        depthProjectionMatrix = glm::ortho(-10.0f, 10.0f, -10.0f, 10.0f, zNear, zFar);
        //depthProjectionMatrix = glm::ortho(-5.0f, 5.0f, -5.0f, 5.0f, zNear, zFar);
        break;
    case Spot:
        depthProjectionMatrix = glm::perspective(glm::radians(lightFOV), 1.0f, zNear, zFar);
        break;
    case Point:
        throw std::runtime_error("Error! Point lights are not yet supported");
        break;

    };

    //
    depthProjectionMatrix[1][1] *= -1;


    glm::mat4 depthmvp = depthProjectionMatrix * depthViewMatrix;

    lightView = depthmvp;
}
