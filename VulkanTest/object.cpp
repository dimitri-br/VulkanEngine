#define GLFW_INCLUDE_VULKAN 
#define GLM_ENABLE_EXPERIMENTAL
#include <GLFW/glfw3.h> // Windowing
#include <iostream> // IDK
#include <stdexcept> // Errors
#include <cstdlib> // STD lib
#include <vector> // For arrays
#include <glm/glm.hpp> // Needed for math functions like matrices etc
#include <glm/gtc/matrix_transform.hpp> // transformations
#include <glm/gtx/rotate_vector.hpp>
#include <glm/gtc/quaternion.hpp>
#include <glm/gtx/quaternion.hpp>
#include "object.h" // holds local files
#include "transform.h"
#include "material.h"
#include "Vertex.h"
#include "btBulletDynamicsCommon.h"

void Object::init(std::string model, std::string texture, std::string normal, Material material, Transform trans)
{
	model_path = model;
	texture_path = texture;
	mat = material;
	transform = trans;
	normal_path = normal;

}

void Object::setupPhysics(float mass, btCollisionShape *collider) {
	// Setup rigidbody transform
	btTransform transform;
	transform.setIdentity(); // This is the transform identity
	transform.setOrigin(btVector3(position.x, position.y, position.z)); // origin is the transform for the object


	glm::quat quaternion = glm::quat(rotation); // Euler -> Quaternion. Much more accurate and needed for rigidbodies

	transform.setRotation(btQuaternion(quaternion.x, quaternion.y, quaternion.z, quaternion.w));
	
	// create a motion state for our rigidbody
	btDefaultMotionState* motionstate = motionstate = new btDefaultMotionState(transform);

	// create a new box collider with the scale of the object. Small issue - this is only box
	// TODO: support all types of colliders
	collisionShape = collider;


	btScalar objMass(mass);

	// Check that the mass is greater than 0
	bool isDynamic = (mass != 0.f);
	btVector3 localInertia(0, 0, 0);
	if (isDynamic)
		collisionShape->calculateLocalInertia(objMass, localInertia);
	


	btRigidBody::btRigidBodyConstructionInfo rigidBodyCI(
		mass,                  // mass, in kg. 0 -> Static object, will never move.
		motionstate,
		collisionShape,  // collision shape of body
		localInertia   // local inertia
	);
	rigidBody = new btRigidBody(rigidBodyCI); // create the rigidbody
}

Transform Object::getTransform()
{
	// initialize new transform
	Transform trans{};

	// get the rigidbody transform 
	btTransform rigidbodyTransform = rigidBody->getWorldTransform();

	//TODO - add rotations
	float x = rigidbodyTransform.getRotation().getX();
	float y = rigidbodyTransform.getRotation().getY();
	float z = rigidbodyTransform.getRotation().getZ();
	float w = rigidbodyTransform.getRotation().getW();
	glm::quat quaternion(w, x, y, z);

	glm::vec3 newPos = glm::vec3(rigidbodyTransform.getOrigin().getX(), rigidbodyTransform.getOrigin().getY(), rigidbodyTransform.getOrigin().getZ());
	// Set new transform to rigidbodies transform
	trans.transform = glm::translate(glm::mat4(1.0f), newPos); // translate the model. vec3 specifies translation. Seems to be (z, x, y)
	trans.transform *= glm::toMat4(quaternion); // rotate by the rigidbody quaternion
	trans.transform *= glm::scale(glm::mat4(1.0f), scale); // set scale


	return trans;
}
