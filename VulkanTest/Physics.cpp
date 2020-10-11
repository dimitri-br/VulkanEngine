#include "Physics.h"

// Initialize the physics world with a gravity value.
void Physics::init(btVector3 gravity) {
	dynamicsWorld->setGravity(gravity);
}

void Physics::cleanup() {
	delete dynamicsWorld;
	delete solver;
	delete broadphase;
	delete dispatcher;
	
}