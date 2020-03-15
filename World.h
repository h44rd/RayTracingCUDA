// ----------------------------------------------------------------------------------------------------
// 
// 	File name: World.h
//	Created By: Haard Panchal
//	Create Date: 03/14/2020
//
//	Description:
//		File has the definition and implementation of the World class
//
//      The class defines the World, based on which the RenderEngine renders the final result
// 
//	History:
//		03/13/19: H. Panchal Created the file
//
//  Declaration:
//      N/A
//
// ----------------------------------------------------------------------------------------------------

#ifndef WORLDH
#define WORLDH

#include "VisibleObject.h"
#include "Camera.h"

class World
{
    private:
        VisibleObject** visible_objects;

        int total_objects;
        int current_index;

        Camera * camera;

    public:
        __host__ __device__ World();
        __host__ __device__ ~World();

        // Adds the new object to the list of objects
        __host__ __device__ void addVisibleObject(VisibleObject * new_visible_object);

        __host__ __device__ inline void resetIteration() { current_index = 0; }

        __host__ __device__ inline int getTotalVisibleObjects() { return total_objects; }

        __host__ __device__ VisibleObject* getItem(int i) const;

        // Camera
        __host__ __device__ inline void setCamera(Camera& camera_p) { camera = &camera_p; }
        __host__ __device__ inline Camera* getCamera() { return camera; }

        __host__ __device__ inline void setLight(Vector3& light_p) { light = light_p; }
        Vector3 light; // Coming soon: LIIIGGGGGTTTTHHHH CLLLLAAASSSSS
};

__host__ __device__ World::World() {
    visible_objects = new VisibleObject * [100];
    total_objects = 0;
    current_index = 0;
}

__host__ __device__ World::~World() {}

__host__ __device__ void World::addVisibleObject(VisibleObject* new_visible_object) {
    visible_objects[total_objects] = new_visible_object; // Adding the object to the array
    total_objects++;
}

__host__ __device__ VisibleObject* World::getNextItem(int i) const {
    if(i < total_objects) {
        return visible_objects[i];
    } 
    return NULL;
}

#endif