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
#include "Light.h"
#include "Managed.h"

class World : public Managed
{
    private:
        VisibleObject** visible_objects;
        int total_objects;
        int current_index;

        Camera * camera;
        
        Light ** lights;
        int total_lights;
        int light_id;

    public:
        __host__ __device__ World();
        __host__ __device__ ~World();

        // Adds the new object to the list of objects
        __host__ __device__ void addVisibleObject(VisibleObject * new_visible_object);

        __host__ __device__ inline void resetIteration() { current_index = 0; }

        __host__ __device__ inline int getTotalVisibleObjects() { return total_objects; }

        __host__ __device__ VisibleObject* getVisibleObject(int i) const;

        // Camera
        __host__ __device__ inline void setCamera(Camera& camera_p) { camera = &camera_p; }
        __host__ __device__ inline Camera* getCamera() { return camera; }

        //Light
        __host__ __device__ inline void addLight(Light* light_p);
        __host__ __device__ inline int getTotalLights() { return total_lights; }
        __host__ __device__ Light * getLight(int i) const;
        __host__ __device__ inline void setLightId(int i) { if(i < total_lights) light_id = i; }
        __host__ __device__ inline int getSelectedLightId() { return light_id; }

};

__host__ __device__ World::World() {
    visible_objects = new VisibleObject * [100];
    lights = new Light * [100];
    total_objects = 0;
    total_lights = 0;
    current_index = 0;
    light_id = 0;
}

__host__ __device__ World::~World() {}

__host__ __device__ void World::addVisibleObject(VisibleObject* new_visible_object) {
    visible_objects[total_objects] = new_visible_object; // Adding the object to the array
    total_objects++;
}

__host__ __device__ VisibleObject* World::getVisibleObject(int i) const {
    if(i < total_objects) {
        return visible_objects[i];
    } 
    return NULL;
}

__host__ __device__ void World::addLight(Light * new_light) {
    lights[total_lights] = new_light;
    total_lights++;
}

__host__ __device__ Light * World::getLight(int i) const {
    if(i < total_lights) {
        return lights[i];
    }
    return NULL;
}

#endif