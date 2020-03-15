// ----------------------------------------------------------------------------------------------------
// 
// 	File name: RenderEngine.h
//	Created By: Haard Panchal
//	Create Date: 03/14/2020
//
//	Description:
//		File has the definition and implementation of the RenderEngine class
//      
//      The engine gets the rays from the camera and checks for intersections with all the visible objects in 
//
//	History:
//		03/14/19: H. Panchal Created the file
//
//  Declaration:
//      N/A
//
// ----------------------------------------------------------------------------------------------------

#ifndef RENDERENGINEH
#define RENDERENGINEH

#include "Vector3.h"
#include "World.h"
#include "Camera.h"
#include "Ray.h"
#include "VisibleObject.h"

class RenderEngine
{
    private:
        int w, h;

    public:
        __host__ __device__ RenderEngine();
        __host__ __device__ RenderEngine(int width, int height, World& world_p);
        __host__ __device__ ~RenderEngine();

        __host__ __device__ void render(int i, int j); //Renders the pixel i,j

        World* world;
        Camera* camera;
};

__host__ __device__ RenderEngine::RenderEngine() {}

__host__ __device__ RenderEngine::RenderEngine(int width, int height, World& world_p) : w(width), h(height) {
    world = &world_p;
    camera = world.getCamera();
}

__host__ __device__ RenderEngine::~RenderEngine() {}

__host__ __device__ Vector3 RenderEngine::render(int i, int j) {
    float u = ((float) i)/((float) w);
    float v = ((float) h - j)/((float) h); // Pixel cordinates have the origin on top left but our screen origin is on the bottom left

    Ray eye_ray = camera->getRay(u, v);

    int total_objects = world->getTotalVisibleObjects();

    float min_t = 0.0f;
    bool if_t_encountered = false;
    Vector3 intersectInfo;

    VisibleObject * closest_object;

    for(int i = 0; i < total_objects; i++) {
        
        intersectInfo = (world->getVisibleObject(i))->getIntersectInfo(eye_ray);
        
        if( !if_t_encountered && intersectInfo[2] > 0.0f) {
            if_t_encountered = true;
            min_t = intersectInfo[0];
            closest_object = world->getVisibleObject(i);
        } else if(if_t_encountered && intersectInfo[2] > 0.0f) {
            if(min_t > intersectInfo[0]) {
                min_t = intersectInfo[0];
                closest_object = world->getVisibleObject(i);
            }
        }

    }

    /* TODO */
    //Calculate color using the closest_object

    if( !if_t_encountered ) {
        return Vector3(0.0, 0.0, 0.0); // Default color
    }
}

#endif