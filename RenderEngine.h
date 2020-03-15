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

class RenderEngine
{
    private:
        int w, h;

    public:
        __host__ __device__ RenderEngine();
        __host__ __device__ ~RenderEngine();

        __host__ __device__ void render(Vector3& i);

        World * world;
};

__host__ __device__ RenderEngine::RenderEngine(int width, int height) : w(width), h(height) {}

__host__ __device__ RenderEngine::RenderEngine() {}

__host__ __device__ RenderEngine::~RenderEngine() {}



#endif