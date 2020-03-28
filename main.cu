// ----------------------------------------------------------------------------------------------------
// 
// 	File name: main.cu
//	Created By: Haard Panchal
//	Create Date: 03/11/2020
//
//	Description:
//		Main file for the Ray Tracing project. The file implements the parallel CUDA algorithm.
//      You can also use it to create the world that would be used to render the final result.
//
//	History:
//		03/10/19: H. Panchal Created the file
//
//  Declaration:
//      N/A
//
// ----------------------------------------------------------------------------------------------------

// #define RENDERDEBUG
// #define ACTUALRENDER
// #define INITDEBUG

#include <iostream>
#include <math.h>

#include "Vector3.h"
#include "Ray.h"

#include "Camera.h"
#include "World.h"

#include "Sphere.h"
#include "Plane.h"
#include "PointLight.h"
#include "DirectionalLight.h"
#include "SpotLight.h"

#include "RenderEngine.h"

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}


/*  Function: initializeEngine
//
//  The function adds different object to World
//
//	Parameters:
//  
//		
//		
//	
//	Return:
//		int: 0 if successful
*/
__global__
void initializeWorld(World * world) {
    world = new World();

    Vector3 * color = new Vector3(1.0f, 0.5f, 1.0f);
    Vector3 * center = new Vector3(-1.0, 0.0, 0.0);
    float r = 1.0f;
    Sphere * s = new Sphere(*center, r, *color);

    world->addVisibleObject(s);

    float beam_angle = 10.0;
    float falloff_angle = 30.0;
    beam_angle = beam_angle * PI / 180.0;
    falloff_angle = falloff_angle * PI / 180.0;
    Vector3 * spotlightpos = new Vector3(-0.3, 0.25, 3.0f);
    Vector3 * spotlightdir = new Vector3();
    * spotlightdir = -1 * (*spotlightpos);
    SpotLight * spotlight = new SpotLight(* spotlightpos, * spotlightdir, beam_angle, falloff_angle);

    world->addLight(spotlight);
}

/*  Function: addWorldToEngine
//
//	The function initializes the RenderEngine
//  The function adds different object to World and passes World to the RenderEngine
//
//	Parameters:
//  
//		
//		
//	
//	Return:
//		int: 0 if successful
*/
__global__
void addWorldToEngine(int w, int h, RenderEngine * r_engine, World * world) {
    r_engine = new RenderEngine(w, h, *world);
}



/*  Function: Parallelize Render for each pixels
//
//	The function parallelizes the render on the GPU
//
//	Parameters:
//
//		
//		
//	
//	Return:
//		int: 0 if successful
*/
__global__
void renderPixels(RenderEngine * r_engine, Vector3 * frame_buffer, int w, int h) {
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    const int r = blockIdx.y * blockDim.y + threadIdx.y;
    const int i = r * w + c;

    frame_buffer[i] =  r_engine->renderPixel(r, c);
    printf("End of renderPixels\n");
    printf("framebuffer: %d %d %d\n", frame_buffer[i].r(), frame_buffer[i].g(), frame_buffer[i].b());
}


/*  Function: main
//
//	Parses the argument list. Initializes the relevant objects and starts rendering.
//
//	Parameters:
//
//		int argc: Number of arguments
//		char *argv[]: List of the arguments
//	
//	Return:
//		int: 0 if successful
*/
int main(int argc, char *argv[]) {

    // JUST FOR REFERENCE
    // Vector3 positioncam(0.0, 0.0, 5.0);
    // Vector3 lookat(0.0f, 0.0f, 0.0f);
    // Vector3 direction = lookat - positioncam;
    // Vector3 updir(0.0, 1.0, 0.0);
    // Camera cam(positioncam, direction, updir, 1.0, 1.0, 1.0);


    int wid_cuda = 16, hgt_cuda = 16;

    Vector3 * frame_buffer_cuda;
    gpuErrchk(cudaMallocManaged(&frame_buffer_cuda, wid_cuda * hgt_cuda * sizeof(Vector3)));

    World * world_cuda;
    gpuErrchk(cudaMallocManaged(&world_cuda, sizeof(World)));

    RenderEngine * r_engine_cuda;
    gpuErrchk(cudaMallocManaged(&r_engine_cuda, sizeof(RenderEngine)));

    initializeWorld<<<1, 1>>>(world_cuda);
    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());

    addWorldToEngine<<<1, 1>>>(wid_cuda, hgt_cuda, r_engine_cuda, world_cuda);
    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());

    const int block_size_side = 16;
    const dim3 block_size(block_size_side, block_size_side);
    const int grid_size_hgt = (hgt_cuda + block_size_side - 1)/block_size_side;
    const int grid_size_wid = (wid_cuda + block_size_side - 1)/block_size_side;
    const dim3 grid_size(grid_size_hgt, grid_size_wid);
    std::cout<<"Grid Sizes: "<<grid_size_hgt<<" "<<grid_size_wid<<std::endl;
    std::cout<<"Block Sizes: "<<block_size_side<<" "<<block_size_side<<std::endl;

    renderPixels<<<grid_size, block_size>>>(r_engine_cuda, frame_buffer_cuda, wid_cuda, hgt_cuda);
    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());

    makeImage(frame_buffer_cuda, wid_cuda, hgt_cuda);

    return 0;
}