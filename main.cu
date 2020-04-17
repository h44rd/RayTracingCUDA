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

// #define MATERIALDEBUG
// #define AREALIGHTDEBUG
// #define SHADOWDEBUG
// #define CUDADEBUG
// #define RENDERDEBUG
#define ACTUALRENDER
// #define INITDEBUG

#include <iostream>
#include <math.h>
#include <curand_kernel.h>

#define STB_IMAGE_IMPLEMENTATION
#include "libs/stb_image.h"

#include "Vector3.h"
#include "Ray.h"

#include "Camera.h"
#include "World.h"

#include "Sphere.h"
#include "Plane.h"
#include "PointLight.h"
#include "DirectionalLight.h"
#include "SpotLight.h"
#include "AreaLight.h"

#include "Material.h"
#include "TextureMaterial.h"

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
//  The function adds different objects to World.
//  All the object must be initialized onto the heap.
//
//	Parameters:
//      World ** world: A pointer to a pointer to a world object
//		int w: The width of the resulting image
//		int h: The height of the resulting image
//	
//	Return:
//		void
*/
__global__
void initializeWorld(World ** world, int w, int h, unsigned char ** array_of_images, int * img_w, int * img_h, int * img_chns, int n_imgs) {
    *world = new World();

    TextureMaterial * m1 = new TextureMaterial();
    m1->setColorImage(img_w[0], img_h[0], img_chns[0], array_of_images[0]);

    TextureMaterial * m2 = new TextureMaterial();
    m2->setColorImage(img_w[1], img_h[1], img_chns[1], array_of_images[1]);

    Vector3 color(0.3f, 0.8f, 0.3f);
    Vector3 center(-2.0, 0.0, 0.0);
    float r = 0.5f;
    Sphere * s = new Sphere(center, r, color);

    (*world)->addVisibleObject(s);

    Vector3 color5(1.0f, 0.0f, 0.1f);
    Vector3 center2(0.5, 0.0, 0.0);
    float r2 = 1.5f;
    Sphere * s2 = new Sphere(center2, r2, color5);
    s2->setMaterial(*m1);
    (*world)->addVisibleObject(s2);

    float beam_angle = 10.0;
    float falloff_angle = 30.0;
    beam_angle = beam_angle * PI / 180.0;
    falloff_angle = falloff_angle * PI / 180.0;
    Vector3 spotlightpos(-3.0, 3.0, 0.0f);
    Vector3 spotlightdir = - spotlightpos;
    SpotLight * spotlight = new SpotLight(spotlightpos, spotlightdir, beam_angle, falloff_angle);
    (*world)->addLight(spotlight);

    Vector3 spotlightpos2(-4.0f, 0.0, 0.0);
    Vector3 spotlightdir2 = - spotlightpos2;
    SpotLight * spotlight2 = new SpotLight(spotlightpos2, spotlightdir2, beam_angle, falloff_angle);
    // (*world)->addLight(spotlight2);

    Vector3 area_light_pos(-4.0, 2.0, 0);
    Vector3 area_light_dir = - area_light_pos;
    Vector3 area_light_up(0.0, 1.0, 0.0);
    AreaLight * areaLigth = new AreaLight(area_light_pos, area_light_dir, area_light_up, 0.1, 0.1);
    // (*world)->addLight(areaLigth);


    Vector3 color2(0.5f, 1.0f, 0.25f);
    Vector3 point(0.0, -2.5, 0.0);
    Vector3 normal(0, 1.0, 0.0);
    Plane * p = new Plane(normal, point, color2);
    p->setMaterial(*m2);
    (*world)->addVisibleObject(p);

    Vector3 color3(0.1f, 0.2f, 0.8f);
    Vector3 point2(2.5, 0.0, 0.0);
    Vector3 normal2(-1.0, 0.2, 0.2f);
    Plane * p2 = new Plane(normal2, point2, color3);
    p2->setMaterial(*m2);
    (*world)->addVisibleObject(p2);

    Vector3 positioncam(-3.0, 0.0, 4.0);
    Vector3 lookat(0.0f, 0.0f, 0.0f);
    Vector3 direction = lookat - positioncam;
    Vector3 updir(0.0, 1.0, 0.0);
    float aspect_ratio = (float(w))/(float(h));
    float distance_from_screen = 1.0;
    Camera * cam = new Camera(positioncam, direction, updir, aspect_ratio, 1.0, distance_from_screen);
    (*world)->setCamera(*cam);
}

/*  Function: addWorldToEngine
//
//	The function initializes the RenderEngine
//  An already initialized World object is passed to the RenderEngine
//
//	Parameters:
//      int w: Width of the rendered image
//      int h: Height of the rendered image		
//		RenderEngine ** r_engine: Pointer to a pointer to the RenderEngine object
//      World ** world: Pointer to a pointer 	
// 
//	Return:
//		void
*/
__global__
void addWorldToEngine(int w, int h, RenderEngine ** r_engine, World ** world, int samples) {
    *r_engine = new RenderEngine(w, h, **world);
    (* r_engine)->setAntiAliasing(samples);
}



/*  Function: Parallelize Render for each pixels
//
//	The kernel CUDA function implements the parallel threads for rendering each pixel.
//  The rendered pixels are stored in the frame_buffer array
//
//	Parameters:
//
//		
//		
//	
//	Return:
//		void
*/
__global__
void renderPixels(RenderEngine ** r_engine, Vector3 * frame_buffer, curandState * rand_sequence, int w, int h) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const int j = blockIdx.y * blockDim.y + threadIdx.y;
    const int index_ij = j * w + i;

    curand_init(1984 + index_ij, 0, 0, &rand_sequence[index_ij]);

    frame_buffer[index_ij] =  (*r_engine)->renderPixelSampling(i, j, rand_sequence[index_ij]);
    #ifdef CUDADEBUG
    printf("End of renderPixels\n");
    printf("framebuffer: i: %d r: %d c: %d\n", index_ij, i, j);
    #endif
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

    // Loading images for textures
    int n_imgs = 2;
    unsigned char * host_imgs[n_imgs];
    int img_w[n_imgs], img_h[n_imgs], img_chns[n_imgs];

    // -------- Load Images Here ------- //
    host_imgs[0] = stbi_load("textures/universe.jpg", &img_w[0], &img_h[0], &img_chns[0], 0);

    host_imgs[1] = stbi_load("textures/wall.jpg", &img_w[1], &img_h[1], &img_chns[1], 0);


    // Allocating devices memory to the images on the device
    unsigned char * temp_array[n_imgs];
    unsigned char ** array_of_images = 0; // Pointer to be allocated device memory
    int * img_w_d;
    int * img_h_d;
    int * img_chns_d;

    gpuErrchk(cudaMalloc(&img_w_d, n_imgs * sizeof(int)));
    gpuErrchk(cudaMemcpy(img_w_d, img_w, n_imgs * sizeof(int), cudaMemcpyHostToDevice));

    gpuErrchk(cudaMalloc(&img_h_d, n_imgs * sizeof(int)));
    gpuErrchk(cudaMemcpy(img_h_d, img_h, n_imgs * sizeof(int), cudaMemcpyHostToDevice));

    gpuErrchk(cudaMalloc(&img_chns_d, n_imgs * sizeof(int)));
    gpuErrchk(cudaMemcpy(img_chns_d, img_chns, n_imgs * sizeof(int), cudaMemcpyHostToDevice));

    if(array_of_images == 0) {
        gpuErrchk(cudaMalloc(&array_of_images,  sizeof(unsigned char*)  * n_imgs));
    }
    for(int i = 0; i < n_imgs; i++) {
        gpuErrchk(cudaMalloc(&temp_array[i], img_w[i] * img_h[i] * img_chns[i] * sizeof(unsigned char)));
        gpuErrchk(cudaMemcpy(&(array_of_images[i]), &(temp_array[i]), sizeof(unsigned char *), cudaMemcpyHostToDevice));//copy child pointer to device
        gpuErrchk(cudaMemcpy(temp_array[i], host_imgs[i], img_w[i] * img_h[i] * img_chns[i] * sizeof(unsigned char), cudaMemcpyHostToDevice)); // copy image to device
    }
     

    // Creating the required arrays for starting the rendering sequence
    int wid_cuda = 1200, hgt_cuda = 800;

    int samples = 32;

    Vector3 * frame_buffer_cuda;
    gpuErrchk(cudaMallocManaged(&frame_buffer_cuda, wid_cuda * hgt_cuda * sizeof(Vector3)));

    curandState * rand_sequence;
    gpuErrchk(cudaMallocManaged(&rand_sequence, wid_cuda * hgt_cuda * sizeof(curandState)));

    // Double Pointer: Done so that memory could be directly allocated to the object 
    // with the call of new constructor inside the global function.
    World ** world_cuda;
    gpuErrchk(cudaMallocManaged(&world_cuda, sizeof(World *)));

    RenderEngine ** r_engine_cuda;
    gpuErrchk(cudaMallocManaged(&r_engine_cuda, sizeof(RenderEngine *)));

    initializeWorld<<<1, 1>>>(world_cuda, wid_cuda, hgt_cuda, array_of_images, img_w_d, img_h_d, img_chns_d, n_imgs);
    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());

    addWorldToEngine<<<1, 1>>>(wid_cuda, hgt_cuda, r_engine_cuda, world_cuda, samples);
    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());

    const int block_size_side = 16;
    const dim3 block_size(block_size_side, block_size_side);
    const int grid_size_hgt = (hgt_cuda + block_size_side - 1)/block_size_side;
    const int grid_size_wid = (wid_cuda + block_size_side - 1)/block_size_side;
    const dim3 grid_size(grid_size_wid, grid_size_hgt);

    #ifdef CUDADEBUG
    std::cout<<"Grid Sizes: "<<grid_size_hgt<<" "<<grid_size_wid<<std::endl;
    std::cout<<"Block Sizes: "<<block_size_side<<" "<<block_size_side<<std::endl;
    #endif

    renderPixels<<<grid_size, block_size>>>(r_engine_cuda, frame_buffer_cuda, rand_sequence, wid_cuda, hgt_cuda);
    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());

    makeImage(frame_buffer_cuda, wid_cuda, hgt_cuda);

    return 0;
}