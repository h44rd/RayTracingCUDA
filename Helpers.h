#ifndef HELPERSH
#define HELPERSH

#define TINYOBJLOADER_IMPLEMENTATION // define this in only *one* .cc
#include "tiny_obj_loader.h"

#define PI 3.1415927
#define EPSILON 0.0000001 // A very small number
#define LARGENUMBER 1000

__host__ __device__ float clamp(float x, float high, float low);
__host__ __device__ float smoothstep(float edge0, float edge1, float x);
__host__ void makeImage(Vector3 * frame_buffer, int w, int h);
void loadOBJ(const char * file_name, float * vertex_data, float * normal_data);


__host__ __device__ float clamp(float x, float low, float high) {
    return (x < low) ? low : (high < x) ? high : x;
}

__host__ __device__ float smoothstep(float edge0, float edge1, float x) {
    // Scale, bias and saturate x to 0..1 range
    x = clamp((x - edge0) / (edge1 - edge0), 0.0f, 1.0f);
    // Evaluate polynomial
    return x * x * (3 - 2 * x);
}

__host__ void makeImage(Vector3 * frame_buffer, int w, int h) {
    // Output Pixel as Image
    #ifdef ACTUALRENDER
    std::cout << "P3\n" << w << " " << h << "\n255\n";
    #endif

    int index_ij;
    for (int j = 0; j < h; j++) {
        for (int i = 0; i < w; i++) {
            index_ij = j * w + i;

            #ifdef CUDADEBUG
            std::cout<<"makeImage: index: "<<index_ij<<" i: "<<i<<" j: "<<j<<std::endl;
            #endif
            
            int ir = int(255.99*frame_buffer[index_ij].r());
            int ig = int(255.99*frame_buffer[index_ij].g());
            int ib = int(255.99*frame_buffer[index_ij].b());

            if(ir > 256 || ir < 0) {
                ir = 0;
            }

            if(ig > 256 || ig < 0) {
                ig = 0;
            }

            if(ib > 256 || ib < 0) {
                ib = 0;
            }
            #ifdef ACTUALRENDER
            std::cout << ir << " " << ig << " " << ib << "\n";
            #endif
        }
    }
}

// vertex_data and normal_data will be appropriately initialized;
void loadOBJ(const char * file_name, Vector3 ** vertex_data, Vector3 ** normal_data) {

}

#endif