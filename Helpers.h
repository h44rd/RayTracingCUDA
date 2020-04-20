#ifndef HELPERSH
#define HELPERSH

#define TINYOBJLOADER_IMPLEMENTATION // define this in only *one* .cc
#include "libs/tiny_obj_loader.h"

#define PI 3.1415927
#define EPSILON 0.0000001 // A very small number
#define LARGENUMBER 1000

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

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
void loadOBJ(std::string  file_name, Vector3 ** vertex_data, Vector3 ** normal_data) {
    tinyobj::attrib_t attrib;
    std::vector<tinyobj::shape_t> shapes;
    std::vector<tinyobj::material_t> materials;
    std::string warn;
    std::string err;

    bool ret = tinyobj::LoadObj(&attrib, &shapes, &materials, &warn, &err, file_name.c_str()); // Will return triangles by default

    if (!warn.empty()) {
        std::cout << warn << std::endl;
    }
    if (!err.empty()) {
        std::cerr << err << std::endl;
    }
    if (!ret) {
        std::cerr << "It failed!" << std::endl;
    }

    #ifdef MESHDEBUG
        std::cout<<"Loading objs works"<<std::endl;
    #endif

    int total_traingles_verts = 0;

    for (size_t s = 0; s < shapes.size(); s++) {
        // Loop over faces(polygon)
        size_t index_offset = 0;
        for (size_t f = 0; f < shapes[s].mesh.num_face_vertices.size(); f++) {
            int fv = shapes[s].mesh.num_face_vertices[f];
            total_traingles_verts += fv;
        }
    }

    // Create memory on CUDA
    Vector3 * vertex_data_array, * normal_data_array;
    gpuErrchk(cudaMallocManaged(&vertex_data_array, total_traingles_verts * sizeof(Vector3)));
    gpuErrchk(cudaMallocManaged(&normal_data_array, total_traingles_verts * sizeof(Vector3)));



    #ifdef MESHDEBUG
        std::cout<<"Created  memory: "<<total_traingles_verts<<std::endl;
    #endif

    // Loop over shapes
    for (size_t s = 0, i = 0; s < shapes.size(); s++) {
        // Loop over faces(polygon)
        size_t index_offset = 0;
        for (size_t f = 0; f < shapes[s].mesh.num_face_vertices.size(); f++) {
            int fv = shapes[s].mesh.num_face_vertices[f];
            // Loop over vertices in the face.
            for (size_t v = 0; v < fv; v++, i++) {
                // access to vertex

                #ifdef MESHDEBUG
                    std::cout<<"i: "<<i<<std::endl;
                #endif

                tinyobj::index_t idx = shapes[s].mesh.indices[index_offset + v];
                vertex_data_array[i].members[0] = (float) attrib.vertices[3*idx.vertex_index+0];
                vertex_data_array[i].members[1] = (float) attrib.vertices[3*idx.vertex_index+1];
                vertex_data_array[i].members[2] = (float) attrib.vertices[3*idx.vertex_index+2];

                #ifdef MESHDEBUG
                    std::cout<<"end i: "<<i<<std::endl;
                #endif
                
                // float x = (float) attrib.normals[3*idx.normal_index+0];
                // float y = (float) attrib.normals[3*idx.normal_index+1];
                // float z = (float) attrib.normals[3*idx.normal_index+2];

                normal_data_array[i].members[0] = (float) attrib.normals[3*idx.normal_index+0];
                normal_data_array[i].members[1] = (float) attrib.normals[3*idx.normal_index+1];
                normal_data_array[i].members[2] = (float) attrib.normals[3*idx.normal_index+2];

                #ifdef MESHDEBUG
                    std::cout<<"end i: "<<i<<std::endl;
                #endif
            }
            index_offset += fv;
            // per-face material
            shapes[s].mesh.material_ids[f];
        }
    }

    *vertex_data = vertex_data_array;
    *normal_data = normal_data_array;
}

#endif