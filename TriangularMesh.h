// ----------------------------------------------------------------------------------------------------
// 
// 	File name: TriangularMesh.h
//	Created By: Haard Panchal
//	Create Date: 03/14/2020
//
//	Description:
//		File has the implementation of the Triangular Mesh class.
//      Each mesh is defined by a set of triangles.
//      The Vertex data  and normal data is stored in an array
//	History:
//		03/13/19: H. Panchal Created the file
//
//  Declaration:
//      N/A
//
// ----------------------------------------------------------------------------------------------------

#ifndef TRIANGULARMESH
#define TRIANGULARMESH

#include "VisibleObject.h"

class TriangularMesh : public VisibleObject {
    private:
        Vector3 p_0; // Center of the mesh
        Vector3 * vertices; // Array of vertices, each triangle is assumed to be three consecutive vertices
        Vector3 * normals; // Array of normals
        int n_triangles; // Number of triangles

        Vector3 c_0; // Color of the Mesh
        
    public:
        __device__ __host__ TriangularMesh();
        __device__ __host__ TriangularMesh(Vector3& center, Vector3& color, Vector3 * vertex_data, Vector3 * normal_data, int no_of_triangles);
        __device__ __host__ ~TriangularMesh();
};

__device__ __host__ TriangularMesh::TriangularMesh() {}

__device__ __host__ TriangularMesh::~TriangularMesh() {}

__device__ __host__ TriangularMesh(Vector3& center, Vector3& color, Vector3 * vertex_data, Vector3 * normal_data, int no_of_triangles) 
    : p_0(color), c_0(color), vertices(vertex_data), normals(normal_data), n_triangles(no_of_triangles) {
        for(int i = 0; i < no_of_triangles; i++) {
            vertex_data[i] += p_0;
        }
}

#endif