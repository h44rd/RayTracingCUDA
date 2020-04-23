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
#include "Plane.h"
#include "Helpers.h"

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

        // The function will return a Vector3 with x : Parameter t, y : slope of hit, z : if hit (+ve if hit, -ve otherwise)
        __host__ __device__ Vector3 getIntersectInfo(const Ray& incoming) const;

        // The normal to the sphere 
        __host__ __device__ Vector3 getNormalAtPoint(Vector3& point, int id_triangle) const;

        // The color
        __device__ Vector3 getColor(Vector3& point, int id_triangle) const;
        
        __host__ __device__ int getTypeID() const { return TMESH_TYPE_ID; }
};

__device__ __host__ TriangularMesh::TriangularMesh() {}

__device__ __host__ TriangularMesh::~TriangularMesh() {}

__device__ __host__ TriangularMesh::TriangularMesh(Vector3& center, Vector3& color, Vector3 * vertex_data, Vector3 * normal_data, int no_of_triangles) 
    : p_0(center), c_0(color), vertices(vertex_data), normals(normal_data), n_triangles(no_of_triangles) {
    for(int i = 0; i < no_of_triangles; i++) {
        vertices[i] += p_0;
    }
    #ifdef MESHDEBUG
    for(int i = 0; i < no_of_triangles * 3; i++) {
        printf("Inside Constructor i: %d V: %f %f %f\n", i, vertices[i].x(), vertices[i].y(), vertices[i].z());
    }
    for(int i = 0; i < no_of_triangles * 3; i++) {
        printf("i: %d N: %f %f %f\n", i, normals[i].x(), normals[i].y(), normals[i].z());
    }
    #endif
}

__host__ __device__ Vector3 TriangularMesh::getIntersectInfo(const Ray& incoming) const {
    Vector3 intersection(0.0f, 0.0f, 0.0f);

    float id_triangle = -1;
    float if_intersect = -1;
    float ray_t = 10000;

    Vector3 A(0.0f, 0.0f, 0.0f);
    Vector3 A_0(0.0f, 0.0f, 0.0f);
    Vector3 A_1(0.0f, 0.0f, 0.0f);
    Vector3 A_2(0.0f, 0.0f, 0.0f);
    Vector3 n(0.0f, 0.0f, 0.0f);
    Vector3 plane_intersection(0.0f, 0.0f, 0.0f);
    Vector3 p_h(0.0f, 0.0f, 0.0f);

    int p_i, p_ip1, p_im1;
    float s, t, mst;
    for(int i = 0; i < n_triangles; i++) {
        p_i = 3 * i + 1;
        p_im1 = 3 * i;
        p_ip1 = 3 * i + 2;

        A = cross(vertices[p_i] - vertices[p_im1], vertices[p_ip1] - vertices[p_i]) / 2.0f;
        n = unit_vector(A);

        Plane p(n, vertices[p_i]);
        plane_intersection = p.getIntersectInfo(incoming);

        if(ifRayIntersected(plane_intersection)) {
            float temp_t = getTFromIntersectInfo(plane_intersection); 
            if(temp_t < ray_t && dot(incoming.getDirection(), n) < 0) {
                p_h = incoming.getPoint(temp_t);
                A_0 = cross(p_h - vertices[p_ip1], vertices[p_i] - p_h) / 2.0f;
                A_1 = cross(p_h - vertices[p_im1], vertices[p_ip1] - p_h) / 2.0f;
                A_2 = cross(p_h - vertices[p_i], vertices[p_im1] - p_h) / 2.0f;

                s = dot(n, A_1) / A.length();
                t = dot(n, A_2) / A.length();
                mst = dot(n, A_0) / A.length(); // 1 - s - t

                if(s > 0.0f && s < 1.0f  && t > 0.0f && t < 1.0f  && mst > 0.0f && mst < 1.0f) {
                    #ifdef MESHDEBUG
                        // printf("Point of intersection: %f %f %f A: %f %f %f\n", p_h.x(), p_h.y(), p_h.z(), n.x(), n.y(), n.z());
                    #endif
                    ray_t = temp_t;
                    id_triangle = i;
                    if_intersect = 1.0f;  
                }
            }
        }
    }

    intersection[0] = t;
    intersection[1] = id_triangle;
    intersection[2] = if_intersect;

    return intersection;
}

__host__ __device__ Vector3 TriangularMesh::getNormalAtPoint(Vector3& point, int id_triangle) const {
    int p_i = 3 * id_triangle + 1;
    int p_im1 = 3 * id_triangle;
    int p_ip1 = 3 * id_triangle + 2;

    Vector3 A = cross(vertices[p_i] - vertices[p_im1], vertices[p_ip1] - vertices[p_i]) / 2;
    Vector3 A_0 = cross(point - vertices[p_ip1], vertices[p_i] - point) / 2;
    Vector3 A_1 = cross(point - vertices[p_im1], vertices[p_ip1] - point) / 2;
    Vector3 A_2 = cross(point - vertices[p_i], vertices[p_im1] - point) / 2;
    Vector3 n = unit_vector(A);

    float s = dot(n, A_1) / A.length();
    float t = dot(n, A_2) / A.length();
    float mst = dot(n, A_0) / A.length();

    Vector3 n_h = mst * normals[p_im1] + s * normals[p_i] + t * normals[p_ip1];
    n_h.make_unit_vector();

    #ifdef MESHDEBUG
        // printf("Normal: %f %f %f\n", n_h.x(), n_h.y(), n_h.z());
    #endif
    return n_h;
}

__device__ Vector3 TriangularMesh::getColor(Vector3& point, int id_triangle) const {
    if(m !=  NULL) {
        int p_i = 3 * id_triangle + 1;
        int p_im1 = 3 * id_triangle;
        int p_ip1 = 3 * id_triangle + 2;

        Vector3 A = cross(vertices[p_i] - vertices[p_im1], vertices[p_ip1] - vertices[p_i]) / 2;
        Vector3 A_1 = cross(point - vertices[p_im1], vertices[p_ip1] - point) / 2;
        Vector3 A_2 = cross(point - vertices[p_i], vertices[p_im1] - point) / 2;
        Vector3 n = unit_vector(A);

        float s = dot(n, A_1) / A.length();
        float t = dot(n, A_2) / A.length();

        float u = (1 - s - t) * 0.1 + s * 0.9 + t * 0.9;
        float v = (1 - s - t) * 0.1 + s * 0.1 + t * 0.9;
        return m->getBilinearColor(u, v);
    }
    return c_0;
}


#endif