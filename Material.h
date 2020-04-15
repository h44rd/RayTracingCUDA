// ----------------------------------------------------------------------------------------------------
// 
// 	File name: Material.h
//	Created By: Haard Panchal
//	Create Date: 04/10/2020
//
//	Description:
//		File has the definition and implementation of the Material class
//      This class gives the basic structure for Materials for VisibleObjects
//      More complicated Material classes, such as TextureMaterial can be built on top of this class.
//      
//	History:
//		04/10/2020: H. Panchal Created the file
//
//  Declaration:
//
// ----------------------------------------------------------------------------------------------------

#ifndef MATERIALH
#define MATERIALH

class Material {
    private:
        float m_k_d, m_k_s; // Diffuse and specular coefficients
        float m_refractive_index; // Refractive index of the material
        Vector3 m_color; // color of the material
    public:
        __device__ Material();
        __device__ ~Material();
        __device__ Material(Vector3& color, float k_diffuse, float k_specular, float refractive_index);
        __device__ virtual Vector3 getColor() { return m_color; }
        __device__ virtual Vector3 getBilinearColor(float u, float v) { return m_color; } // returns bilinear color 

        __device__ inline float getKDiffuse() { return m_k_d; }
        __device__ inline float getKSpecular() { return m_k_s; }
        __device__ inline float getRefractiveIndex() { return m_refractive_index; }
};

__device__ Material::Material() {
    m_color = Vector3(1.0f, 1.0f, 1.0f);
    m_k_d = 1.0;
    m_k_s = 1.0;
    m_refractive_index = 1.0;
}

__device__ Material::~Material() {}

__device__ Material::Material(Vector3& color, float k_diffuse, float k_specular, float refractive_index)
    : m_color(color), m_k_d(k_diffuse), m_k_s(k_specular), m_refractive_index(refractive_index) {}

#endif