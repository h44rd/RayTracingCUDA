// ----------------------------------------------------------------------------------------------------
// 
// 	File name: Material.h
//	Created By: Haard Panchal
//	Create Date: 04/10/2020
//
//	Description:
//		File has the definition and implementation of the abstract Light class
//      This is the base class for all the types of lights.
//      
//	History:
//		03/16/19: H. Panchal Created the file
//
//  Declaration:
//
// ----------------------------------------------------------------------------------------------------

#ifndef MATERIALH
#define MATERIALH

class Material {
    private:
        
    public:
        __device__ Material();
        __device__ ~Material();
};

__device__ Material::Material()) {}

__device__ Material::~Material() {}

#endif