// ----------------------------------------------------------------------------------------------------
// 
// 	File name: VisibleObject.h
//	Created By: Haard Panchal
//	Create Date: 03/14/2020
//
//	Description:
//		File has the definition and implementation of the VisibleObject class
//      
//      Pure Abstract Class for Visible Objects (Any Renderable object)
//
//      IMP: Function getIntersectInfo has to be overridden by derived classes
//
//	History:
//		03/13/19: H. Panchal Created the file
//
//  Declaration:
//      N/A
//
// ----------------------------------------------------------------------------------------------------

#ifndef VISIBLEOBJECTH
#define VISIBLEOBJECTH

#include "Vector3.h"
#include "Ray.h"
#include "Material.h"
#include "Helpers.h"

class VisibleObject
{
    private:
        
    public:
        __host__ __device__ VisibleObject();
        __host__ __device__ ~VisibleObject();

        __device__ inline virtual void setMaterial(Material& material) { m = &material; }

        // The function will return a Vector3 with x : Parameter t, y : slope of hit, z : yet to be decided (-1)
        // Thought: We could also return the normal vector
        __host__ __device__ virtual Vector3 getIntersectInfo(const Ray& incoming) const = 0;

        __host__ __device__ virtual Vector3 getNormalAtPoint(Vector3& point) const;  

        __device__ virtual Vector3 getColor(Vector3& point) const;

        __host__ __device__ virtual int getTypeID() const { return PLANE_TYPE_ID; }

        // For traingular meshes
        __host__ __device__ virtual Vector3 getNormalAtPoint(Vector3& point, int id_triangle) const { return Vector3(0.0f, 0.0f, 0.0f); };
        __device__ virtual Vector3 getColor(Vector3& point, int id_triangle) const { return Vector3(0.0f, 0.0f, 0.0f); };

    protected:
        Material * m = NULL; // Material of the object
};

__host__ __device__ VisibleObject::VisibleObject() {}

__host__ __device__ VisibleObject::~VisibleObject() {}

__host__ __device__ Vector3 VisibleObject::getNormalAtPoint(Vector3& point) const { return Vector3(0.0f, 0.0f, 0.0f); }

__device__ Vector3 VisibleObject::getColor(Vector3& point) const { return Vector3(0.0f, 0.0f, 0.0f); }

#endif