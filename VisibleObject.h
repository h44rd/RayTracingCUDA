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

class VisibleObject
{
    private:
        
    public:
        __host__ VisibleObject();
        __host__ ~VisibleObject();

        // The function will return a Vector3 with x : Parameter t, y : slope of hit, z : yet to be decided (-1)
        // Thought: We could also return the normal vector
        __host__ virtual Vector3 getIntersectInfo(const Ray& incoming) const = 0;

        __host__ virtual Vector3 getNormalAtPoint(Vector3& point) const;  

        __host__ virtual Vector3 getColor(Vector3& point) const;

};

__host__ VisibleObject::VisibleObject() {}

__host__ VisibleObject::~VisibleObject() {}

__host__ Vector3 VisibleObject::getNormalAtPoint(Vector3& point) const { return Vector3(0.0f, 0.0f, 0.0f); }

__host__ Vector3 VisibleObject::getColor(Vector3& point) const { return Vector3(0.0f, 0.0f, 0.0f); }

#endif