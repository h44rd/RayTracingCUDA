// ----------------------------------------------------------------------------------------------------
// 
// 	File name: RenderEngine.h
//	Created By: Haard Panchal
//	Create Date: 03/14/2020
//
//	Description:
//		File has the definition and implementation of the RenderEngine class
//      
//      The engine gets the rays from the camera and checks for intersections with all the visible objects in 
//
//	History:
//		03/14/19: H. Panchal Created the file
//
//  Declaration:
//      N/A
//
// ----------------------------------------------------------------------------------------------------

#ifndef RENDERENGINEH
#define RENDERENGINEH

#include "Vector3.h"
#include "World.h"
#include "Camera.h"
#include "Ray.h"
#include "VisibleObject.h"
#include "Helpers.h"

class RenderEngine
{
    private:
        int w, h;

        float sharp_edge0, sharp_edge1;

        float ambient_intensity;
    public:
        __host__ RenderEngine();
        __host__ RenderEngine(int width, int height, World& world_p);
        __host__ ~RenderEngine();

        __host__ Vector3 render(float u, float v); //Renders the point u,v on the screen (0 <= u,v <= 1)
        __host__ Vector3 computeColor(VisibleObject* closest_object, Ray& eye_ray, Vector3& t); // Compute color given the closest object

        __host__ Vector3 renderPixel(int i, int j); // Renders the pixel i,j

        __host__ void renderAllPixels();

        __host__ void inline setSharpEdge(float edge0, float edge1) {sharp_edge0 = edge0; sharp_edge1 = edge1;}
        World* world;
        Camera* camera;
};

__host__ RenderEngine::RenderEngine() {}

__host__ RenderEngine::RenderEngine(int width, int height, World& world_p) : w(width), h(height) {
    world = &world_p;
    camera = world->getCamera();
    sharp_edge0 = 0.0;
    sharp_edge1 = 1.0;
    ambient_intensity = 0.3;
}

__host__ RenderEngine::~RenderEngine() {}

__host__ Vector3 RenderEngine::render(float u, float v) {
    Ray eye_ray = camera->getRay(u, v);

    int total_objects = world->getTotalVisibleObjects();

    float min_t = 0.0f;
    bool if_t_encountered = false;
    Vector3 intersectInfo;

    VisibleObject * closest_object;

    for(int i = 0; i < total_objects; i++) {
        
        intersectInfo = (world->getVisibleObject(i))->getIntersectInfo(eye_ray);
        
        #ifdef DEBUG
        std::cout<<"Intersect info: "<<intersectInfo<<std::endl;
        #endif

        if( !if_t_encountered && intersectInfo[2] > 0.0f) {
            if_t_encountered = true;
            min_t = intersectInfo[0];
            closest_object = world->getVisibleObject(i);
        } else if(if_t_encountered && intersectInfo[2] > 0.0f) {
            if(min_t > intersectInfo[0]) {
                min_t = intersectInfo[0];
                closest_object = world->getVisibleObject(i);
            }
        }

    }

    if( !if_t_encountered ) {
        return Vector3(0.0, 0.0, 0.0); // Default color
    }   
    
    Vector3 point_of_intersection = eye_ray.getPoint(min_t);

    #ifdef RENDERDEBUG
    std::cout<<"Eye ray starting point: "<<eye_ray.getStartingPoint()<<std::endl;
    std::cout<<"Eye ray direction: "<<eye_ray.getDirection()<<std::endl;    
    std::cout<<"t: "<<min_t<<std::endl;    
    std::cout<<"Point of intersection: "<<point_of_intersection<<std::endl;
    #endif

    return computeColor(closest_object, eye_ray, point_of_intersection);
}

/*  Function: computeColor
//
//  The function computes the diffuse, specular (for now) color using the ray and normal information
//
//  Parameters:
//      VisibleObject* closest_object: The closest object intersecting the ray
//      Ray& eye_ray: The ray
//      Vector3& point_of_intersection: The point of intersection of ray and the object
//     
//	Return:
//      Vector3 color
*/
__host__ Vector3 RenderEngine::computeColor(VisibleObject* closest_object, Ray& eye_ray, Vector3& point_of_intersection) {
    // Vector3 light_direction = (point_of_intersection - world->light);
    // light_direction.make_unit_vector();
    Vector3 light_direction = (world->getLight(world->getSelectedLightId()))->getLightAtPoint(point_of_intersection);
    light_direction = -light_direction;

    Vector3 normal = closest_object->getNormalAtPoint(point_of_intersection);
    normal.make_unit_vector();

    Vector3 eye = -1.0f * (point_of_intersection - eye_ray.getStartingPoint());
    eye.make_unit_vector();

    float diffuse_intensity = max(0.0f, dot(normal, light_direction));
    diffuse_intensity = smoothstep(sharp_edge0, sharp_edge1, diffuse_intensity);

    #ifdef RENDERDEBUG
    std::cout<<"Point of intersection: "<<point_of_intersection<<std::endl;
    std::cout<<"normal: "<<normal<<std::endl;
    std::cout<<"light point: "<<world->light<<std::endl;
    std::cout<<"light: "<<light_direction<<std::endl;
    std::cout<<"Dot normal light: "<<dot(normal, light_direction)<<std::endl;
    std::cout<<"Diffuse: "<<diffuse_intensity<<std::endl<<std::endl;
    #endif

    Vector3 reflection = -light_direction + 2.0f * dot(light_direction, normal) * normal;
    reflection.make_unit_vector();
    float specular_intensity = max(0.0f, dot(eye, reflection));
    
    specular_intensity = smoothstep(sharp_edge0, sharp_edge1, specular_intensity);

    Vector3 object_color = closest_object->getColor(point_of_intersection);

    Vector3 final_object_color = diffuse_intensity * object_color;
    final_object_color = specular_intensity * object_color + (1.0f - specular_intensity) * final_object_color;

    final_object_color = ambient_intensity * object_color + (1 - ambient_intensity) * final_object_color;
    return final_object_color;
}

__host__ Vector3 RenderEngine::renderPixel(int i, int j)  {
    float u = ((float) i)/((float) w);
    float v = ((float) h - j)/((float) h); // Pixel cordinates have the origin on top left but our screen origin is on the bottom left

    #ifdef DEBUG
    std::cout<<u<<","<<v<<std::endl;
    #endif
    
    return render(u, v);
}

__host__ void RenderEngine::renderAllPixels() {
    Vector3 color_ij(0.0, 0.0, 0.0);

    // Output Pixel as Image
    std::cout << "P3\n" << w << " " << h << "\n255\n";
    for (int j = 0; j < h; j++) {
        for (int i = 0; i < w; i++) {
            color_ij = renderPixel(i, j);

            #ifdef DEBUG
            std::cout<<color_ij<<std::endl;
            #endif

            int ir = int(255.99*color_ij.r());
            int ig = int(255.99*color_ij.g());
            int ib = int(255.99*color_ij.b());

            #ifdef ACTUALRENDER
            std::cout << ir << " " << ig << " " << ib << "\n";
            #endif
        }
    }
}

// __host__ float max(float& a, float& b) {
//     if(a > b) {
//         return a;
//     }
//     return b;
// }
#endif