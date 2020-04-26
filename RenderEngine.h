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

#include <math.h>
#include <curand_kernel.h>

#include "Vector3.h"
#include "World.h"
#include "Camera.h"
#include "Ray.h"
#include "VisibleObject.h"
#include "Helpers.h"
#include "Managed.h"
#include "Light.h"

class RenderEngine {
    private:
        int w, h;

        int n_samples;
        bool use_antialising;

        float sharp_edge0, sharp_edge1;

        float ambient_intensity;

        bool if_border;
        float border_thinkness;

        int recursive_steps;
        int iterative_ray_steps;

        // Given a point and a light, compute how much the ray from the point to the light crosses any other object
        __device__ float computeShadowIntensityAtPoint(Vector3 point_of_intersection, Light * light);

        __device__ inline bool ifRayIntersected(const Vector3& intersectInfo) { return (intersectInfo[2] > 0.0f); }

        __device__ inline float getTFromIntersectInfo(const Vector3& intersectInfo) { return intersectInfo[0]; }

        __device__ inline float ifIntersectPointInside(const Vector3& intersectInfo) { return (intersectInfo[1] > 0.0f); }

        __device__ inline int getTriangleIDFromIntersectInfo(const Vector3& intersectInfo) { return int(intersectInfo[1]); }

    public:
        __device__ RenderEngine();
        __device__ RenderEngine(int width, int height, World& world_p);
        __device__ ~RenderEngine();

        __device__ Vector3 render(float u, float v, curandState& rand_state); //Renders the point u,v on the screen (0 <= u,v <= 1)
        __device__ Vector3 computeColor(VisibleObject* closest_object, Ray& eye_ray, Vector3& t, curandState& rand_state, Vector3& extraIntersectInfo); // Compute color given the closest object

        // Computes color iteratively
        __device__ Vector3 rayTrace(Ray incoming_ray, int rec_steps, curandState& rand_state);

        __device__ bool getHitObject(Ray incoming_ray, VisibleObject* & object, Vector3& intersectInfo);

        __device__ Vector3 renderPixel(int i, int j, curandState& rand_state); // Renders the pixel i,j

        __device__ inline void setSharpEdge(float edge0, float edge1) {sharp_edge0 = edge0; sharp_edge1 = edge1;}

        __device__ void setBorder(bool border, float thickness) { if_border = border; border_thinkness = thickness; }

        __device__ inline void setAntiAliasing(int samples) { use_antialising = true; n_samples = samples; }

        __device__ Vector3 renderPixelSampling(int i, int j, curandState& rand_state);
        World* world;
        Camera* camera;
};

__device__ RenderEngine::RenderEngine() {}

__device__ RenderEngine::RenderEngine(int width, int height, World& world_p) : w(width), h(height) {
    world = &world_p;
    camera = world->getCamera();
    sharp_edge0 = 0.0;
    sharp_edge1 = 1.0;
    ambient_intensity = 0.2;
    use_antialising = false;
    n_samples = 32;
    if_border = false;
    border_thinkness = 0.6;
    recursive_steps = 3;
    iterative_ray_steps = 3;
}

__device__ RenderEngine::~RenderEngine() {}

__device__ Vector3 RenderEngine::render(float u, float v, curandState& rand_state) {
    Ray eye_ray = camera->getRay(u, v);

    return rayTrace(eye_ray, iterative_ray_steps, rand_state);
}

__device__ Vector3 RenderEngine::rayTrace(Ray incoming_ray, int steps, curandState& rand_state) {
    Vector3 final_color(0.0f, 0.0f, 0.0f);

    Ray current_ray = incoming_ray;

    float out_ri = 1.0f; // Outer refractive index
    float inner_ri = 1.3; // Innter refractive index

    // TEMPORARY ONLY
    float idx_refr = 0.6;

    Vector3 intersectInfo;

    VisibleObject * closest_object;

    Vector3 point_of_intersection, normal, refl_dir, refr_dir, ray_dir;
    float min_t;
    Ray reflected_ray, refracted_ray;

    curandState rand_state_pixel = rand_state;
    float lambda = 0.5;
    double lambda_acc = 1; // Accumalates the lambda factor 
    for(int i = 0; i < steps; i++) {
        bool if_intersected = getHitObject(current_ray, closest_object, intersectInfo);
        if(if_intersected) {
            min_t = getTFromIntersectInfo(intersectInfo);
            point_of_intersection = current_ray.getPoint(min_t);
            
            lambda_acc *= lambda;
            final_color += lambda_acc * computeColor(closest_object, current_ray, point_of_intersection, rand_state, intersectInfo);
            
            normal = closest_object->getNormalAtPoint(point_of_intersection);
            if(ifIntersectPointInside(intersectInfo)) {
                normal *= -1;
            }
            // Special case for Triangular Mesh    
            if(closest_object->getTypeID() == TMESH_TYPE_ID) {
                normal = closest_object->getNormalAtPoint(point_of_intersection, getTriangleIDFromIntersectInfo(intersectInfo));
            }

            ray_dir = - current_ray.getDirection();
            refl_dir = -ray_dir + 2.0f * dot(ray_dir, normal) * normal;
            reflected_ray  = Ray(point_of_intersection, refl_dir);

            // Calculating a random refracted ray for now
            idx_refr = abs(dot(normal, current_ray.getDirection()));
            refr_dir = -normal * idx_refr + (-ray_dir - dot(-normal, -ray_dir) *(- normal)) * (1 - idx_refr);
            refracted_ray = Ray(point_of_intersection, refr_dir);

            // current_ray = reflected_ray;
            // current_ray = refracted_ray;

            if(curand_uniform(&rand_state_pixel) > 0.5f) {
                current_ray = reflected_ray;
            } else {
                current_ray = refracted_ray;
            }

        } else {
            break;
        }
    }
    rand_state = rand_state_pixel;
    return final_color;
}

__device__ bool RenderEngine::getHitObject(Ray incoming_ray, VisibleObject*& closest_object, Vector3& extraIntersectInfo) {
    int total_objects = world->getTotalVisibleObjects();

    float min_t = 0.0f;
    bool if_t_encountered = false;
    Vector3 intersectInfo;

    for(int i = 0; i < total_objects; i++) {
        
        intersectInfo = (world->getVisibleObject(i))->getIntersectInfo(incoming_ray);
        
        #ifdef DEBUG
        std::cout<<"Intersect info: "<<intersectInfo<<std::endl;
        #endif

        if( !if_t_encountered && intersectInfo[2] > 0.0f) {
            if_t_encountered = true;
            min_t = intersectInfo[0];
            extraIntersectInfo = intersectInfo;
            closest_object = world->getVisibleObject(i);
        } else if(if_t_encountered && intersectInfo[2] > 0.0f) {
            if(min_t > intersectInfo[0]) {
                min_t = intersectInfo[0];
                extraIntersectInfo = intersectInfo;
                closest_object = world->getVisibleObject(i);
            }
        }

    }

    return if_t_encountered; 
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
__device__ Vector3 RenderEngine::computeColor(VisibleObject* closest_object, Ray& eye_ray, Vector3& point_of_intersection, curandState& rand_state, Vector3& extraIntersectInfo) {
    
    int total_lights = world->getTotalLights();

    Vector3 normal = closest_object->getNormalAtPoint(point_of_intersection);
    if(ifIntersectPointInside(extraIntersectInfo)) {
        normal *= -1;
    }

    // Special case for Triangular Mesh    
    if(closest_object->getTypeID() == TMESH_TYPE_ID) {
        normal = closest_object->getNormalAtPoint(point_of_intersection, getTriangleIDFromIntersectInfo(extraIntersectInfo));
        #ifdef MESHDEBUG
            // printf("RenderNormal: %f %f %f\n", normal.x(), normal.y(), normal.z());
            printf("Intersection Mesh: %f %f %f\n", point_of_intersection.x(), point_of_intersection.y(), point_of_intersection.z());
            // printf("RenderExtraInInfo: %f %f %f\n", extraIntersectInfo.x(), extraIntersectInfo.y(), extraIntersectInfo.z());
        #endif
    }

    normal.make_unit_vector();
    

    Vector3 object_color = closest_object->getColor(point_of_intersection);

    // Special casee for Triangular Mesh
    if(closest_object->getTypeID() == TMESH_TYPE_ID) {
        object_color = closest_object->getColor(point_of_intersection, getTriangleIDFromIntersectInfo(extraIntersectInfo));
    }

    Vector3 final_object_color(0.0, 0.0, 0.0);
    Vector3 specular_light_color(1.0, 1.0, 1.0);
    
    Vector3 eye = -1.0f * (point_of_intersection - eye_ray.getStartingPoint());
    eye.make_unit_vector();

    Vector3 reflection;

    //Iterating through each light
    for(int i = 0; i < total_lights; i++) {
        if((world->getLight(i))->ifSamplingRequired()) {
            (world->getLight(i))->setRandomSamplePosition(rand_state);
        }
        Vector3 light_direction = (world->getLight(i))->getLightAtPoint(point_of_intersection);
        light_direction = -light_direction;

        // Computing Diffuse 
        float diffuse_intensity = max(0.0f, dot(normal, light_direction));
        // diffuse_intensity = smoothstep(sharp_edge0, sharp_edge1, diffuse_intensity);

        // Computing Specular
        Vector3 light_direction_unit = unit_vector(light_direction);
        reflection = -light_direction + 2.0f * dot(light_direction_unit, normal) * normal;
        reflection.make_unit_vector();
        float specular_intensity = max(0.0f, dot(eye, reflection));
        specular_intensity = pow(specular_intensity, 4);
        
        // specular_intensity = smoothstep(sharp_edge0, sharp_edge1, specular_intensity);
        specular_intensity = smoothstep(0.8, 1.0, specular_intensity);

        float shadow_intensity = 0.0f;
        if(dot(normal, light_direction) >= 0.0f) { // Compute the shadow only if dot(N,L) > 0
            shadow_intensity = computeShadowIntensityAtPoint(point_of_intersection, world->getLight(i));
            if(shadow_intensity > 1.0f) {
                shadow_intensity = 1.0;
            }
        }

        shadow_intensity = smoothstep(0.0f, 0.2, shadow_intensity);

        diffuse_intensity = (1.0f - shadow_intensity) * diffuse_intensity;
        specular_intensity = (1.0f - shadow_intensity) * specular_intensity;
        
        #ifdef SHADOWDEBUG
            printf("Shadow Intensity: %f\n", shadow_intensity);

            if(shadow_intensity > 0.0f) {
                specular_intensity = 0.0f;
                diffuse_intensity = 0.0f;
            }
        #endif
        
        // Computing the final color
        Vector3 color_from_light;
        color_from_light = diffuse_intensity * object_color;
        color_from_light = specular_intensity * specular_light_color + (1.0f - specular_intensity) * color_from_light;

        // Adding the color to the final color
        final_object_color += color_from_light;

        #ifdef RENDERDEBUG
            if(border_intensity < 0.0 || border_intensity > 1.0) {
                std::cout<<"Point of intersection: "<<point_of_intersection<<std::endl;
                std::cout<<"normal: "<<normal<<std::endl;
                std::cout<<"View: "<<view_unit_vector<<std::endl;
                std::cout<<"light point: "<<world->light<<std::endl;
                std::cout<<"light: "<<light_direction<<std::endl;
                std::cout<<"Dot normal light: "<<dot(normal, light_direction)<<std::endl;
                std::cout<<"Diffuse: "<<diffuse_intensity<<std::endl;
                std::cout<<"Dot: "<<dot(normal, view_unit_vector)<<std::endl;
                std::cout<<"Border param: "<<border_intensity<<std::endl<<std::endl;
            }
        #endif  
    }

    if(total_lights > 0) {
        final_object_color = final_object_color / total_lights;
    }

    // Computing border parameter

    float border_intensity = 0;
    if(if_border) {
        border_intensity =  max(0.0f, 1.0f - abs(dot(normal, eye)));
        border_intensity = smoothstep(border_thinkness, border_thinkness + 0.1, border_intensity);
    } else {
        border_intensity = 0;
    }
    final_object_color = (1.0f - border_intensity) * final_object_color;

    // Adding Ambient intensity
    final_object_color = ambient_intensity * object_color + (1 - ambient_intensity) * final_object_color;

    return final_object_color;
}

__device__ float RenderEngine::computeShadowIntensityAtPoint(Vector3 point_of_intersection, Light * light) {
    float shadow_intensity = 0.0f;

    Vector3 light_direction = light->getLightAtPoint(point_of_intersection);
    light_direction.make_unit_vector();
    Vector3 light_direction_from_object = -light_direction;
    Vector3 light_position = light->getLightPosition();
    float distance_intersection_light = (light_position - point_of_intersection).length();

    Ray ray_from_light = Ray(light_position, light_direction);
    Ray ray_from_object = Ray(point_of_intersection, light_direction_from_object);

    
    int total_objects = world->getTotalVisibleObjects();

    float t_from_object, t_from_light;
    Vector3 intersectInfo_from_object, intersectInfo_from_light;

    int hit_objects = 0;

    for(int i = 0; i < total_objects; i++) {    
        intersectInfo_from_light = (world->getVisibleObject(i))->getIntersectInfo(ray_from_light);
        intersectInfo_from_object = (world->getVisibleObject(i))->getIntersectInfo(ray_from_object);
        
        if(ifRayIntersected(intersectInfo_from_light) && ifRayIntersected(intersectInfo_from_object)) { // Checking if light ray intersected with both the objects
            t_from_light = getTFromIntersectInfo(intersectInfo_from_light);
            t_from_object = getTFromIntersectInfo(intersectInfo_from_object);

            if(t_from_object < distance_intersection_light && t_from_object >  0.0001) { // If the ray does not go beyond the light position
                                                                                          // Also checking if the distance from the object is not 0 (it is not intersecting with itself)
                shadow_intensity += (distance_intersection_light - t_from_light - t_from_object) / distance_intersection_light; 
                hit_objects++;

                #ifdef SHADOWDEBUG
                    printf("Shadow Intensity Inside Function: %f %f %f %f\n", shadow_intensity, t_from_light, t_from_object, distance_intersection_light);
                #endif
                #ifdef MESHDEBUG
                    // printf("Shadow Intensity Inside Function: %f %f %f %f Type: %d Triangle: %f %f\n", shadow_intensity, t_from_light, t_from_object, distance_intersection_light, (world->getVisibleObject(i))->getTypeID(), intersectInfo_from_light[1], intersectInfo_from_object[1]);
                #endif
            }
        }
    }
    
    return shadow_intensity;
}

__device__ Vector3 RenderEngine::renderPixelSampling(int i, int j, curandState& rand_state)  {
    float origin_u = ((float) i)/((float) w);
    float origin_v = ((float) h - j)/((float) h); // Pixel cordinates have the origin on top left but our screen origin is on the bottom left

    float u, v;
    curandState rand_state_pixel = rand_state;

    Vector3 final_color(0.0, 0.0, 0.0);

    for(int i = 0; i < n_samples; i++) {
        u = origin_u + curand_uniform(&rand_state_pixel)/(float(w));
        v = origin_v + curand_uniform(&rand_state_pixel)/(float(h));
        final_color += render(u, v, rand_state_pixel);

        #ifdef AREALIGHTDEBUG
            // printf("u: %f\n", u);
        #endif
    }
    final_color = final_color / n_samples;

    rand_state = rand_state_pixel;

    return final_color;
}

__device__ Vector3 RenderEngine::renderPixel(int i, int j, curandState& rand_state)  {
    float u = ((float) i)/((float) w);
    float v = ((float) h - j)/((float) h); // Pixel cordinates have the origin on top left but our screen origin is on the bottom left

    #ifdef DEBUG
    std::cout<<u<<","<<v<<std::endl;
    #endif
    return render(u, v, rand_state);
}

#endif