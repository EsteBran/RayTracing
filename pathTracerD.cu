// smallptCUDA by Sam Lapere, 2015
// based on smallpt, a path tracer by Kevin Beason, 2008  

#include <iostream>

#include <GL/glew.h>
#include <cuda_gl_interop.h>

#include <cuda_runtime.h>

#include <curand.h>
#include <curand_kernel.h>
#include <device_launch_parameters.h>



#include "pathTracer.h"

#define PI 3.14159265359f  // pi


cudaGraphicsResource* cudapbo;
uchar4* dev_map = NULL;
float3* output_device = NULL; //pointer to image on vram
float3* output_previous = NULL;
float3* accumulate_buffer = NULL;

//assert style function and wrapper macro, error checking can be done by 
//wrapping each API call 
#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char* file, int line, bool abort = true) {
    if (code != cudaSuccess) {
        fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort) exit(code);
    }
}


// __device__ : executed on the device (GPU) and callable only from the device

struct Ray {
    float3 orig; // ray origin
    float3 dir;  // ray direction 
    __device__ Ray(float3 o_, float3 d_) : orig(o_), dir(d_) {}
};

struct Camera {
    float3 orig;
    float3 dir;
    float fov; //field of view
   
    Camera(float3 o_, float3 d_, float fov_) : orig(o_), dir(d_), fov(fov_) {}
};

Camera mainCam(make_float3(0.0f), make_float3(0.0f), 0.0f); //main viewpoint camera
bool camChanged = false;

//set initial camera parameters
void setCamera(float ox, float oy, float oz, float dx, float dy, float dz, float fov_) {
    mainCam.orig = make_float3(ox, oy, oz);
    mainCam.dir = make_float3(dx, dy, dz);
    mainCam.fov = fov_;

    //printf("set camera is %f %f %f  \n", mainCam.orig.x, mainCam.orig.y, mainCam.orig.z);
}

//function for moving camera, positive theta means rotate right, positive phi means rotate up
void changeCamera(float movex, float movey, float movez, float theta) {
    mainCam.orig += { movex, movey, movez }; //translation

    //horizontal rotation (relative to camera start, positive angle is rotation rightward)
    if (theta > 0.0f) {
        //convert degrees to radians
        theta = PI / 180 * theta;
        theta = -theta;
        mainCam.dir.x = cos(theta)*mainCam.dir.x + sin(theta)*mainCam.dir.z;
        mainCam.dir.y = mainCam.dir.y;
        mainCam.dir.z = -sin(theta) * mainCam.dir.x + cos(theta) * mainCam.dir.z;
        mainCam.dir = normalize(mainCam.dir);
    }

    

    camChanged = true;
    //printf("changed camera is %f %f %f  \n", mainCam.orig.x, mainCam.orig.y, mainCam.orig.z);
};



enum material { DIFFUSE, SPECULAR, REFRACT };  // material types, used in radiance(), only DIFF used here

struct Sphere {

    float radius;            // radius 
    float3 position, emission, color; // position, emission, colour 
    material mat;          // reflection type (e.g. diffuse)

    __device__ float intersect_sphere(const Ray& r) const {
        float3 op = position - r.orig;    // distance from ray.orig to center sphere 
        float t, epsilon = 0.0001f;  // epsilon required to prevent floating point precision artefacts
        float b = dot(op, r.dir);    // b in quadratic equation
        float discriminant = b * b - dot(op, op) + radius * radius;  // discriminant quadratic equation
        if (discriminant < 0) return 0;       // if disc < 0, no real solution (we're not interested in complex roots) 
        else discriminant = sqrtf(discriminant);    // if disc >= 0, check for solutions using negative and positive discriminant
        return (t = b - discriminant) > epsilon ? t : ((t = b + discriminant) > epsilon ? t : 0); // pick closest point in front of ray origin
    }
};

// SCENE
// 9 spheres forming a Cornell box, small enough for gpu constant memory
__constant__ Sphere spheres[] = {
 { 1e5f, { 1e5f + 1.0f, 40.8f, 81.6f }, { 0.0f, 0.0f, 0.0f }, { 0.95f, 0.05f, 0.05f }, DIFFUSE }, //Left 
 { 1e5f, { -1e5f + 99.0f, 40.8f, 81.6f }, { 0.0f, 0.0f, 0.0f }, { .05f, .95f, .05f }, DIFFUSE }, //Rght 
 { 1e5f, { 50.0f, 40.8f, 1e5f }, { 0.0f, 0.0f, 0.0f }, { .75f, .75f, .75f }, DIFFUSE }, //Back 
 { 1e5f, { 50.0f, 40.8f, -1e5f + 600.0f }, { 0.0f, 0.0f, 0.0f }, { 1.00f, 1.00f, 1.00f }, DIFFUSE }, //Frnt 
 { 1e5f, { 50.0f, 1e5f, 81.6f }, { 0.0f, 0.0f, 0.0f }, { .75f, .75f, .75f }, DIFFUSE }, //Botm 
 { 1e5f, { 50.0f, -1e5f + 81.6f, 81.6f }, { 0.0f, 0.0f, 0.0f }, { .75f, .75f, .75f }, DIFFUSE }, //Top 
 { 16.5f, { 27.0f, 16.5f, 47.0f }, { 0.0f, 0.0f, 0.0f }, { 1.0f, 1.0f, 1.0f }, DIFFUSE }, // small sphere 1
 { 16.5f, { 73.0f, 16.5f, 78.0f }, { 0.0f, 0.0f, 0.0f }, { 1.0f, 1.0f, 1.0f }, REFRACT }, // small sphere 2
 { 16.5f, { 40.0f, 56.5f, 78.0f }, { 0.0f, 0.0f, 0.0f }, { 1.0f, 1.0f, 1.0f }, SPECULAR }, // small sphere 3
 { 600.0f, { 50.0f, 681.6f - .77f, 81.6f }, { 1.0f, 0.9f, 0.9f }, { 0.0f, 0.0f, 0.0f }, DIFFUSE }  // Light
};

//returns if any intersections happen and sets t to distance of closest ray intersection
//sets id as the id of object hit
__device__ inline bool intersect_scene(const Ray& r, float& t, int& id) {
    float n = sizeof(spheres) / sizeof(Sphere), distance, inf = t = 1e20;  // t is distance to closest intersection, initialise t to a huge number outside scene
    for (int i = int(n); i--;)  // test all scene objects for intersection
        if ((distance = spheres[i].intersect_sphere(r)) && distance < t) {  //true if newly computed intersection distance d is smaller than current closest intersection distance
            t = distance; 
            id = i; 
        }
    return t < inf; // returns true if an intersection with the scene occurred, false when no hit
}

//for hashed frame number
uint wangHash(uint a) {
    a = (a ^ 61) ^ (a >> 16);
    a = a + (a << 3);
    a = a ^ (a >> 4);
    a = a * 0x27d4eb2d;
    a = a ^ (a >> 15);
    return a;
}



// radiance function, the meat of path tracing 
// solves the rendering equation: 
__device__ float3 radiance(Ray& r, curandState* randState) { // returns ray color

    float3 accumulatedColor = make_float3(0.0f, 0.0f, 0.0f); // accumulates ray colour with each iteration through bounce loop
    float3 mask = make_float3(1.0f, 1.0f, 1.0f);

    // ray bounce loop (no Russian Roulette used) 
    for (int bounces = 0; bounces < 10; bounces++) {  // iteration up to 4 bounces (replaces recursion in CPU code)

        float t;           // distance to closest intersection 
        int id = 0;        // index of closest intersected sphere 

        // test ray for intersection with scene
        if (!intersect_scene(r, t, id))
            return make_float3(0.0f, 0.0f, 0.0f); // if miss, return black

        // else, we've got a hit
        const Sphere& obj = spheres[id];  // hitobject
        float3 hitpoint = r.orig + r.dir * t;          
        float3 normal = normalize(hitpoint - obj.position);    
        float3 front_facing_normal = dot(normal, r.dir) < 0 ? normal : -normal; 
        float3 d; //ray direction of next path 

        // add emission of current sphere to accumulated colour
        // (first term in rendering equation sum) 
        accumulatedColor += mask * obj.emission;
        

        //Shading passes - Diffuse, Reflect, Refract

        //ideal lambertian diffuse reflector
        if (obj.mat == DIFFUSE) {
            // create 2 random numbers
            float r1 = 2 * PI * curand_uniform(randState); // pick random number on unit circle (radius = 1, circumference = 2*Pi) for azimuth
            float r2 = curand_uniform(randState);  // pick random number for elevation
            float r2s = sqrtf(r2);

            // compute local orthonormal basis uvw at hitpoint to use for calculation random ray direction 
            float3 w = front_facing_normal;
            float3 u = normalize(cross((fabs(w.x) > .1 ? make_float3(0, 1, 0) : make_float3(1, 0, 0)), w));
            float3 v = cross(w, u);

            // compute random ray direction on hemisphere using polar coordinates
            // cosine weighted importance sampling (favours ray directions closer to normal direction)
            d = normalize(u * cos(r1) * r2s + v * sin(r1) * r2s + w * sqrtf(1 - r2));

            // new ray origin is intersection point of previous ray with scene
            hitpoint += front_facing_normal * 0.03f; // offset ray origin slightly to prevent self intersection
           

            mask *= obj.color;    // multiply with colour of object       
            mask *= dot(d, front_facing_normal);  // weigh light contribution using cosine of angle between incident light and normal
            mask *= 2;          // fudge factor
        }

        if (obj.mat == SPECULAR) {

            d = reflect(r.dir, normal); //law of reflection

            hitpoint += front_facing_normal * 0.01f;

            mask *= obj.color;
        }

        if (obj.mat == REFRACT) {

            bool into = dot(front_facing_normal, normal) > 0; // is ray entering or leaving refractive material
            float index1 = 1.0f;  // Index of Refraction air
            float index2 = 1.5f; //index of refraction water
            float refraction_ratio = into ? index1/index2 : index2/index1;  // IOR ratio of refractive materials
            float cos_theta = dot(r.dir, normal);
            float cos2phi = 1.0f - refraction_ratio * refraction_ratio * (1.f - cos_theta*cos_theta);

            if (cos2phi < 0.0f) // total internal reflection 
            {
                d = reflect(r.dir, front_facing_normal); //d = r.dir - 2.0f * n * dot(n, r.dir);
                hitpoint += front_facing_normal * 0.001f;
            }
            else // cos2t > 0
            {
                // compute direction of transmission ray

                float3 tdir = refract(r.dir, (into ? normal : -normal), refraction_ratio);
                //float3 tdir = normalize(r.dir * refraction_ratio - (into ? normal : -normal) * (sqrtf(1 - cos2phi)));

               

                //Schlick's approximation
                float R0 = (index2 - index1) * (index2 - index1) / (index2 + index1) * (index2 + index1);
                float c = 1.f - (into ? -cos_theta : dot(tdir, normal));
                float schlick = R0 + (1.f - R0) * c * c * c * c * c; 
                float transmission = 1 - schlick; // Transmission
                float P = .25f + .5f * schlick;
                float RP = schlick / P;
                float TP = transmission / (1.f - P);
                

                // randomly choose reflection or transmission ray
                if (schlick > curand_uniform(randState)) // reflection ray
                {
                    mask *= RP;
                    d = reflect(r.dir, normal);
                    hitpoint += front_facing_normal * 0.02f;

                }
                else // transmission ray
                {
                    mask *= TP;
                    d = tdir; //r = Ray(x, tdir); 
                    hitpoint += front_facing_normal * 0.000005f; // epsilon must be small to avoid artefacts
                                               
                }

            }

        }
        //set up origin/direction of next path segment
        r.orig = hitpoint;
        r.dir = d;
        
    }

    return accumulatedColor;
}


// __global__ : executed on the device (GPU) and callable only from host (CPU) 
// this kernel runs in parallel on all the CUDA threads

__global__ void render_kernel(float3* output, float3* previous, Camera cam, int width, int height, int hashedFrame, int iteration, int samples) {

    // assign a CUDA thread to every pixel (x,y) 
    // blockIdx, blockDim and threadIdx are CUDA specific keywords
    // replaces nested outer loops in CPU code looping over image rows and image columns 
    unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;

    int threadId = (blockIdx.x + blockIdx.y * gridDim.x) * (blockDim.x * blockDim.y) + (threadIdx.y * blockDim.x) + threadIdx.x;

    curandState randState;
    curand_init(hashedFrame + threadId, 0, 0, &randState); //for getting a new hash every frame, which results in more spread out random rays

    int i = x + y*width; // pixel index

    // generate ray directed at lower left corner of the screen
    // compute directions for all other rays by adding cx and cy increments in x and y direction
    float3 cx = make_float3(width * cam.fov / height, 0.0f, 0.0f); // ray direction offset in x direction
    float3 cy = normalize(cross(cx, cam.dir)) * cam.fov; // ray direction offset in y direction (.5135 is field of view angle)
    float3 r; // r is final pixel color       

    r = make_float3(0.0f); // reset r to zero for every pixel 

   // compute primary ray direction, added jitter for anti aliasing
   float3 d = cam.dir + cx * ((.25 + x + curand_uniform(&randState)) / width - .5) + cy * ((.25 + y + curand_uniform(&randState)) / height - .5);

    // create primary ray, add incoming radiance to pixelcolor
    r = radiance(Ray(cam.orig + d * 40, normalize(d)), &randState) * (1.0f / samples);
   
    // write rgb value of pixel to image buffer on the GPU, clamp value to [0.0f, 1.0f] range
   
    // if iterations < samples then keep accumulating, save to previous buffer
    if (iteration < samples) { 
         output[i].x += clamp(r.x, 0.0f, 1.0f);
         output[i].y += clamp(r.y, 0.0f, 1.0f);
         output[i].z += clamp(r.z, 0.0f, 1.0f);

         previous[i] = output[i];
    }
    else { //once iteration == samples then just display the previous buffer, since no more calculation needs to be done
        output[i] = previous[i];
        return;
    }

    
        
}

  


inline float clamp(float x) { return x < 0.0f ? 0.0f : x > 1.0f ? 1.0f : x; }
inline int toInt(float x) { return int(pow(clamp(x), 1 / 2.2) * 255 + .5); }  // convert RGB float in range [0,1] to int in range [0, 255] and perform gamma correction

void allocateMemory(int width, int height) {
    cudaMalloc(&output_device, width * height * sizeof(float3)); // allocate memory on the CUDA device (GPU VRAM)
    cudaMalloc(&output_previous, width * height * sizeof(float3));
}

// free CUDA memory
void deallocateMemory() {
    cudaFree(output_device);
}




//binds pixel buffer to cuda graphics resource, allowing cuda to access pbo
void kernelBindPbo(GLuint pixelBufferObj) {
    gpuErrchk(cudaGraphicsGLRegisterBuffer(&cudapbo, pixelBufferObj, cudaGraphicsRegisterFlagsWriteDiscard));
}

//unregisters the pbo and unbinds from pbo
void kernelExit(GLuint pixelBufferObj) {
    gpuErrchk(cudaGLUnregisterBufferObject(pixelBufferObj));
    gpuErrchk(cudaGraphicsUnregisterResource(cudapbo));
}


//sends image from output buffer to PBO, which then sends image to screen
__global__ void sendImageToPBO(uchar4* pbo, int width, int height, float3* image) {
    int x = (blockIdx.x * blockDim.x) + threadIdx.x;
    int y = (blockIdx.y * blockDim.y) + threadIdx.y;

    if (x < width && y < height) {
        int index = x + (y * width);
        float3 pix = image[index]; //color of a pixel

        float3 color;
        color.x = clamp((int)(powf(pix.x, 1 / 2.2f) * 255.0), 0, 255);
        color.y = clamp((int)(powf(pix.y, 1 / 2.2f) * 255.0), 0, 255);
        color.z = clamp((int)(powf(pix.z, 1 / 2.2f) * 255.0), 0, 255);

        pbo[index].w = 0;
        pbo[index].x = color.x;
        pbo[index].y = color.y;
        pbo[index].z = color.z;
    }

}

__global__ void clearScreen(uchar4* pbo, int width, int height, float3* image) {
    int x = (blockIdx.x * blockDim.x) + threadIdx.x;
    int y = (blockIdx.y * blockDim.y) + threadIdx.y;

    if (x < width && y < height) {
        int index = x + (y * width);
        

        image[index].x = 0.0f;
        image[index].y = 0.0f;
        image[index].z = 0.0f;


        pbo[index].w = 0;
        pbo[index].x = 0.0f;
        pbo[index].y = 0.0f;
        pbo[index].z = 0.0f;
    }
}


//renders each sample iteratively
void renderKernel(int width, int height, int iteration, int samples) {


    gpuErrchk(cudaGraphicsMapResources(1, &cudapbo, NULL));
    gpuErrchk(cudaGraphicsResourceGetMappedPointer((void**)&dev_map, NULL, cudapbo));

    // dim3 is CUDA specific type, block and grid are required to schedule CUDA threads over streaming multiprocessors
    dim3 block(8, 8, 1);
    dim3 grid(width / block.x, height / block.y, 1);


    if (camChanged) {
        clearScreen << <grid, block >> > (dev_map, width, height, output_device);
        camChanged = false;
    }
    render_kernel << < grid, block >> > (output_device, output_previous, mainCam, width, height, wangHash(iteration), iteration, samples);
    sendImageToPBO << <grid, block >> > (dev_map, width, height, output_device);
    
    gpuErrchk(cudaDeviceSynchronize());
    gpuErrchk(cudaGraphicsUnmapResources(1, &cudapbo, NULL));


}
