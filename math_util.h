#pragma once

#include <cuda_runtime.h>

typedef unsigned int uint;
typedef unsigned short ushort;


//host implementation of cuda functions
#ifndef __CUDACC__
#include <math.h>

inline float fminf(float a, float b)
{
    return a < b ? a : b;
}

inline float fmaxf(float a, float b)
{
    return a > b ? a : b;
}

inline int max(int a, int b)
{
    return a > b ? a : b;
}

inline int min(int a, int b)
{
    return a < b ? a : b;
}

inline float rsqrtf(float x)
{
    return 1.0f / sqrtf(x);
}
#endif


// lerp
inline __device__ __host__ float lerp(float a, float b, float t)
{
    return a + t * (b - a);
}

// clamp
inline __device__ __host__ float clamp(float f, float a, float b)
{
    return fmaxf(a, fminf(f, b));
}

//initializations

inline __host__ __device__ float3 make_float3(float s)
{
    return make_float3(s, s, s);
}



//operations
inline __host__ __device__ float3 operator-(float3& a)
{
    return make_float3(-a.x, -a.y, -a.z);
}

inline __host__ __device__ float3 operator+(float3 a, float3 b)
{
    return make_float3(a.x + b.x, a.y + b.y, a.z + b.z);
}

inline __host__ __device__ void operator+=(float3& a, float3 b)
{
    a.x += b.x; a.y += b.y; a.z += b.z;
}

inline __host__ __device__ float3 operator+(float3 a, float b)
{
    return make_float3(a.x + b, a.y + b, a.z + b);
}

inline __host__ __device__ void operator+=(float3& a, float b)
{
    a.x += b; a.y += b; a.z += b;
}

inline __host__ __device__ float3 operator-(float3 a, float3 b)
{
    return make_float3(a.x - b.x, a.y - b.y, a.z - b.z);
}

inline __host__ __device__ void operator-=(float3& a, float3 b)
{
    a.x -= b.x; a.y -= b.y; a.z -= b.z;
}

inline __host__ __device__ float3 operator-(float3 a, float b)
{
    return make_float3(a.x - b, a.y - b, a.z - b);
}

inline __host__ __device__ float3 operator-(float b, float3 a)
{
    return make_float3(b - a.x, b - a.y, b - a.z);
}

inline __host__ __device__ void operator-=(float3& a, float b)
{
    a.x -= b; a.y -= b; a.z -= b;
}

inline __host__ __device__ float3 operator*(float3 a, float3 b)
{
    return make_float3(a.x * b.x, a.y * b.y, a.z * b.z);
}

inline __host__ __device__ void operator*=(float3& a, float3 b)
{
    a.x *= b.x; a.y *= b.y; a.z *= b.z;
}

inline __host__ __device__ float3 operator*(float3 a, float b)
{
    return make_float3(a.x * b, a.y * b, a.z * b);
}

inline __host__ __device__ float3 operator*(float b, float3 a)
{
    return make_float3(b * a.x, b * a.y, b * a.z);
}

inline __host__ __device__ void operator*=(float3& a, float b)
{
    a.x *= b; a.y *= b; a.z *= b;
}

inline __host__ __device__ float3 operator/(float3 a, float3 b)
{
    return make_float3(a.x / b.x, a.y / b.y, a.z / b.z);
}
inline __host__ __device__ void operator/=(float3& a, float3 b)
{
    a.x /= b.x; a.y /= b.y; a.z /= b.z;
}
inline __host__ __device__ float3 operator/(float3 a, float b)
{
    return make_float3(a.x / b, a.y / b, a.z / b);
}
inline __host__ __device__ void operator/=(float3& a, float b)
{
    a.x /= b; a.y /= b; a.z /= b;
}
inline __host__ __device__ float3 operator/(float b, float3 a)
{
    return make_float3(b / a.x, b / a.y, b / a.z);
}


inline /*__host__*/ __device__ bool operator>(float3 a, float3 b) {
    return (a.x > b.x && a.y > b.y && a.z > b.z);
}

inline /*__host__*/ __device__ bool operator<(float3 a, float3 b) {
    return (a.x < b.x && a.y < b.y && a.z < b.z);
}

inline /*__host__*/ __device__ bool operator==(float3 a, float3 b) {
    return (a.x == b.x && a.y == b.y && a.z == b.z);
}

inline /*__host__*/ __device__ bool operator!=(float3 a, float3 b) {
    return (a.x != b.x && a.y != b.y && a.z != b.z);
}

//min-max float3

inline __host__ __device__ float3 fminf(float3 a, float3 b)
{
    return make_float3(fminf(a.x, b.x), fminf(a.y, b.y), fminf(a.z, b.z));
}

inline __host__ __device__ float3 fmaxf(float3 a, float3 b)
{
    return make_float3(fmaxf(a.x, b.x), fmaxf(a.y, b.y), fmaxf(a.z, b.z));
}

//lerp float3
inline __device__ __host__ float3 lerp(float3 a, float3 b, float t)
{
    return a + t * (b - a);
}

//clamp float3

inline __device__ __host__ float3 clamp(float3 v, float a, float b)
{
    return make_float3(clamp(v.x, a, b), clamp(v.y, a, b), clamp(v.z, a, b));
}
inline __device__ __host__ float3 clamp(float3 v, float3 a, float3 b)
{
    return make_float3(clamp(v.x, a.x, b.x), clamp(v.y, a.y, b.y), clamp(v.z, a.z, b.z));
}

//floor float3

inline __host__ __device__ float3 floorf(float3 v)
{
    return make_float3(floorf(v.x), floorf(v.y), floorf(v.z));
}

//3D Vector operations

inline __host__ __device__ float dot(float3 a, float3 b)
{
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

inline __host__ __device__ float length(float3 v)
{
    return sqrtf(dot(v, v));
}

inline __host__ __device__ float length_squared(float3 v) {
    return v.x * v.x + v.y * v.y + v.z * v.z;
}

inline __host__ __device__ float3 normalize(float3 v)
{
    float invLen = rsqrtf(dot(v, v));
    return v * invLen;
}

inline __host__ __device__ float3 cross(float3 a, float3 b)
{
    return make_float3(a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x);
}

//Ray reflect based on law of reflection (angle of incidence = angle of reflection)

inline __host__ __device__ float3 reflect(float3 i, float3 n)
{
    return i - 2.0f * n * dot(n, i);
}

inline __host__ __device__ float3 refract(float3& uv, float3& n, float etai_over_etat) {
    float cos_theta = dot(-normalize(uv), n);
    float3 r_out_perp = etai_over_etat * (uv + cos_theta * n);
    float3 r_out_parallel = -sqrtf(fabs(1.0f - length_squared(r_out_perp))) * n;
    return r_out_parallel + r_out_perp;
}

inline __host__ __device__ float schlick(float cosine, float ratio) {
    float R0 = (1.0f - ratio) * (1.0f - ratio) / (1.0f + ratio) * (1.0f + ratio);
    return R0 + (1.0f - R0) * cosine * cosine * cosine * cosine * cosine;
}


// absolute value
inline __host__ __device__ float3 fabs(float3 v)
{
    return make_float3(fabs(v.x), fabs(v.y), fabs(v.z));
}









