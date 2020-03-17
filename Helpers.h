#ifndef HELPERSH
#define HELPERSH

__host__ float clamp(float x, float high, float low);
__host__ float smoothstep(float edge0, float edge1, float x);


__host__ float clamp(float x, float low, float high) {
    return (x < low) ? low : (high < x) ? high : x;
}

__host__ float smoothstep(float edge0, float edge1, float x) {
    // Scale, bias and saturate x to 0..1 range
    x = clamp((x - edge0) / (edge1 - edge0), 0.0f, 1.0f);
    // Evaluate polynomial
    return x * x * (3 - 2 * x);
}

#endif