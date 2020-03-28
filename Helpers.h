#ifndef HELPERSH
#define HELPERSH

#define PI 3.1415927

__host__ __device__ float clamp(float x, float high, float low);
__host__ __device__ float smoothstep(float edge0, float edge1, float x);
__host__ void makeImage(Vector3 * frame_buffer, int w, int h);


__host__ __device__ float clamp(float x, float low, float high) {
    return (x < low) ? low : (high < x) ? high : x;
}

__host__ __device__ float smoothstep(float edge0, float edge1, float x) {
    // Scale, bias and saturate x to 0..1 range
    x = clamp((x - edge0) / (edge1 - edge0), 0.0f, 1.0f);
    // Evaluate polynomial
    return x * x * (3 - 2 * x);
}

__host__ void makeImage(Vector3 * frame_buffer, int w, int h) {
    // Output Pixel as Image
    #ifdef ACTUALRENDER
    std::cout << "P3\n" << w << " " << h << "\n255\n";
    #endif

    int index_ij;
    for (int j = h - 1; j >= 0; j--) {
        for (int i = 0; i < w; i++) {
            index_ij = j * w + i;

            #ifdef CUDADEBUG
            std::cout<<"makeImage: index: "<<index_ij<<" i: "<<i<<" j: "<<j<<std::endl;
            #endif
            
            int ir = int(255.99*frame_buffer[index_ij].r());
            int ig = int(255.99*frame_buffer[index_ij].g());
            int ib = int(255.99*frame_buffer[index_ij].b());

            #ifdef ACTUALRENDER
            std::cout << ir << " " << ig << " " << ib << "\n";
            #endif
        }
    }
}
#endif