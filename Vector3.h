#ifndef VECTOR3H
#define VECTOR3H

class Vector3
{
    private:

    public:

        __host__ __device__ Vector3();
        __host__ __device__ Vector3(float x, float y, float z);
        __host__ __device__ ~Vector3();

        __host__ __device__ inline float x() const { return members[0]; }
        __host__ __device__ inline float y() const { return members[1]; }
        __host__ __device__ inline float z() const { return members[2]; }
        __host__ __device__ inline float r() const { return members[0]; }
        __host__ __device__ inline float g() const { return members[1]; }
        __host__ __device__ inline float b() const { return members[2]; }

        __host__ __device__ inline const Vector3& operator+() const { return *this; }
        __host__ __device__ inline Vector3 operator-() const { return Vector3(-members[0], -members[1], -members[2]); }
        __host__ __device__ inline float operator[](int i) const { return members[i]; }
        __host__ __device__ inline float& operator[](int i) { return members[i]; };

        __host__ __device__ inline Vector3& operator+=(const Vector3 &v2);
        __host__ __device__ inline Vector3& operator-=(const Vector3 &v2);
        __host__ __device__ inline Vector3& operator*=(const Vector3 &v2);
        __host__ __device__ inline Vector3& operator/=(const Vector3 &v2);
        __host__ __device__ inline Vector3& operator*=(const float t);
        __host__ __device__ inline Vector3& operator/=(const float t);

        __host__ __device__ inline float length() const { return sqrt(members[0]*members[0] + members[1]*members[1] + members[2]*members[2]); }
        __host__ __device__ inline float squared_length() const { return members[0]*members[0] + members[1]*members[1] + members[2]*members[2]; }
        __host__ __device__ inline void make_unit_vector();

        float members[3];
};

__host__ __device__ Vector3::Vector3() {}

__host__ __device__ Vector3::Vector3(float x, float y, float z)
{
    members[0] = x;
    members[1] = y;
    members[2] = z;
}

__host__ __device__ Vector3::~Vector3() {}

inline std::istream& operator>>(std::istream &is, Vector3 &t) {
    is >> t.members[0] >> t.members[1] >> t.members[2];
    return is;
}

inline std::ostream& operator<<(std::ostream &os, const Vector3 &t) {
    os << t.members[0] << " " << t.members[1] << " " << t.members[2];
    return os;
}

__host__ __device__ inline void Vector3::make_unit_vector() {
    float k = 1.0 / sqrt(members[0]*members[0] + members[1]*members[1] + members[2]*members[2]);
    members[0] *= k; members[1] *= k; members[2] *= k;
}

__host__ __device__ inline Vector3 operator+(const Vector3 &v1, const Vector3 &v2) {
    return Vector3(v1.members[0] + v2.members[0], v1.members[1] + v2.members[1], v1.members[2] + v2.members[2]);
}

__host__ __device__ inline Vector3 operator-(const Vector3 &v1, const Vector3 &v2) {
    return Vector3(v1.members[0] - v2.members[0], v1.members[1] - v2.members[1], v1.members[2] - v2.members[2]);
}

__host__ __device__ inline Vector3 operator*(const Vector3 &v1, const Vector3 &v2) {
    return Vector3(v1.members[0] * v2.members[0], v1.members[1] * v2.members[1], v1.members[2] * v2.members[2]);
}

__host__ __device__ inline Vector3 operator/(const Vector3 &v1, const Vector3 &v2) {
    return Vector3(v1.members[0] / v2.members[0], v1.members[1] / v2.members[1], v1.members[2] / v2.members[2]);
}

__host__ __device__ inline Vector3 operator*(float t, const Vector3 &v) {
    return Vector3(t*v.members[0], t*v.members[1], t*v.members[2]);
}

__host__ __device__ inline Vector3 operator/(Vector3 v, float t) {
    return Vector3(v.members[0]/t, v.members[1]/t, v.members[2]/t);
}

__host__ __device__ inline Vector3 operator*(const Vector3 &v, float t) {
    return Vector3(t*v.members[0], t*v.members[1], t*v.members[2]);
}

__host__ __device__ inline float dot(const Vector3 &v1, const Vector3 &v2) {
    return v1.members[0] *v2.members[0] + v1.members[1] *v2.members[1]  + v1.members[2] *v2.members[2];
}

__host__ __device__ inline Vector3 cross(const Vector3 &v1, const Vector3 &v2) {
    return Vector3( (v1.members[1]*v2.members[2] - v1.members[2]*v2.members[1]),
                (-(v1.members[0]*v2.members[2] - v1.members[2]*v2.members[0])),
                (v1.members[0]*v2.members[1] - v1.members[1]*v2.members[0]));
}


__host__ __device__ inline Vector3& Vector3::operator+=(const Vector3 &v){
    members[0]  += v.members[0];
    members[1]  += v.members[1];
    members[2]  += v.members[2];
    return *this;
}

__host__ __device__ inline Vector3& Vector3::operator*=(const Vector3 &v){
    members[0]  *= v.members[0];
    members[1]  *= v.members[1];
    members[2]  *= v.members[2];
    return *this;
}

__host__ __device__ inline Vector3& Vector3::operator/=(const Vector3 &v){
    members[0]  /= v.members[0];
    members[1]  /= v.members[1];
    members[2]  /= v.members[2];
    return *this;
}

__host__ __device__ inline Vector3& Vector3::operator-=(const Vector3& v) {
    members[0]  -= v.members[0];
    members[1]  -= v.members[1];
    members[2]  -= v.members[2];
    return *this;
}

__host__ __device__ inline Vector3& Vector3::operator*=(const float t) {
    members[0]  *= t;
    members[1]  *= t;
    members[2]  *= t;
    return *this;
}

__host__ __device__ inline Vector3& Vector3::operator/=(const float t) {
    float k = 1.0/t;

    members[0]  *= k;
    members[1]  *= k;
    members[2]  *= k;
    return *this;
}

__host__ __device__ inline Vector3 unit_vector(Vector3 v) {
    return v / v.length();
}

#endif