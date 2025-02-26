#include <cmath>
#include <array>
#include <vector>
#include <tuple>
#include <algorithm>
#include <functional>
#include <random>
#include <iostream>
#include <float.h>
#include <chrono>
#include <omp.h>

#ifdef _MSC_VER
#pragma warning(disable: 4244)
#endif

constexpr uint32_t SCREEN_WIDTH  = 4 * 960;
constexpr uint32_t SCREEN_HEIGHT = 4 * 540;
constexpr float PI = 3.14159f;

struct Vec4;

struct Vec3 {
    float x, y, z;

    Vec3() : x(0), y(0), z(0) {}
    Vec3(float x) : x(x), y(x), z(x) {}
    Vec3(float x, float y, float z) : x(x), y(y), z(z) {}
    explicit Vec3(const Vec4& v);

    Vec3 operator+(const Vec3& v) const { return {x + v.x, y + v.y, z + v.z}; }
    Vec3 operator-(const Vec3& v) const { return {x - v.x, y - v.y, z - v.z}; }
    Vec3 operator*(const Vec3& v) const { return {x * v.x, y * v.y, z * v.z}; }
    Vec3 operator*(float scalar) const { return {x * scalar, y * scalar, z * scalar}; }
    Vec3 operator/(float scalar) const { return *this * (1 / scalar); }
    Vec3& operator+=(const Vec3& v) {
        x += v.x;
        y += v.y;
        z += v.z;
        return *this;
    }
    Vec3& operator-=(const Vec3& v) {
        x -= v.x;
        y -= v.y;
        z -= v.z;
        return *this;
    }
    Vec3& operator*=(const Vec3& v) {
        x *= v.x;
        y *= v.y;
        z *= v.z;
        return *this;
    }
    Vec3& operator/=(const Vec3& v) {
        x /= v.x;
        y /= v.y;
        z /= v.z;
        return *this;
    }
    bool operator==(const Vec3& v) const { return x == v.x && y == v.y && z == v.z; }

    float dot(const Vec3& v) const { return x * v.x + y * v.y + z * v.z; }
    Vec3 cross(const Vec3& v) const {   
        return {y * v.z - z * v.y, z * v.x - x * v.z, x * v.y - y * v.x}; 
    }   

    float length() const { return std::sqrt(x * x + y * y + z * z); }
    Vec3 normalize() const { float len = length(); return {x / len, y / len, z / len}; }
};

struct Vec4 {
    float x, y, z, w;

    Vec4() : x(0), y(0), z(0), w(0) {}
    Vec4(float x) : x(x), y(x), z(x), w(x) {}
    Vec4(float x, float y, float z, float w) : x(x), y(y), z(z), w(w) {}
    explicit Vec4(const Vec3& v, float w) : x(v.x), y(v.y), z(v.z), w(w) {}

    Vec4 operator+(const Vec4& v) const { return {x + v.x, y + v.y, z + v.z, w + v.w}; }
    Vec4 operator-(const Vec4& v) const { return {x - v.x, y - v.y, z - v.z, w - v.w}; }
    Vec4 operator*(float scalar) const { return {x * scalar, y * scalar, z * scalar, w * scalar}; }
    Vec4 operator/(float scalar) const { return *this * (1 / scalar); }
    Vec4& operator+=(const Vec4& v) {
        x += v.x;
        y += v.y;
        z += v.z;
        w += v.w;
        return *this;
    }
    Vec4& operator-=(const Vec4& v) {
        x -= v.x;
        y -= v.y;
        z -= v.z;
        w -= v.w;
        return *this;
    }
    Vec4& operator*=(const Vec4& v) {
        x *= v.x;
        y *= v.y;
        z *= v.z;
        w *= v.w;
        return *this;
    }
    Vec4& operator/=(const Vec4& v) {
        x /= v.x;
        y /= v.y;
        z /= v.z;
        w /= v.w;
        return *this;
    }

    float dot(const Vec4& v) const { return x * v.x + y * v.y + z * v.z + w * v.w; }
};

struct Vec2 {
    float x, y;

    Vec2() : x(0), y(0) {}
    Vec2(float x, float y) : x(x), y(y) {}
    Vec2(const Vec3& v) : x(v.x), y(v.y) {}

    Vec2 operator+(const Vec2& v) const { return {x + v.x, y + v.y}; }
    Vec2 operator*(float scalar) const { return {x * scalar, y * scalar}; }
    Vec2 operator/(float scalar) const { return *this * (1 / scalar); }
};

Vec3::Vec3(const Vec4& v): x(v.x), y(v.y), z(v.z) {}

Vec3 operator+(float scalar, const Vec3& vec) {
    return {scalar + vec.x, scalar + vec.y, scalar + vec.z};
}

Vec3 operator-(float scalar, const Vec3& vec) {
    return {scalar - vec.x, scalar - vec.y, scalar - vec.z};
}

Vec3 operator*(float scalar, const Vec3& vec) {
    return {scalar * vec.x, scalar * vec.y, scalar * vec.z};
}

Vec4 operator*(float scalar, const Vec4& vec) {
    return {scalar * vec.x, scalar * vec.y, scalar * vec.z, scalar * vec.w};
}

struct Mat3;

struct Mat4 {
    float m[4][4];

    Mat4() {
        for (int i = 0; i < 4; ++i)
            for (int j = 0; j < 4; ++j)
                m[i][j] = (i == j) ? 1.0f : 0.0f;
    }

    Mat4(float a11, float a12, float a13, float a14,
         float a21, float a22, float a23, float a24,
         float a31, float a32, float a33, float a34,
         float a41, float a42, float a43, float a44) {
        m[0][0] = a11; m[0][1] = a12; m[0][2] = a13; m[0][3] = a14;
        m[1][0] = a21; m[1][1] = a22; m[1][2] = a23; m[1][3] = a24;
        m[2][0] = a31; m[2][1] = a32; m[2][2] = a33; m[2][3] = a34;
        m[3][0] = a41; m[3][1] = a42; m[3][2] = a43; m[3][3] = a44;
    }

   Mat4 operator*(const Mat4& mat) const {
        Mat4 result;
        for (int i = 0; i < 4; ++i)
            for (int j = 0; j < 4; ++j)
                result.m[i][j] = m[i][0] * mat.m[0][j] + m[i][1] * mat.m[1][j] + 
                                 m[i][2] * mat.m[2][j] + m[i][3] * mat.m[3][j];
        return result;
    }

    Vec4 operator*(const Vec4& v) const {
        return {
            m[0][0] * v.x + m[0][1] * v.y + m[0][2] * v.z + m[0][3] * v.w,
            m[1][0] * v.x + m[1][1] * v.y + m[1][2] * v.z + m[1][3] * v.w,
            m[2][0] * v.x + m[2][1] * v.y + m[2][2] * v.z + m[2][3] * v.w,
            m[3][0] * v.x + m[3][1] * v.y + m[3][2] * v.z + m[3][3] * v.w
        };
    }

    float determinant() const {
        float det = 0;
        for (int i = 0; i < 4; ++i) {
            det += m[0][i] * cofactor(0, i);
        }
        return det;
    }

    float cofactor(int row, int col) const;
    Mat3 getMinorMatrix(int row, int col) const;

    bool inverse() {
        float det = determinant();
        if (det == 0) return false;

        Mat4 adjugate;
        for (int i = 0; i < 4; ++i) {
            for (int j = 0; j < 4; ++j) {
                adjugate.m[j][i] = cofactor(i, j);
            }
        }

        float invDet = 1.0f / det;
        for (int i = 0; i < 4; ++i) {
            for (int j = 0; j < 4; ++j) {
                m[i][j] = adjugate.m[i][j] * invDet;
            }
        }

        return true;
    }
};

struct Mat3 {
    float m[3][3];

    Mat3() {
        for (int i = 0; i < 3; ++i)
            for (int j = 0; j < 3; ++j)
                m[i][j] = (i == j) ? 1.0f : 0.0f;
    }

    Mat3(float a11, float a12, float a13,
         float a21, float a22, float a23,
         float a31, float a32, float a33) {
        m[0][0] = a11; m[0][1] = a12; m[0][2] = a13;
        m[1][0] = a21; m[1][1] = a22; m[1][2] = a23;
        m[2][0] = a31; m[2][1] = a32; m[2][2] = a33;
    }

    Mat3(const Mat4& mat4) {
        m[0][0] = mat4.m[0][0];
        m[0][1] = mat4.m[0][1];
        m[0][2] = mat4.m[0][2];

        m[1][0] = mat4.m[1][0];
        m[1][1] = mat4.m[1][1];
        m[1][2] = mat4.m[1][2];

        m[2][0] = mat4.m[2][0];
        m[2][1] = mat4.m[2][1];
        m[2][2] = mat4.m[2][2];
    }

    Mat3 operator*(const Mat3& mat) const {
        Mat3 result;
        for (int i = 0; i < 3; ++i)
            for (int j = 0; j < 3; ++j)
                result.m[i][j] = m[i][0] * mat.m[0][j] + m[i][1] * mat.m[1][j] + m[i][2] * mat.m[2][j];
        return result;
    }

    Mat3 operator*(float scalar) const {
        Mat3 result;
        for (int i = 0; i < 3; ++i)
            for (int j = 0; j < 3; ++j)
                result.m[i][j] = m[i][j] * scalar;
        return result;
    }

    Vec3 operator*(const Vec3& v) const {
        return {
            m[0][0] * v.x + m[0][1] * v.y + m[0][2] * v.z,
            m[1][0] * v.x + m[1][1] * v.y + m[1][2] * v.z,
            m[2][0] * v.x + m[2][1] * v.y + m[2][2] * v.z
        };
    }

    void transpose() {
        for (int i = 0; i < 3; ++i)
            for (int j = 0; j < 3; ++j)
                std::swap(m[i][j], m[j][i]);
    }

    float determinant() const {
        return m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1])
             - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
             + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0]);
    }

    bool inverse() {
        float det = determinant();
        if (det == 0) return false;

        Mat3 adjugate;
        adjugate.m[0][0] = m[1][1] * m[2][2] - m[1][2] * m[2][1];
        adjugate.m[0][1] = -(m[1][0] * m[2][2] - m[1][2] * m[2][0]);
        adjugate.m[0][2] = m[1][0] * m[2][1] - m[1][1] * m[2][0];

        adjugate.m[1][0] = -(m[0][1] * m[2][2] - m[0][2] * m[2][1]);
        adjugate.m[1][1] = m[0][0] * m[2][2] - m[0][2] * m[2][0];
        adjugate.m[1][2] = -(m[0][0] * m[2][1] - m[0][1] * m[2][0]);

        adjugate.m[2][0] = m[0][1] * m[1][2] - m[0][2] * m[1][1];
        adjugate.m[2][1] = -(m[0][0] * m[1][2] - m[0][2] * m[1][0]);
        adjugate.m[2][2] = m[0][0] * m[1][1] - m[0][1] * m[1][0];

        float invDet = 1.0f / det;
        for (int i = 0; i < 3; ++i)
            for (int j = 0; j < 3; ++j)
                m[i][j] = adjugate.m[i][j] * invDet;

        return true;
    }
};

float Mat4::cofactor(int row, int col) const {
    Mat3 minorMatrix = getMinorMatrix(row, col);
    return ((row + col) % 2 == 0 ? 1 : -1) * minorMatrix.determinant();
}

Mat3 Mat4::getMinorMatrix(int row, int col) const {
    Mat3 minor;
    int minorRow = 0;
    for (int i = 0; i < 4; ++i) {
        if (i == row) continue;
        int minorCol = 0;
        for (int j = 0; j < 4; ++j) {
            if (j == col) continue;
            minor.m[minorRow][minorCol] = m[i][j];
            minorCol++;
        }
        minorRow++;
    }
    return minor;
}

template <typename T>
T lerp(T a, T b, float t) {
    return a + t * (b - a);
}

float clamp(float x, float min, float max) {
    return (x < min) ? min : (x > max) ? max : x;
}

Vec3 clamp(Vec3 v, Vec3 min, Vec3 max) {
    float x = clamp(v.x, min.x, max.x);
    float y = clamp(v.y, min.y, max.y);
    float z = clamp(v.z, min.z, max.z);
    return { x, y, z };
}

float saturate(float x) {
    return clamp(x, 0.0f, 1.0f);
}

float pow5(float v) {
    return v * v * v * v * v;
}

float smoothstep(float edge0, float edge1, float x) {
    float t = (x - edge0) / (edge1 - edge0);   
    t = clamp(t, 0.0f, 1.0f);  
    return t * t * (3.0f - 2.0f * t);
}

struct Vertex {
    Vec3 position;
    Vec3 normal;
    Vec2 uv;
};

std::vector<float> cube_vertices = {
    // 0~3: position, 4~6: normal, 7~8: UV
    // Front face
    -1.0f, -1.0f, -1.0f,  0.0f, 0.0f, -1.0f,  0.0f, 0.0f,
     1.0f, -1.0f, -1.0f,  0.0f, 0.0f, -1.0f,  1.0f, 0.0f,
     1.0f,  1.0f, -1.0f,  0.0f, 0.0f, -1.0f,  1.0f, 1.0f,
    -1.0f,  1.0f, -1.0f,  0.0f, 0.0f, -1.0f,  0.0f, 1.0f,

    // Back face
    -1.0f, -1.0f,  1.0f,  0.0f, 0.0f,  1.0f,  0.0f, 0.0f,
     1.0f, -1.0f,  1.0f,  0.0f, 0.0f,  1.0f,  1.0f, 0.0f,
     1.0f,  1.0f,  1.0f,  0.0f, 0.0f,  1.0f,  1.0f, 1.0f,
    -1.0f,  1.0f,  1.0f,  0.0f, 0.0f,  1.0f,  0.0f, 1.0f,

    // Bottom face
    -1.0f, -1.0f, -1.0f,  0.0f, -1.0f, 0.0f,  0.0f, 0.0f,
     1.0f, -1.0f, -1.0f,  0.0f, -1.0f, 0.0f,  1.0f, 0.0f,
     1.0f, -1.0f,  1.0f,  0.0f, -1.0f, 0.0f,  1.0f, 1.0f,
    -1.0f, -1.0f,  1.0f,  0.0f, -1.0f, 0.0f,  0.0f, 1.0f,

    // Top face
    -1.0f,  1.0f, -1.0f,  0.0f,  1.0f, 0.0f,  0.0f, 0.0f,
     1.0f,  1.0f, -1.0f,  0.0f,  1.0f, 0.0f,  1.0f, 0.0f,
     1.0f,  1.0f,  1.0f,  0.0f,  1.0f, 0.0f,  1.0f, 1.0f,
    -1.0f,  1.0f,  1.0f,  0.0f,  1.0f, 0.0f,  0.0f, 1.0f,

    // Left face
    -1.0f, -1.0f, -1.0f, -1.0f, 0.0f, 0.0f,  0.0f, 0.0f,
    -1.0f, -1.0f,  1.0f, -1.0f, 0.0f, 0.0f,  1.0f, 0.0f,
    -1.0f,  1.0f,  1.0f, -1.0f, 0.0f, 0.0f,  1.0f, 1.0f,
    -1.0f,  1.0f, -1.0f, -1.0f, 0.0f, 0.0f,  0.0f, 1.0f,

    // Right face
     1.0f, -1.0f, -1.0f,  1.0f, 0.0f, 0.0f,  0.0f, 0.0f,
     1.0f, -1.0f,  1.0f,  1.0f, 0.0f, 0.0f,  1.0f, 0.0f,
     1.0f,  1.0f,  1.0f,  1.0f, 0.0f, 0.0f,  1.0f, 1.0f,
     1.0f,  1.0f, -1.0f,  1.0f, 0.0f, 0.0f,  0.0f, 1.0f
};


std::vector<uint32_t> cube_indices = {
    0, 2, 1, 0, 3, 2,
    4, 5, 6, 4, 6, 7,
    8, 9, 10, 8, 10, 11,
    12, 14, 13, 12, 15, 14,
    16, 17, 18, 16, 18, 19,
    20, 22, 21, 20, 23, 22
};

std::vector<float> quad_vertices = {
    -1.0f, -1.0f, 0.0f, 0.0f, 0.0f,
    1.0f, -1.0f, 0.0f, 1.0f, 0.0f,
    1.0f, 1.0f, 0.0f, 1.0f, 1.0f,
    -1.0f, 1.0f, 0.0f, 0.0f, 1.0f
};

std::vector<uint32_t> quad_indices = {
    0, 2, 1, 0, 3, 2
};

void generateSphere(float radius, int segments, int rings, std::vector<float>& vertices, 
                    std::vector<uint32_t>& indices) {   
    vertices.clear();
    indices.clear();

    for (int ring = 0; ring <= rings; ++ring) {
        float theta = PI * ring / rings;
        float v = static_cast<float>(ring) / rings;
        for (int segment = 0; segment <= segments; ++segment) {
            float phi = 2 * PI * segment / segments;
            float u = static_cast<float>(segment) / segments;

            float x = radius * sin(theta) * cos(phi);
            float y = radius * sin(theta) * sin(phi);
            float z = radius * cos(theta);
            float nx = x / radius;
            float ny = y / radius;
            float nz = z / radius;

            vertices.push_back(x);
            vertices.push_back(y);
            vertices.push_back(z);
            vertices.push_back(nx);
            vertices.push_back(ny);
            vertices.push_back(nz);
            vertices.push_back(u);
            vertices.push_back(v);
        }
    }

    for (int ring = 0; ring < rings; ++ring) {
        for (int segment = 0; segment < segments; ++segment) {
            int current = ring * (segments + 1) + segment;
            int next_segment = current + 1;
            int next_ring = current + (segments + 1);

            indices.push_back(current);
            indices.push_back(next_ring);
            indices.push_back(next_segment);

            indices.push_back(next_segment);
            indices.push_back(next_ring);
            indices.push_back(next_ring + 1);
        }
    }
}

Mat4 createTranslateMatrix(const Vec3& translation) {
    Mat4 result;
    result.m[0][3] = translation.x;
    result.m[1][3] = translation.y;
    result.m[2][3] = translation.z;
    return result;
}

Mat4 createRotateMatrix(float angle, Vec3 axis) {
    Mat4 result;
    float rad = angle * PI / 180.0f;
    axis = axis.normalize();
    float c = cos(rad);
    float s = sin(rad);
    float omc = 1.0f - c; // 1 - cos(angle)

    result.m[0][0] = axis.x * axis.x * omc + c;
    result.m[0][1] = axis.x * axis.y * omc - axis.z * s; 
    result.m[0][2] = axis.x * axis.z * omc + axis.y * s;
    result.m[1][0] = axis.y * axis.x * omc + axis.z * s;
    result.m[1][1] = axis.y * axis.y * omc + c;
    result.m[1][2] = axis.y * axis.z * omc - axis.x * s; 
    result.m[2][0] = axis.z * axis.x * omc - axis.y * s;
    result.m[2][1] = axis.z * axis.y * omc + axis.x * s;
    result.m[2][2] = axis.z * axis.z * omc + c;
    result.m[3][3] = 1.0f;

    return result;
}

Mat4 createScaleMatrix(const Vec3& scale) {
    Mat4 result;
    result.m[0][0] = scale.x;
    result.m[1][1] = scale.y;
    result.m[2][2] = scale.z;
    return result;
}

Mat4 createModelMatrix(const Vec3& translation = Vec3(0, 0, 0), const Vec3& rotationAxis = Vec3(0, 1, 0), float rotationAngle = 0, const Vec3& scale = Vec3(1, 1, 1)) {
    return createTranslateMatrix(translation) * createRotateMatrix(rotationAngle, rotationAxis) * createScaleMatrix(scale);
}

Mat4 createLookAtMatrix(const Vec3& eyePosition, const Vec3& focusPosition, const Vec3& upDirection = Vec3(0, 1, 0)) {
    Vec3 z_axis = (focusPosition - eyePosition).normalize();
    Vec3 x_axis = upDirection.cross(z_axis);
    if (x_axis.length() < 1e-6f) {
        Vec3 alternativeUp = Vec3(0, 1, 0.1f).normalize();
        x_axis = alternativeUp.cross(z_axis).normalize();
    }
    else {
        x_axis = x_axis.normalize();
    }
    Vec3 y_axis = z_axis.cross(x_axis);
    return Mat4{x_axis.x, x_axis.y, x_axis.z, -eyePosition.dot(x_axis),
                y_axis.x, y_axis.y, y_axis.z, -eyePosition.dot(y_axis),
                z_axis.x, z_axis.y, z_axis.z, -eyePosition.dot(z_axis),
                0, 0, 0, 1};
}


Mat4 createPerspectiveMatrix(const float fovY, const float aspectRatio, const float near, const float far) {
    Mat4 res;
    float fovYR = fovY * PI / 180.0f;
    res.m[0][0] = 1 / (aspectRatio * tan(fovYR / 2));
    res.m[1][1] = 1 / tan(fovYR / 2);
    res.m[2][2] = far / (far - near);
    res.m[2][3] = -near * far / (far - near);
    res.m[3][2] = 1;
    res.m[3][3] = 0;
    return res;
}

Mat4 createViewportMatrix(uint32_t width, uint32_t height) {
    Mat4 res;
    res.m[0][0] = (float)width / 2;
    res.m[0][3] = (float)width / 2;
    res.m[1][1] = (float)height / 2;
    res.m[1][3] = (float)height / 2;
    return res;
}

Vec3 computeNormal(const Vec3& v0, const Vec3& v1, const Vec3& v2) {
    return (v1 - v0).cross(v2 - v0).normalize();
}

bool isBackFace(const Vec3& cameraPosition, const Vec3& v0, const Vec3& v1, const Vec3& v2) {
    Vec3 normal = computeNormal(v0, v1, v2);
    Vec3 viewDir = (cameraPosition - (v0 + v1 + v2) / 3).normalize();
    return normal.dot(viewDir) <= 0.0f;
}

void backFaceCulling(const std::vector<Vertex>& vertex_buffer, std::vector<uint32_t>& indices, const Vec3& cameraPosition) {
    std::vector<uint32_t> culledIndices;
    for (uint32_t i = 0; i < indices.size(); i += 3) {
        uint32_t idx0 = indices[i];
        uint32_t idx1 = indices[i + 1];
        uint32_t idx2 = indices[i + 2];

        Vec3 pos0 = vertex_buffer[idx0].position;
        Vec3 pos1 = vertex_buffer[idx1].position;
        Vec3 pos2 = vertex_buffer[idx2].position;

        if (!isBackFace(cameraPosition, pos0, pos1, pos2)) {
            culledIndices.push_back(idx0);
            culledIndices.push_back(idx1);
            culledIndices.push_back(idx2);
        }
    }
    indices = std::move(culledIndices);
}

void computeBarycentricCoords(const Vec2& v0, const Vec2& v1, const Vec2& v2, const Vec2& p, float& lambda1, float& lambda2, float& lambda3) {
    float denominator = (v1.y - v2.y) * (v0.x - v2.x) + (v2.x - v1.x) * (v0.y - v2.y);
    lambda1 = ((v1.y - v2.y) * (p.x - v2.x) + (v2.x - v1.x) * (p.y - v2.y)) / denominator;
    lambda2 = ((v2.y - v0.y) * (p.x - v2.x) + (v0.x - v2.x) * (p.y - v2.y)) / denominator;
    lambda3 = 1.0f - lambda1 - lambda2;
}

template <typename T>
T perspectiveCorrectInterpolate(const T& v0, const T& v1, const T& v2, float lambda1, float lambda2, float lambda3, 
        float z0, float z1, float z2) {
    float w0 = lambda1 / z0;
    float w1 = lambda2 / z1;
    float w2 = lambda3 / z2;
    float w_sum = w0 + w1 + w2;

    return (v0 * w0 + v1 * w1 + v2 * w2) / w_sum;
}

struct ColorBuffer {
    ColorBuffer(uint32_t width, uint32_t height, const Vec4& colorDefault = Vec4(0.0f)) : width(width), height(height) {
        m_colorBuffer.resize(height, std::vector<Vec4>(width, colorDefault));
    }

    Vec4& getColor(uint32_t x, uint32_t y) {
        return m_colorBuffer[height - y - 1][x];
    }

    void resize(uint32_t width, uint32_t height, const Vec4& colorDefault = Vec4(0.0f)) {
        this->width = width;
        this->height = height;
        m_colorBuffer.clear();
        m_colorBuffer.resize(height, std::vector<Vec4>(width, colorDefault));
    }

public:
    uint32_t width;
    uint32_t height;
private:
    std::vector<std::vector<Vec4>> m_colorBuffer;
};

struct DepthBuffer {
    DepthBuffer(uint32_t width, uint32_t height, float depthDefault = 1.0f) : width(width), height(height) {
        m_depthBuffer.resize(height, std::vector<float>(width, depthDefault));
    }

    float& getDepth(uint32_t x, uint32_t y) {
        return m_depthBuffer[height - y - 1][x];
    }
public:
    uint32_t width;
    uint32_t height;
private:
    std::vector<std::vector<float>> m_depthBuffer;
};

struct FrameBuffer {
    FrameBuffer(uint32_t width, uint32_t height, const Vec4& colorDefault = Vec4(0), float depthDefault = 1.0f) 
        : width(width), height(height), 
          colorBuffers{ ColorBuffer(width, height, colorDefault), ColorBuffer(width, height, colorDefault), 
                        ColorBuffer(width, height, colorDefault), ColorBuffer(width, height, colorDefault) },
          depthBuffer(width, height, depthDefault) {}

    Vec4& getColori(uint32_t i, uint32_t x, uint32_t y) {
        return colorBuffers[i].getColor(x, y);
    }

    float& getDepth(uint32_t x, uint32_t y) {
        return depthBuffer.getDepth(x, y);
    }
public:
    uint32_t width;
    uint32_t height;
    ColorBuffer colorBuffers[4];
    DepthBuffer depthBuffer;
};

void generateCheckerboradTexture(ColorBuffer& texture, uint32_t checkSize) {
    if (texture.width != texture.height) {
        std::clog << "texture's width and height are not equal.\n";
        return;
    }

    for (uint32_t y = 0; y < texture.height; y++) {
        for (uint32_t x = 0; x < texture.width; x++) {
            uint32_t checkX = x / checkSize;
            uint32_t checkY = y / checkSize;
            bool isWhite = (checkX + checkY) % 2 == 0;
            texture.getColor(x, y) = isWhite ? Vec4(1) : Vec4(0);
        }
    }
}

void generateStarTexture(ColorBuffer& texture, uint32_t starSize) {
    if (texture.width != texture.height) {
        std::clog << "texture's width and height are not equal.\n";
        return;
    }

    float centerX = texture.width / 2.0f;
    float centerY = texture.height / 2.0f;
    float outerRadius = starSize / 2.0f;
    float innerRadius = outerRadius * 0.5f;

    const int numPoints = 10;
    std::vector<std::pair<float, float>> points(numPoints);

    for (int i = 0; i < numPoints; ++i) {
        float angle = i * (2 * PI / numPoints) - PI / 2;
        float radius = (i % 2 == 0) ? outerRadius : innerRadius;
        points[i] = { centerX + radius * std::cos(angle), centerY + radius * std::sin(angle) };
    }

    for (uint32_t y = 0; y < texture.height; y++) {
        for (uint32_t x = 0; x < texture.width; x++) {
            bool inside = false;
            for (size_t i = 0, j = points.size() - 1; i < points.size(); j = i++) {
                auto [xi, yi] = points[i];
                auto [xj, yj] = points[j];

                bool intersect = ((yi > y) != (yj > y)) &&
                    (x < (xj - xi) * (y - yi) / (yj - yi) + xi);
                if (intersect) {
                    inside = !inside;
                }
            }

            texture.getColor(x, y) = inside ? Vec4(0) : Vec4(1);
        }
    }
}

void generateCircleTexture(ColorBuffer& texture, uint32_t radius) {
    if (texture.width != texture.height) {
        std::clog << "texture's width and height are not equal.\n";
        return;
    }

    float centerX = texture.width / 2.0f;
    float centerY = texture.height / 2.0f;

    for (uint32_t y = 0; y < texture.height; y++) {
        for (uint32_t x = 0; x < texture.width; x++) {
            float dx = x - centerX;
            float dy = y - centerY;
            float distanceSquared = dx * dx + dy * dy;

            bool inside = distanceSquared <= radius * radius;

            texture.getColor(x, y) = inside ? Vec4(0) : Vec4(1);
        }
    }
}

// 简化的伪随机噪声函数（可以用更复杂的噪声库替换）
float randomNoise(float x, float y) {
    // 改进的伪随机噪声生成，归一化到 [0, 1]
    float value = std::sin(x * 12.9898f + y * 78.233f) * 43758.5453f;
    return value - std::floor(value); // 结果在 [0, 1) 内
}

void generateWoodTexture(ColorBuffer& texture, float ringSpacing, float noiseStrength, float directionX, float directionY) {
    float centerX = texture.width / 2.0f;
    float centerY = texture.height / 2.0f;

    for (uint32_t y = 0; y < texture.height; y++) {
        for (uint32_t x = 0; x < texture.width; x++) {
            float dx = x - centerX;
            float dy = y - centerY;

            float direction = dx * directionX + dy * directionY;
            float grain = std::sin(direction / ringSpacing * 2.0f * PI);

            float noise1 = randomNoise(x * 0.05f, y * 0.05f) * noiseStrength;
            float noise2 = randomNoise(x * 0.15f, y * 0.15f) * noiseStrength * 0.5f;
            float noise3 = randomNoise(x * 0.25f, y * 0.25f) * noiseStrength * 0.25f;

            float totalNoise = noise1 + noise2 + noise3;

            float intensity = (grain + totalNoise) * 0.5f + 0.5f;

            Vec3 woodColor = clamp(lerp(Vec3(0.7f, 0.5f, 0.3f), Vec3(0.5f, 0.3f, 0.1f), intensity), Vec3(0.0f), Vec3(1.0f));
            
            texture.getColor(x, y) = Vec4(woodColor, 1.0f);
        }
    }
}

void generateWaveNormalMap(ColorBuffer& texture, float waveScale, float amplitude) {
    if (texture.width != texture.height) {
        std::clog << "texture's width and height are not equal.\n";
        return;
    }

    for (uint32_t y = 0; y < texture.height; y++) {
        for (uint32_t x = 0; x < texture.width; x++) {
            float u = static_cast<float>(x) / texture.width;
            float v = static_cast<float>(y) / texture.height;

            float wave1 = sin((u + v) * waveScale) * amplitude;
            float wave2 = cos((u - v) * waveScale) * amplitude;

            float dx = wave1;
            float dy = wave2;
            float dz = -1.0f;

            Vec3 normal = Vec3(dx, dy, dz).normalize();

            Vec3 encodedNormal = (normal + Vec3(1.0f)) * 0.5f;

            texture.getColor(x, y) = Vec4(encodedNormal, 1.0f);
        }
    }
}


void gaussianFilter(ColorBuffer& buffer) {
    float weight[] = { 0.4026f, 0.2442f, 0.0545f };

    ColorBuffer new_buffer(buffer.width + 4, buffer.height + 4);
    for (uint32_t y = 2; y < buffer.height + 2; y++)
        for (uint32_t x = 2; x < buffer.width + 2; x++)
            new_buffer.getColor(x, y) = buffer.getColor(x - 2, y - 2);
    
    // X-axis
    for (uint32_t y = 2; y < buffer.height + 2; y++) {
        for (uint32_t x = 2; x < buffer.width + 2; x++) {
            Vec3 sum = Vec3(new_buffer.getColor(x, y) * weight[0]);
            for (uint32_t i = 1; i < 3; i++) {
                sum += Vec3(new_buffer.getColor(x + i, y) * weight[i]);
                sum += Vec3(new_buffer.getColor(x - i, y) * weight[i]);
            }
            new_buffer.getColor(x, y) = Vec4(sum, 1);
        }
    }

    // Y-axis
    for (uint32_t y = 2; y < buffer.height + 2; y++) {
        for (uint32_t x = 2; x < buffer.width + 2; x++) {
            Vec3 sum = Vec3(new_buffer.getColor(x, y) * weight[0]);
            for (uint32_t i = 1; i < 3; i++) {
                sum += Vec3(new_buffer.getColor(x, y + i) * weight[i]);
                sum += Vec3(new_buffer.getColor(x, y - i) * weight[i]);
            }
            new_buffer.getColor(x, y) = Vec4(sum, 1);
        }
    }

    for (uint32_t y = 2; y < buffer.height + 2; y++)
        for (uint32_t x = 2; x < buffer.width + 2; x++)
            buffer.getColor(x - 2, y - 2) = new_buffer.getColor(x, y);
}

void gaussianFilterNTimes(ColorBuffer& buffer, uint32_t N) {
    for (uint32_t i = 0; i < N; i++)
        gaussianFilter(buffer);
}

// only support downscaling
void nearestScaling(ColorBuffer& buffer, float scale_width, float scale_height) {
    if (scale_width > 1 || scale_height > 1) return;
    uint32_t src_width = buffer.width;
    uint32_t src_height = buffer.height;
    uint32_t dst_width = static_cast<uint32_t>(src_width * scale_width);
    uint32_t dst_height = static_cast<uint32_t>(src_height * scale_height);

    ColorBuffer new_buffer(dst_width, dst_height);
    for (uint32_t y = 0; y < dst_height; y++) {
        for (uint32_t x = 0; x < dst_width; x++) {
            uint32_t src_x = clamp(x / scale_width, 0, src_width - 1);
            uint32_t src_y = clamp(y / scale_height, 0, src_height - 1);
            new_buffer.getColor(x, y) = buffer.getColor(src_x, src_y);
        }
    }

    buffer.resize(dst_width, dst_height);
    for (uint32_t y = 0; y < dst_height; y++)
        for (uint32_t x = 0; x < dst_width; x++)
            buffer.getColor(x, y) = new_buffer.getColor(x, y);
}

void displayImage(ColorBuffer& buffer) {
    std::cout << "P3\n" << buffer.width << ' ' << buffer.height << "\n255\n";
    for (uint32_t y = 0; y < buffer.height; y++) {
        std::clog << "\rScanlines remaining: " << buffer.height - y << ' ' << std::flush;
        for (uint32_t x = 0; x < buffer.width; x++) {
            std::cout << static_cast<int>(std::clamp(buffer.getColor(x, buffer.height - y - 1).x * 255, 0.0f, 255.0f)) << ' '
                << static_cast<int>(std::clamp(buffer.getColor(x, buffer.height - y - 1).y * 255, 0.0f, 255.0f)) << ' '
                << static_cast<int>(std::clamp(buffer.getColor(x, buffer.height - y - 1).z * 255, 0.0f, 255.0f)) << '\n';
        }
    }
    std::clog << "\rDone.                 \n";
}

template <typename... Attributes>
struct GenericV2F {
    using AttributeTuple = std::tuple<Attributes...>;

    Vec4 gl_Position;
    Vec2 gl_FragCoord;
    float gl_ZCamera;
    float gl_ZDepth;
    AttributeTuple attributes;
};

template <typename Tuple, typename Func, std::size_t... Is>
void forEachInTupleImpl(Tuple& tuple, Func func, std::index_sequence<Is...>) {
    (func(std::get<Is>(tuple), std::integral_constant<std::size_t, Is>{}), ...);
}

template <typename Tuple, typename Func>
void forEachInTuple(Tuple& tuple, Func func) {
    constexpr std::size_t tupleSize = std::tuple_size_v<std::remove_reference_t<Tuple>>;
    forEachInTupleImpl(tuple, func, std::make_index_sequence<tupleSize>{});
}

struct PointLight {
    float intensity;
    float radius;
    Vec3 color;
    Vec3 position;
};

Vec3 gammaToLinear(Vec3 gammaValue, float gamma = 2.2f) {
    return { powf(gammaValue.x, gamma), powf(gammaValue.y, gamma), powf(gammaValue.z, gamma) };
}

Vec3 linearToGamma(Vec3 linearValue, float gamma = 2.2f) {
    return gammaToLinear(linearValue, 1 / gamma);
}

struct DepthMapMaterial {
    // 0: position, 1: normal (both in world space), 2: uv
    using V2F = GenericV2F<Vec3, Vec3, Vec2>;

    V2F vert(const Vertex& i) {
        V2F o;
        // Vec3 position
        std::get<0>(o.attributes) = Vec3(model * Vec4(i.position, 1));

        // Vec3 normal
        Mat3 normal2World(model);
        normal2World.transpose();
        normal2World.inverse();
        std::get<1>(o.attributes) = normal2World * i.normal;    

        // Vec2 uv
        std::get<2>(o.attributes) = i.uv; 

        o.gl_Position = lightSpaceMatrix * model * Vec4(i.position, 1);
        o.gl_ZCamera = o.gl_Position.w;
        return o;
    }

    std::array<Vec4, 4> frag(const V2F& fs_in) {
        Vec3 fragPos = std::get<0>(fs_in.attributes);
        Vec3 normal = std::get<1>(fs_in.attributes);

        Vec3 L = (light.position - fragPos).normalize();
        float NdotL = std::max(normal.dot(L), 0.0f);
        //Vec3 incidentFlux = light.intensity * light.color * NdotL / pow((fragPos - light.position).length(), 2);
        Vec3 incidentFlux = light.intensity * light.color * NdotL;
        if (useAlbedoMap) {
            Vec2 uv = std::get<2>(fs_in.attributes);
            uint32_t coord_u = clamp(uv.x * albedoMap->width, 0.0f, albedoMap->width - 1);
            uint32_t coord_v = clamp(uv.y * albedoMap->height, 0.0f, albedoMap->height - 1);
            albedo = Vec3(albedoMap->getColor(coord_u, coord_v));
        }
        Vec3 flux = incidentFlux * albedo / PI;

        Vec4 target0(fragPos, 1);
        Vec4 target1(normal, 0);
        Vec4 target2(flux, 1);
        return {target0, target1, target2, Vec4()};
    }
public:
    Mat4 lightSpaceMatrix;
    Mat4 model;
    Vec3 albedo;
    bool useAlbedoMap = false;
    ColorBuffer* albedoMap;
    PointLight light;
};

struct GBufferMaterial {
    // 0: position, 1: normal (both in view space), 2: uv
    using V2F = GenericV2F<Vec3, Vec3, Vec2>;

    V2F vert(const Vertex& i) {
        V2F o;

        Mat4 MVP = projection * view * model;
        o.gl_Position = MVP * Vec4(i.position, 1);
        o.gl_ZCamera = o.gl_Position.w;

        // Vec3 position
        std::get<0>(o.attributes) = Vec3(view * model * Vec4(i.position, 1));

        // Vec3 normal
        normal2View = view * model;
        normal2View.transpose();
        normal2View.inverse();
        std::get<1>(o.attributes) = normal2View * i.normal;

        // Vec2 uv
        std::get<2>(o.attributes) = i.uv;

        return o;
    }

    std::array<Vec4, 4> frag(const V2F& fs_in) {    
        Vec3 viewSpacePosition = std::get<0>(fs_in.attributes);
        Vec3 viewSpaceNormal = std::get<1>(fs_in.attributes).normalize();
        Vec2 uv = std::get<2>(fs_in.attributes);

        if (useNormalMap) {
            uint32_t coord_u = clamp(uv.x * normalMap->width, 0.0f, normalMap->width - 1);
            uint32_t coord_v = clamp(uv.y * normalMap->height, 0.0f, normalMap->height - 1);
            Vec3 normal = Vec3(normalMap->getColor(coord_u, coord_v) * 2 - 1).normalize();
            viewSpaceNormal = (normal2View * normal).normalize();
        }
            
        Vec4 target0(uv.x, uv.y, objectID, smoothness);
        Vec4 target1(albedo, metallic);
        Vec4 target2(viewSpacePosition, 1);
        Vec4 target3(viewSpaceNormal, 0);
        return {target0, target1, target2, target3};
    }

public:
    Mat4 model;
    Mat4 view;
    Mat4 projection;
    float smoothness;
    float metallic;
    Vec3 emissive;
    Vec3 albedo;
    uint32_t objectID;
    bool useNormalMap = false;
    ColorBuffer* normalMap;
private:
    Mat3 normal2View;
};

struct SSAOMaterial {
    using V2F = GenericV2F<>;

    SSAOMaterial() : randomFloats(std::uniform_real_distribution<float>(0.0f, 1.0f)) {
        generateSampleKernel();
        generateNoise();
    }

    V2F vert(const Vertex& i) {
        V2F o;
        o.gl_Position = Vec4(i.position, 1);
        o.gl_ZCamera = 1;
        return o;
    }

    std::array<Vec4, 4> frag(const V2F& fs_in) {
        uint32_t kernelSize = 64;
        float radius = 0.5f;
        float bias = 0.0f;

        uint32_t x = fs_in.gl_FragCoord.x;
        uint32_t y = fs_in.gl_FragCoord.y;

        Vec3 fragPos = Vec3(gbuffer->getColori(2, x, y));
        Vec3 normal = Vec3(gbuffer->getColori(3, x, y));
        Vec3 randomVec = ssaoNoise[y % 4][x % 4].normalize();
        Vec3 tangent = (randomVec - normal * randomVec.dot(normal)).normalize();
        Vec3 bitangent = normal.cross(tangent);
        Mat3 TBN(tangent.x, bitangent.x, normal.x,
                 tangent.y, bitangent.y, normal.y,
                 tangent.z, bitangent.z, normal.z);

        float occlusion = 0.0f;
        for (uint32_t i = 0; i < kernelSize; i++) {
            Vec3 samplePos = TBN * ssaoKernel[i];
            samplePos = fragPos + samplePos * radius;

            Vec4 offset = projection * Vec4(samplePos, 1.0f);
            if (offset.w == 0)
                continue;
            Vec4 ndc = (offset / offset.w) * 0.5f + 0.5f; // only x and y are valid
            float z = (offset / offset.w).z;

            if (z < 0.0f || z > 1.0f) {
                occlusion += 0.0f;
                continue;
            }

            uint32_t u = clamp(ndc.x * gbuffer->width, 0.0f, static_cast<float>(gbuffer->width - 1));
            uint32_t v = clamp(ndc.y * gbuffer->height, 0.0f, static_cast<float>(gbuffer->height - 1));
            float sampleDepth = gbuffer->getColori(2, u, v).z;
            float rangeCheck = smoothstep(0.0f, 1.0f, radius / (abs(fragPos.z - sampleDepth) + 1e-7));
            occlusion += (sampleDepth <= samplePos.z + bias ? 1.0f : 0.0f) * rangeCheck;
        }
        occlusion = 1 - occlusion / kernelSize;
        return {Vec4(occlusion), Vec4(), Vec4(), Vec4()};           
    }

public:
    FrameBuffer* gbuffer;
    Mat4 view;
    Mat4 projection;  

    Mat4 model; // Identity matrix, for back-face culling, actually useless 
private:
    void generateSampleKernel() {
        for (uint32_t i = 0; i < 64; i++) {
            Vec3 sample(randomFloats(generator) * 2.0f - 1.0f, randomFloats(generator) * 2.0f - 1.0f, randomFloats(generator));
            sample = sample.normalize() * randomFloats(generator);
            float scale = float(i) / 64;
            scale = lerp(0.1f, 1.0f, scale * scale);
            sample *= scale;
            ssaoKernel.push_back(sample);
        }
    }

    void generateNoise() {
        ssaoNoise.resize(4, std::vector<Vec3>(4));
        for (uint32_t i = 0; i < 4; i++)
            for (uint32_t j = 0; j < 4; j++)
                ssaoNoise[i][j] = Vec3(randomFloats(generator) * 2.0f - 1.0f, randomFloats(generator) * 2.0f - 1.0f, 0.0f);             
    }

private:
    std::uniform_real_distribution<float> randomFloats;
    std::default_random_engine generator;
    std::vector<Vec3> ssaoKernel;
    std::vector<std::vector<Vec3>> ssaoNoise;
};

struct SSAOBlurMaterial {
    using V2F = GenericV2F<>;

    V2F vert(const Vertex& i) {
        V2F o;
        o.gl_Position = Vec4(i.position, 1);
        o.gl_ZCamera = 1;
        return o;
    }

    std::array<Vec4, 4> frag(const V2F& fs_in) {
        uint32_t x = fs_in.gl_FragCoord.x;
        uint32_t y = fs_in.gl_FragCoord.y;

        float result = 0.0f;
        uint32_t u, v;
        for (int32_t i = -2; i < 2; i++) {
            for (int32_t j = -2; j < 2; j++) {
                u = clamp(x + i, 0, ssao->width - 1);
                v = clamp(y + j, 0, ssao->height - 1);
                result += ssao->getColori(0, u, v).x;
            }
        }
        return {Vec4(result / 16), Vec4(), Vec4(), Vec4()};
    }

public:
    Mat4 view;
    FrameBuffer* ssao;

    Mat4 model; // Identity matrix, for back-face culling, actually useless
};

struct StandardMaterial {
    using V2F = GenericV2F<>;

    StandardMaterial() : u(std::uniform_real_distribution<float>(0, 1)) {
        generateRSMSamples();
    }

    V2F vert(const Vertex& i) {
        V2F o;
        o.gl_Position = Vec4(i.position, 1);
        o.gl_ZCamera = 1;
        return o;
    }

    std::array<Vec4, 4> frag(const V2F& fs_in) {
        uint32_t x = fs_in.gl_FragCoord.x;
        uint32_t y = fs_in.gl_FragCoord.y;

        if (Vec3(gbuffer->getColori(3, x, y)) == Vec3(0))
            return { Vec4(), Vec4(), Vec4(), Vec4() };
        
        Vec3 viewPos = Vec3(gbuffer->getColori(2, x, y));
        Vec3 viewNormal = Vec3(gbuffer->getColori(3, x, y));
        Vec3 worldPos = Vec3(invView * Vec4(viewPos, 1));
        Vec3 V = (cameraPos - worldPos).normalize();
        Vec3 N = Vec3(invView * Vec4(viewNormal, 0)).normalize();

        float u = gbuffer->getColori(0, x, y).x;
        float v = gbuffer->getColori(0, x, y).y;
        uint32_t objectID = gbuffer->getColori(0, x, y).z;
        Vec3 albedo;
        if (objectID == 0) { // cube left
            uint32_t coord_u = clamp(u * star->width, 0.0f, star->width - 1);
            uint32_t coord_v = clamp(v * star->height, 0.0f, star->height - 1);
            albedo = Vec3(star->getColor(coord_u, coord_v));
        } else if (objectID == 1) { // cube right
            uint32_t coord_u = clamp(u * checkerborad->width, 0.0f, checkerborad->width - 1);
            uint32_t coord_v = clamp(v * checkerborad->height, 0.0f, checkerborad->height - 1);
            albedo = Vec3(checkerborad->getColor(coord_u, coord_v));
        } else if (objectID == 2) { // quad down
            uint32_t coord_u = clamp(u * wood->width, 0.0f, wood->width - 1);
            uint32_t coord_v = clamp(v * wood->height, 0.0f, wood->height - 1);
            albedo = Vec3(wood->getColor(coord_u, coord_v));        
        } else {
            albedo = Vec3(gbuffer->getColori(1, x, y));
        }
    
        float ao = ssao->getColori(0, x, y).x;
        Vec3 ambient = 0.2f * Vec3((float)54 / 255, (float)58 / 255, (float)66 / 255) * albedo * ao;
        float smoothness = gbuffer->getColori(0, x, y).w;
        float metallic = gbuffer->getColori(1, x, y).w;
        Vec3 F0 = lerp<Vec3>(Vec3(0.04f), albedo, metallic);
        float roughness = (1 - smoothness) * (1 - smoothness);

        Vec3 result;
        for (uint32_t i = 0; i < pointLights.size(); i++) {
            PointLight light = pointLights[i];
            Vec3 L = (light.position - worldPos).normalize();
            Vec3 H = (L + V).normalize();
            float NdotL = std::max(N.dot(L), 0.0f);
            float NdotV = std::max(N.dot(V), 0.0f);
            float NdotH = std::max(N.dot(H), 0.0f);
            float LdotH = std::max(L.dot(H), 0.0f);

            float diffuseTerm = DisneyDiffuse(NdotV, NdotL, LdotH, 1 - smoothness) * NdotL;
            float reflectivity = std::lerp(0.04f, 1.0f, metallic);

            Vec3 diffColor = gammaToLinear(albedo) * (1 - reflectivity);

            float Vis = V_SmithJointGGX(NdotL, NdotV, roughness);
            float D = D_GGX(NdotH, roughness);
            float specularTerm = std::max(0.0f, Vis * D * NdotL);

            float distance = (light.position - worldPos).length();
            float falloff = lightFallOff(distance, light.radius);
            Vec3 lightColor = light.color * light.intensity * falloff;

            result += diffColor * lightColor * diffuseTerm + specularTerm * lightColor * FresnelTerm(F0, LdotH);
            result *= (1 - computeShadow(worldPos, NdotL, depthMaps[i]));
            result += 4 * computeIndirectLight(worldPos, N, albedo, depthMaps[i]) * ao;
        }

        result += ambient;
        result = linearToGamma(result); 
        return { Vec4(result, 1), Vec4(), Vec4(), Vec4() };
    }
public:
    Vec3 cameraPos;
    Mat4 lightSpaceMatrix;
    Mat4 view;
    Mat4 invView;
    FrameBuffer* gbuffer;
    FrameBuffer* ssao;
    ColorBuffer* checkerborad;
    ColorBuffer* star;
    ColorBuffer* wood;
    std::vector<FrameBuffer*> depthMaps;
    std::vector<PointLight> pointLights;
    uint32_t rsmSampleNums = 256;

    Mat4 model; // Identity matrix, for back-face culling, actually useless
private:
    Vec3 FresnelTerm(Vec3 F0, float cosA) {
        float t = pow5(1 - cosA);
        return F0 + (1 - F0) * t;
    }

    float D_GGX(float NdotH, float roughness) {
        float a2 = roughness * roughness;
        float d = (NdotH * a2 - NdotH) * NdotH + 1;
        return a2 / (PI * d * d + 1e-7f);
    }

    // V = G / (4 * NdotL * NdotV)
    float V_SmithJointGGX(float NdotL, float NdotV, float roughness) {
        float a2 = roughness * roughness;
        float lambdaV = NdotL * sqrt((-NdotV * a2 + NdotV) * NdotV + a2);
        float lambdaL = NdotV * sqrt((-NdotL * a2 + NdotL) * NdotL + a2);
        return 0.5f / (lambdaV + lambdaL + 1e-7f);
    }

    float DisneyDiffuse(float NdotV, float NdotL, float LdotH, float perceptualRoughness) {
        float fd90 = 0.5f + 2 * LdotH * LdotH * perceptualRoughness;
        float lightScatter = (1 + (fd90 - 1) * pow5(1 - NdotL));
        float viewScatter = (1 + (fd90 - 1) * pow5(1 - NdotV));
        return lightScatter * viewScatter / PI;
    }

    float lightFallOff(float distance, float lightRadius) {
        float r = distance / lightRadius;
        float r2 = r * r;
        float r4 = r2 * r2;
        float num = saturate(1 - r4);
        float num2 = num * num;
        return num2 / (1 + distance * distance);
    }

    float linearizeDepth(float z, float near = 0.1f, float far = 100) {
        return near * far / (far - (far - near) * z);
    }

    float computeShadow(const Vec3& worldPos, float NdotL, FrameBuffer* depthMap) {
        Vec4 lightSpacePos = lightSpaceMatrix * Vec4(worldPos, 1);
        Vec4 uv = (lightSpacePos / lightSpacePos.w) * 0.5f + 0.5f; // only x and y are valid
        float z = (lightSpacePos / lightSpacePos.w).z;

        if (z < 0.0f || z > 1.0f)
            return 0.0f;

        float sum = 0.0f;
        uint32_t sampleCount = 16;
        for (uint32_t i = 0; i < sampleCount; i++) {
            uint32_t x = clamp(uv.x * depthMap->width + 5 * POISSON16[i].x, 0.0f, static_cast<float>(depthMap->width - 1));
            uint32_t y = clamp(uv.y * depthMap->height + 5 * POISSON16[i].y, 0.0f, static_cast<float>(depthMap->height - 1));
            float pcfDepth = depthMap->getDepth(x, y);
            sum += linearizeDepth(z) - 0.3f > linearizeDepth(pcfDepth) ? 1.0f : 0.0f;
        }
        return sum / sampleCount;
    }

    Vec3 computeIndirectLight(const Vec3& worldPos, const Vec3& N, const Vec3& albedo, FrameBuffer* depthMap) {
        Vec4 lightSpacePos = lightSpaceMatrix * Vec4(worldPos, 1);
        Vec4 uv = (lightSpacePos / lightSpacePos.w) * 0.5f + 0.5f; // only x and y are valid
        float z = (lightSpacePos / lightSpacePos.w).z;

        if (z < 0.0f || z > 1.0f)
            return 0.0f;

        Vec3 indirect;
        for (uint32_t i = 0; i < rsmSampleNums; i++) {
            uint32_t x = clamp((uv.x + 0.3 * sampleCoordsAndWeights[i].x) * depthMap->width, 0.0f, static_cast<float>(depthMap->width - 1));
            uint32_t y = clamp((uv.y + 0.3 * sampleCoordsAndWeights[i].y) * depthMap->height, 0.0f, static_cast<float>(depthMap->height - 1));

            Vec3 lightPos = Vec3(depthMap->getColori(0, x, y));
            Vec3 lightNormal = Vec3(depthMap->getColori(1, x, y));
            Vec3 flux = Vec3(depthMap->getColori(2, x, y));
            Vec3 L = (lightPos - worldPos).normalize();
            float NdotL = std::max(N.dot(L), 0.0f);

            float cosP = NdotL;
            float cosQ = std::max(lightNormal.dot(-1 * L), 0.0f);
            float weight = sampleCoordsAndWeights[i].z;
            indirect += (albedo / PI) * flux * cosP * cosQ * weight / pow((lightPos - worldPos).length(), 2);
        }
        return clamp(indirect / rsmSampleNums, 0.0f, 1.0f);
    }

    void generateRSMSamples() {
        for (uint32_t i = 0; i < rsmSampleNums; i++) {
            float xi1 = u(e);
            float xi2 = u(e);
            sampleCoordsAndWeights.push_back({ xi1 * sin(2 * PI * xi2), xi1 * cos(2 * PI * xi2), xi1 * xi1 });
        }
    }

private:
    Vec2 POISSON16[16] = {
	    Vec2(0.3040781f,-0.1861200f), Vec2(0.1485699f,-0.0405212f), Vec2(0.4016555f,0.1252352f),
	    Vec2(-0.1526961f,-0.1404687f), Vec2(0.3480717f,0.3260515f), Vec2(0.0584860f,-0.3266001f),
	    Vec2(0.0891062f,0.2332856f), Vec2(-0.3487481f,-0.0159209f), Vec2(-0.1847383f,0.1410431f),
	    Vec2(0.4678784f,-0.0888323f), Vec2(0.1134236f,0.4119219f), Vec2(0.2856628f,-0.3658066f),
	    Vec2(-0.1765543f,0.3937907f), Vec2(-0.0238326f,0.0518298f), Vec2(-0.2949835f,-0.3029899f),
	    Vec2(-0.4593541f,0.1720255f) };
    std::vector<Vec3> sampleCoordsAndWeights;
    std::default_random_engine e;
    std::uniform_real_distribution<float> u;
};

struct SSRMaterial {
    using V2F = GenericV2F<>;

    SSRMaterial() : randomFloats(std::uniform_real_distribution<float>(0.0f, 1.0f)) {}

    V2F vert(const Vertex& i) {
        V2F o;
        o.gl_Position = Vec4(i.position, 1);
        o.gl_ZCamera = 1;
        return o;
    }

    std::array<Vec4, 4> frag(const V2F& fs_in) {
        Vec4 target;
        float x = fs_in.gl_FragCoord.x;
        float y = fs_in.gl_FragCoord.y;
        Vec2 uv(x / frame->width, y / frame->height);
        uint32_t objectID = gbuffer->getColori(0, x, y).z;
        float metallic = gbuffer->getColori(1, x, y).y;

        Vec3 position(gbuffer->getColori(2, x, y));
        // exponential fog
        Vec3 worldPos = Vec3(invView * Vec4(position, 1));
        float distance = (worldPos - cameraPos).length();
        float distanceFactor = 1 - exp(-fogDensity * distance);
        float heightFactor = clamp(exp(-fogDensity * (worldPos.y + fogHmin)), 0.0f, 0.6f);
        float fogFactor = distanceFactor * heightFactor;
        
        if (!enableSSR || objectID != 2) {
            target = frame->getColori(0, x, y);
            //target = lerp(target, fogColor, fogFactor);
            return {target, Vec4(), Vec4(), Vec4()};
        }
  
        Vec3 normal = Vec3(gbuffer->getColori(3, x, y));
        Vec3 viewDir = -1 * position.normalize();
        Vec3 reflection = 2.0f * normal * viewDir.dot(normal) - viewDir;

        Vec4 SSRColor;
        if (isSamplingEnabled) {
            for (uint32_t i = 0; i < sampleCount; i++) {
                Vec2 jitter = randomJitter(uv);
                SSRColor += trace(position, reflection + Vec3(jitter.x, jitter.y, 0.0f));
            }
            SSRColor /= static_cast<float>(sampleCount);
        } else {
            SSRColor = trace(position, reflection);
        }

        target = lerp(SSRColor, frame->getColori(0, x, y), 0.9f);
        //target = lerp(target, fogColor, fogFactor);
        return {target, Vec4(), Vec4(), Vec4()}; 
    }

public:
    FrameBuffer* frame;
    FrameBuffer* gbuffer;

    Mat4 proj;
    Mat4 invProj;
    Mat4 view;
    Mat4 invView;
    Vec3 cameraPos;

    bool enableSSR = true;
    bool isSamplingEnabled = true;
    bool isAdaptiveStepEnabled = false;
    uint32_t sampleCount = 4;
    float rayStep = 0.15f;
    float distanceBias = 0.08f;
    uint32_t iterationCount = 64;

    float fogDensity = 0.5f;
    float fogHmin = 0.3f;
    Vec4 fogColor = Vec4(1.0f);
    Mat4 model; // Identity matrix, for back-face culling, actually useless
private:
    Vec4 trace(const Vec3& position, const Vec3& reflection) {
        Vec3 currentPos = position;
        Vec3 step = reflection * rayStep;
        for (uint32_t i = 0; i < iterationCount; i++) {
            currentPos += step;
            Vec2 projectedUV = generateProjectedUV(currentPos);
            if (!isValidUV(projectedUV)) {
                return Vec4(0.0f);
            }

            uint32_t coord_u = clamp(projectedUV.x * frame->width, 0.0f, frame->width - 1);
            uint32_t coord_v = clamp(projectedUV.y * frame->height, 0.0f, frame->height - 1);
            float depth = gbuffer->getDepth(coord_u, coord_v);
            if (std::abs(linearizeDepth(depth) - currentPos.z) < distanceBias) {
                return frame->getColori(0, coord_u, coord_v);
            }
            if (isAdaptiveStepEnabled) {
                float delta = std::abs(linearizeDepth(depth) - currentPos.z);
                step *= 0.5f / delta;
            }
        }

        return Vec4(0.0f); // No intersection found, return black
    }

    Vec2 generateProjectedUV(const Vec3& position) {
        Vec4 clipPos = proj * Vec4(position, 1.0f); // Transform position to clip space
        Vec3 ndcPos = Vec3(clipPos / clipPos.w); // Normalize to NDC space
        return Vec2(ndcPos.x, ndcPos.y) * 0.5f + Vec2(0.5f); // Convert to UV coordinates
    }

    bool isValidUV(const Vec2& uv) {
        return uv.x >= 0.0f && uv.x <= 1.0f && uv.y >= 0.0f && uv.y <= 1.0f;
    }

    Vec3 generatePositionFromDepth(const Vec2& uv, float depth) {
        Vec4 clipPos = Vec4(uv.x * 2.0f - 1.0f, uv.y * 2.0f - 1.0f, depth * 2.0f - 1.0f, 1.0f);
        Vec4 viewPos = invProj * clipPos; 
        return Vec3(viewPos / viewPos.w);
    }

    Vec2 randomJitter(const Vec2& uv) {
        float randX = randomFloats(generator) * 2.0f - 1.0f; // Random jitter in X direction
        float randY = randomFloats(generator) * 2.0f - 1.0f; // Random jitter in Y direction
        return Vec2(randX, randY) * 0.01f; // Scale jitter for anti-aliasing
    }

    float linearizeDepth(float z, float near = 0.1f, float far = 100) {
        return near * far / (far - (far - near) * z);
    }

    std::uniform_real_distribution<float> randomFloats; // Distribution for random float generation
    std::default_random_engine generator; // Random number generator
};

struct Renderer {
    template <typename MaterialType>
    void render(MaterialType& material, FrameBuffer& frame);
    bool writeDepth = true;
    Vec3 cameraPos;
    std::vector<Vertex> vertex_buffer;
    std::vector<uint32_t> indices;
};

template <typename MaterialType>
void Renderer::render(MaterialType& material, FrameBuffer& frame) {
    using V2F = typename MaterialType::V2F;
    // Vertex Shader & Perspective Divide
    std::vector<V2F> v2fs;
    for (auto& v : vertex_buffer) {
        auto vs_out = material.vert(v);
        vs_out.gl_Position = vs_out.gl_Position / vs_out.gl_Position.w;
        v2fs.push_back(vs_out);

        // for back-face culling
        v.position = Vec3(material.model * Vec4(v.position, 1)); 
    }

    // Back-Face Culling
    backFaceCulling(vertex_buffer, indices, cameraPos);

    // Viewport Transformation
    Mat4 viewport = createViewportMatrix(frame.width, frame.height);
    for (auto& v : v2fs) 
        v.gl_Position = viewport * v.gl_Position;
    
    // Rasterization & Fragment Shader
    #pragma omp parallel for
    for (int i = 0; i < indices.size(); i += 3) {
        uint32_t idx0 = indices[i];
        uint32_t idx1 = indices[i + 1];
        uint32_t idx2 = indices[i + 2];

        Vec3 v0 = Vec3(v2fs[idx0].gl_Position);
        Vec3 v1 = Vec3(v2fs[idx1].gl_Position);
        Vec3 v2 = Vec3(v2fs[idx2].gl_Position);

        int xMin = std::max(0, std::min(std::min((int)v0.x, (int)v1.x), (int)v2.x));
        int xMax = std::min((int)frame.width - 1, std::max(std::max((int)v0.x, (int)v1.x), (int)v2.x));
        int yMin = std::max(0, std::min(std::min((int)v0.y, (int)v1.y), (int)v2.y));
        int yMax = std::min((int)frame.height - 1, std::max(std::max((int)v0.y, (int)v1.y), (int)v2.y));

        for (int y = yMin; y <= yMax; y++) {
            #pragma omp parallel for
            for (int x = xMin; x <= xMax; x++) {
                float lambda1, lambda2, lambda3;
                computeBarycentricCoords(v0, v1, v2, Vec2(x + 0.5f, y + 0.5f), lambda1, lambda2, lambda3);

                if (lambda1 >= 0.0f && lambda2 >= 0.0f && lambda3 >= 0.0f) {
                    float z = 1.0f / (lambda1 / v0.z + lambda2 / v1.z + lambda3 / v2.z);

                    if (z >= 0 && z <= frame.getDepth(x, y)) {
                        if (writeDepth)
                            frame.getDepth(x, y) = z;

                        V2F fs_in;
                        fs_in.gl_ZDepth = z;
                        fs_in.gl_FragCoord = Vec2(x, y);
                        forEachInTuple(fs_in.attributes, [&](auto& attr, auto i) {
                            constexpr std::size_t index = decltype(i)::value;
                            auto& v0_attr = std::get<index>(v2fs[idx0].attributes);
                            auto& v1_attr = std::get<index>(v2fs[idx1].attributes);
                            auto& v2_attr = std::get<index>(v2fs[idx2].attributes);

                            attr = perspectiveCorrectInterpolate(v0_attr, v1_attr, v2_attr, lambda1, lambda2, lambda3,
                                v2fs[idx0].gl_ZCamera, v2fs[idx1].gl_ZCamera, v2fs[idx2].gl_ZCamera);
                        });

                        std::array<Vec4, 4> frag_colors = material.frag(fs_in);                  
                        for (uint32_t i = 0; i < 4; i++) {
                            frame.getColori(i, x, y) = frag_colors[i];
                        }
                    }
                }
            }
        }
    }
}

void enableFloatingPointExceptions() {
    unsigned int currentControl;
    _controlfp_s(&currentControl, 0, 0); // 获取当前设置
    currentControl &= ~(EM_INVALID | EM_ZERODIVIDE); // 启用无效操作和除零异常
    _controlfp_s(&currentControl, currentControl, _MCW_EM); // 设置新的控制模式
}

int main() {
    //enableFloatingPointExceptions();

    auto start = std::chrono::high_resolution_clock::now();

    // Application Set Up
    Renderer renderer;
    FrameBuffer depth_map(1024, 1024);
    FrameBuffer g_buffer(SCREEN_WIDTH, SCREEN_HEIGHT);
    FrameBuffer ao(SCREEN_WIDTH, SCREEN_HEIGHT);
    FrameBuffer ao_final(SCREEN_WIDTH, SCREEN_HEIGHT);
    FrameBuffer frame(SCREEN_WIDTH, SCREEN_HEIGHT);
    std::vector<Vertex> vertex_buffer;
    Vec3 light_position(0, 8.4f, -5.49f);
    PointLight light{ 300, 20, Vec3(1, 1, 1), light_position};
    Vec3 camera_position(0, 6.22f, -8.03f);
    Vec3 lookat_position(0, 0, 5);
    
    std::vector<float> sphere_vertices;
    std::vector<uint32_t> sphere_indices;
    generateSphere(1, 40, 40, sphere_vertices, sphere_indices);

    ColorBuffer star(256, 256);
    generateStarTexture(star, 128);
    ColorBuffer checkerborad(256, 256);
    generateCheckerboradTexture(checkerborad, 20);
    ColorBuffer wood(512, 512);
    generateWoodTexture(wood, 50.0f, 0.6f, 1.0f, 0.0f);
    ColorBuffer normal_map(512, 512);
    generateWaveNormalMap(normal_map, 30, 0.15f);

    // Shadow Pass : Cube Right
    for (uint32_t i = 0; i < cube_vertices.size(); i += 8) {
        Vertex v;
        v.position = Vec3(cube_vertices[i], cube_vertices[i + 1], cube_vertices[i + 2]);
        v.normal = Vec3(cube_vertices[i + 3], cube_vertices[i + 4], cube_vertices[i + 5]);
        v.uv = Vec2(cube_vertices[i + 6], cube_vertices[i + 7]);
        vertex_buffer.push_back(v);
    }

    DepthMapMaterial cube_depth;
    cube_depth.model = createModelMatrix(Vec3(1.82f, 1.5f, -0.04f), Vec3(0, 1, 0), 15, Vec3(1.5f));
    cube_depth.lightSpaceMatrix = createPerspectiveMatrix(90, 1, 0.1f, 100) * createLookAtMatrix(light_position, Vec3(0, 0, 5));
    cube_depth.albedo = Vec3(1);
    cube_depth.light = light;

    renderer.vertex_buffer = std::move(vertex_buffer);
    renderer.indices = cube_indices;
    renderer.cameraPos = light_position;
    renderer.render(cube_depth, depth_map);

    // Shadow Pass : Cube Left
    for (uint32_t i = 0; i < cube_vertices.size(); i += 8) {
        Vertex v;
        v.position = Vec3(cube_vertices[i], cube_vertices[i + 1], cube_vertices[i + 2]);
        v.normal = Vec3(cube_vertices[i + 3], cube_vertices[i + 4], cube_vertices[i + 5]);
        v.uv = Vec2(cube_vertices[i + 6], cube_vertices[i + 7]);
        vertex_buffer.push_back(v);
    }

    cube_depth.model = createModelMatrix(Vec3(-2.35f, 3, 1.86f), Vec3(0, 1, 0), -15, Vec3(1.5f, 3, 1.5f));
    cube_depth.lightSpaceMatrix = createPerspectiveMatrix(90, 1, 0.1f, 100) * createLookAtMatrix(light_position, Vec3(0, 0, 5));
    cube_depth.albedo = Vec3(1);
    cube_depth.light = light;

    renderer.vertex_buffer = std::move(vertex_buffer);
    renderer.indices = cube_indices;
    renderer.render(cube_depth, depth_map);

    // Shadow Pass : Quad_Down
    for (uint32_t i = 0; i < quad_vertices.size(); i += 5) {
        Vertex v;
        v.position = Vec3(quad_vertices[i], quad_vertices[i + 1], quad_vertices[i + 2]);
        v.uv = Vec2(quad_vertices[i + 3], quad_vertices[i + 4]);
        v.normal = Vec3(0, 0, -1);
        vertex_buffer.push_back(v);
    }

    DepthMapMaterial quad_depth;
    quad_depth.model = createModelMatrix(Vec3(0), Vec3(1, 0, 0), 90, Vec3(5, 5, 1));
    quad_depth.lightSpaceMatrix = createPerspectiveMatrix(90, 1, 0.1f, 100) * createLookAtMatrix(light_position, Vec3(0, 0, 5));
    quad_depth.albedo = Vec3(0.94f, 1, 0);
    quad_depth.light = light;
    quad_depth.useAlbedoMap = true;
    quad_depth.albedoMap = &wood;

    renderer.vertex_buffer = vertex_buffer;
    renderer.indices = quad_indices;
    renderer.render(quad_depth, depth_map);
    quad_depth.useAlbedoMap = false;

    // Shadow Pass : Quad_Back
    quad_depth.model = createModelMatrix(Vec3(0, 5, 5), Vec3(1, 0, 0), 0, Vec3(5, 5, 1));
    quad_depth.albedo = Vec3(0.65f, 0.05f, 0.05f);

    renderer.vertex_buffer = vertex_buffer;
    renderer.indices = quad_indices;
    renderer.render(quad_depth, depth_map);

    // Shadow Pass : Quad_Left
    quad_depth.model = createModelMatrix(Vec3(-5, 5, 0), Vec3(0, 1, 0), -90, Vec3(5, 5, 1));
    quad_depth.albedo = Vec3(0.12f, 0.45f, 0.15f);

    renderer.writeDepth = false;
    renderer.vertex_buffer = vertex_buffer;
    renderer.indices = quad_indices;
    renderer.render(quad_depth, depth_map);
    renderer.writeDepth = true;

    // Shadow Pass : Quad_Right
    quad_depth.model = createModelMatrix(Vec3(5, 5, 0), Vec3(0, 1, 0), 90, Vec3(5, 5, 1));
    quad_depth.albedo = Vec3(0, (float)249 / 255, 1);

    renderer.vertex_buffer = std::move(vertex_buffer);
    renderer.indices = quad_indices;
    renderer.render(quad_depth, depth_map);

    // Geometry Pass : Cube_Right
    for (uint32_t i = 0; i < cube_vertices.size(); i += 8) {
        Vertex v;
        v.position = Vec3(cube_vertices[i], cube_vertices[i + 1], cube_vertices[i + 2]);
        v.normal = Vec3(cube_vertices[i + 3], cube_vertices[i + 4], cube_vertices[i + 5]);
        v.uv = Vec2(cube_vertices[i + 6], cube_vertices[i + 7]);
        vertex_buffer.push_back(v);
    }

    GBufferMaterial cube_gbuffer;
    cube_gbuffer.model = createModelMatrix(Vec3(1.82f, 1.5f, -0.04f), Vec3(0, 1, 0), 15, Vec3(1.5f));
    cube_gbuffer.view = createLookAtMatrix(camera_position, lookat_position);
    cube_gbuffer.projection = createPerspectiveMatrix(60, (float)SCREEN_WIDTH / SCREEN_HEIGHT, 0.1f, 100);
    cube_gbuffer.albedo = Vec3(1);
    cube_gbuffer.emissive = Vec3(0);
    cube_gbuffer.metallic = 0.5f;
    cube_gbuffer.smoothness = 0.8f;
    cube_gbuffer.objectID = 1;

    renderer.vertex_buffer = std::move(vertex_buffer);
    renderer.indices = cube_indices;
    renderer.cameraPos = camera_position;
    renderer.render(cube_gbuffer, g_buffer);

    // Geometry Pass : Cube
    for (uint32_t i = 0; i < cube_vertices.size(); i += 8) {
        Vertex v;
        v.position = Vec3(cube_vertices[i], cube_vertices[i + 1], cube_vertices[i + 2]);
        v.normal = Vec3(cube_vertices[i + 3], cube_vertices[i + 4], cube_vertices[i + 5]);
        v.uv = Vec2(cube_vertices[i + 6], cube_vertices[i + 7]);
        vertex_buffer.push_back(v);
    }

    cube_gbuffer.model = createModelMatrix(Vec3(-2.35f, 3, 1.86f), Vec3(0, 1, 0), -15, Vec3(1.5f, 3, 1.5f));
    cube_gbuffer.view = createLookAtMatrix(camera_position, lookat_position);
    cube_gbuffer.projection = createPerspectiveMatrix(60, (float)SCREEN_WIDTH / SCREEN_HEIGHT, 0.1f, 100);
    cube_gbuffer.albedo = Vec3(1);
    cube_gbuffer.emissive = Vec3(0);
    cube_gbuffer.metallic = 0.5f;
    cube_gbuffer.smoothness = 0.8f;
    cube_gbuffer.objectID = 0;

    renderer.vertex_buffer = std::move(vertex_buffer);
    renderer.indices = cube_indices;
    renderer.render(cube_gbuffer, g_buffer);

    // Geometry Pass : Quad_Down
    for (uint32_t i = 0; i < quad_vertices.size(); i += 5) {
        Vertex v;
        v.position = Vec3(quad_vertices[i], quad_vertices[i + 1], quad_vertices[i + 2]);
        v.uv = Vec2(quad_vertices[i + 3], quad_vertices[i + 4]);
        v.normal = Vec3(0, 0, -1);
        vertex_buffer.push_back(v);
    }

    GBufferMaterial quad_gbuffer;
    quad_gbuffer.model = createModelMatrix(Vec3(0), Vec3(1, 0, 0), 90, Vec3(5, 5, 1));
    quad_gbuffer.view = createLookAtMatrix(camera_position, lookat_position);
    quad_gbuffer.projection = createPerspectiveMatrix(60, (float)SCREEN_WIDTH / SCREEN_HEIGHT, 0.1f, 100);
    quad_gbuffer.albedo = Vec3(0.94f, 1, 0);
    quad_gbuffer.emissive = Vec3(0);
    quad_gbuffer.metallic = 0.5f;
    quad_gbuffer.smoothness = 0;
    quad_gbuffer.objectID = 2;
    //quad_gbuffer.useNormalMap = true;
    quad_gbuffer.normalMap = &normal_map;

    renderer.vertex_buffer = vertex_buffer;
    renderer.indices = quad_indices;
    renderer.render(quad_gbuffer, g_buffer);
    quad_gbuffer.useNormalMap = false;

    // Geometry Pass : Quad_Back
    quad_gbuffer.model = createModelMatrix(Vec3(0, 5, 5), Vec3(1, 0, 0), 0, Vec3(5, 5, 1));
    quad_gbuffer.albedo = Vec3(0.65f, 0.05f, 0.05f);
    quad_gbuffer.objectID = 3;

    renderer.vertex_buffer = vertex_buffer;
    renderer.indices = quad_indices;
    renderer.render(quad_gbuffer, g_buffer);

    // Geometry Pass : Quad_Left
    quad_gbuffer.model = createModelMatrix(Vec3(-5, 5, 0), Vec3(0, 1, 0), -90, Vec3(5, 5, 1));
    quad_gbuffer.albedo = Vec3(0.12f, 0.45f, 0.15f);
    quad_gbuffer.objectID = 4;

    renderer.vertex_buffer = vertex_buffer;
    renderer.indices = quad_indices;
    renderer.render(quad_gbuffer, g_buffer);

    // Geometry Pass : Quad_Right
    quad_gbuffer.model = createModelMatrix(Vec3(5, 5, 0), Vec3(0, 1, 0), 90, Vec3(5, 5, 1));
    quad_gbuffer.albedo = Vec3(0, (float)249 / 255, 1);
    quad_gbuffer.objectID = 5;

    renderer.vertex_buffer = vertex_buffer;
    renderer.indices = quad_indices;
    renderer.render(quad_gbuffer, g_buffer);

    // SSAO Pass
    SSAOMaterial ssao;
    ssao.gbuffer = &g_buffer;
    ssao.view = createLookAtMatrix(camera_position, lookat_position);
    ssao.projection = createPerspectiveMatrix(60, (float)SCREEN_WIDTH / SCREEN_HEIGHT, 0.1f, 100);

    renderer.vertex_buffer = vertex_buffer;
    renderer.indices = quad_indices;
    renderer.render(ssao, ao);

    // SSAO-Blur Pass
    SSAOBlurMaterial ssao_blur;
    ssao_blur.ssao = &ao;
    ssao_blur.view = createLookAtMatrix(camera_position, lookat_position);

    renderer.vertex_buffer = vertex_buffer;
    renderer.indices = quad_indices;
    renderer.render(ssao_blur, ao_final);

    // Shading Pass
    Mat4 view = createLookAtMatrix(camera_position, lookat_position);
    Mat4 inv_view = view;
    inv_view.inverse();

    StandardMaterial standard_shading;
    standard_shading.star = &star;
    standard_shading.checkerborad = &checkerborad;
    standard_shading.wood = &wood;
    standard_shading.gbuffer = &g_buffer;
    standard_shading.ssao = &ao_final;
    standard_shading.cameraPos = camera_position;
    standard_shading.view = view;
    standard_shading.invView = inv_view;
    standard_shading.lightSpaceMatrix = createPerspectiveMatrix(90, 1, 0.1f, 100) * createLookAtMatrix(light_position, Vec3(0, 0, 5));
    standard_shading.depthMaps.push_back(&depth_map);
    standard_shading.pointLights.push_back(light);
    
    renderer.vertex_buffer = vertex_buffer;
    renderer.indices = quad_indices;
    renderer.render(standard_shading, frame);

    // SSR Pass
    Mat4 proj = createPerspectiveMatrix(60, (float)SCREEN_WIDTH / SCREEN_HEIGHT, 0.1f, 100);
    Mat4 inv_proj = proj;
    inv_proj.inverse();
    
    SSRMaterial ssr;
    ssr.frame = &frame;
    ssr.gbuffer = &g_buffer;
    ssr.view = view;
    ssr.invView = inv_view;
    ssr.proj = proj;
    ssr.invProj = inv_proj;
    ssr.cameraPos = camera_position;

    renderer.vertex_buffer = vertex_buffer;
    renderer.indices = quad_indices;
    renderer.render(ssr, frame);

    // Image Output
    gaussianFilterNTimes(frame.colorBuffers[0], 3);
    nearestScaling(frame.colorBuffers[0], 0.25f, 0.25f);
    displayImage(frame.colorBuffers[0]);

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> serial_time = end - start;
    std::clog << "execution time: " << serial_time.count() << " seconds\n";
}