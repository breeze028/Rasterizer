#pragma once

#include <cmath>

constexpr float PI = 3.14159f;

struct Vec4;

struct Vec3 {
    float x, y, z;

    Vec3() : x(0), y(0), z(0) {}
    Vec3(float x) : x(x), y(x), z(x) {}
    Vec3(float x, float y, float z) : x(x), y(y), z(z) {}
    explicit Vec3(const Vec4& v);

    Vec3 operator+(const Vec3& v) const { return { x + v.x, y + v.y, z + v.z }; }
    Vec3 operator-(const Vec3& v) const { return { x - v.x, y - v.y, z - v.z }; }
    Vec3 operator*(const Vec3& v) const { return { x * v.x, y * v.y, z * v.z }; }
    Vec3 operator*(float scalar) const { return { x * scalar, y * scalar, z * scalar }; }
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
        return { y * v.z - z * v.y, z * v.x - x * v.z, x * v.y - y * v.x };
    }

    float length() const { return std::sqrt(x * x + y * y + z * z); }
    Vec3 normalize() const { float len = length(); return { x / len, y / len, z / len }; }
};

struct Vec4 {
    float x, y, z, w;

    Vec4() : x(0), y(0), z(0), w(0) {}
    Vec4(float x) : x(x), y(x), z(x), w(x) {}
    Vec4(float x, float y, float z, float w) : x(x), y(y), z(z), w(w) {}
    explicit Vec4(const Vec3& v, float w) : x(v.x), y(v.y), z(v.z), w(w) {}

    Vec4 operator+(const Vec4& v) const { return { x + v.x, y + v.y, z + v.z, w + v.w }; }
    Vec4 operator-(const Vec4& v) const { return { x - v.x, y - v.y, z - v.z, w - v.w }; }
    Vec4 operator*(float scalar) const { return { x * scalar, y * scalar, z * scalar, w * scalar }; }
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

    Vec2 operator+(const Vec2& v) const { return { x + v.x, y + v.y }; }
    Vec2 operator*(float scalar) const { return { x * scalar, y * scalar }; }
    Vec2 operator/(float scalar) const { return *this * (1 / scalar); }
};

Vec3::Vec3(const Vec4& v) : x(v.x), y(v.y), z(v.z) {}

Vec3 operator+(float scalar, const Vec3& vec) {
    return { scalar + vec.x, scalar + vec.y, scalar + vec.z };
}

Vec3 operator-(float scalar, const Vec3& vec) {
    return { scalar - vec.x, scalar - vec.y, scalar - vec.z };
}

Vec3 operator*(float scalar, const Vec3& vec) {
    return { scalar * vec.x, scalar * vec.y, scalar * vec.z };
}

Vec4 operator*(float scalar, const Vec4& vec) {
    return { scalar * vec.x, scalar * vec.y, scalar * vec.z, scalar * vec.w };
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

inline float clamp(float x, float min, float max) {
    return (x < min) ? min : (x > max) ? max : x;
}

inline Vec3 clamp(Vec3 v, Vec3 min, Vec3 max) {
    float x = clamp(v.x, min.x, max.x);
    float y = clamp(v.y, min.y, max.y);
    float z = clamp(v.z, min.z, max.z);
    return { x, y, z };
}

inline float saturate(float x) {
    return clamp(x, 0.0f, 1.0f);
}

inline float pow5(float v) {
    return v * v * v * v * v;
}

inline float smoothstep(float edge0, float edge1, float x) {
    float t = (x - edge0) / (edge1 - edge0);
    t = clamp(t, 0.0f, 1.0f);
    return t * t * (3.0f - 2.0f * t);
}

inline Vec3 gammaToLinear(Vec3 gammaValue, float gamma = 2.2f) {
    return { powf(gammaValue.x, gamma), powf(gammaValue.y, gamma), powf(gammaValue.z, gamma) };
}

inline Vec3 linearToGamma(Vec3 linearValue, float gamma = 2.2f) {
    return gammaToLinear(linearValue, 1 / gamma);
}

inline Mat4 createTranslateMatrix(const Vec3& translation) {
    Mat4 result;
    result.m[0][3] = translation.x;
    result.m[1][3] = translation.y;
    result.m[2][3] = translation.z;
    return result;
}

inline Mat4 createRotateMatrix(float angle, Vec3 axis) {
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

inline Mat4 createScaleMatrix(const Vec3& scale) {
    Mat4 result;
    result.m[0][0] = scale.x;
    result.m[1][1] = scale.y;
    result.m[2][2] = scale.z;
    return result;
}

inline Mat4 createModelMatrix(const Vec3& translation = Vec3(0, 0, 0), const Vec3& rotationAxis = Vec3(0, 1, 0), float rotationAngle = 0, const Vec3& scale = Vec3(1, 1, 1)) {
    return createTranslateMatrix(translation) * createRotateMatrix(rotationAngle, rotationAxis) * createScaleMatrix(scale);
}

inline Mat4 createLookAtMatrix(const Vec3& eyePosition, const Vec3& focusPosition, const Vec3& upDirection = Vec3(0, 1, 0)) {
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
    return Mat4{ x_axis.x, x_axis.y, x_axis.z, -eyePosition.dot(x_axis),
                y_axis.x, y_axis.y, y_axis.z, -eyePosition.dot(y_axis),
                z_axis.x, z_axis.y, z_axis.z, -eyePosition.dot(z_axis),
                0, 0, 0, 1 };
}


inline Mat4 createPerspectiveMatrix(const float fovY, const float aspectRatio, const float near, const float far) {
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

inline Mat4 createViewportMatrix(uint32_t width, uint32_t height) {
    Mat4 res;
    res.m[0][0] = (float)width / 2;
    res.m[0][3] = (float)width / 2;
    res.m[1][1] = (float)height / 2;
    res.m[1][3] = (float)height / 2;
    return res;
}

inline Vec3 computeNormal(const Vec3& v0, const Vec3& v1, const Vec3& v2) {
    return (v1 - v0).cross(v2 - v0).normalize();
}

inline void computeBarycentricCoords(const Vec2& v0, const Vec2& v1, const Vec2& v2, const Vec2& p, float& lambda1, float& lambda2, float& lambda3) {
    float denominator = (v1.y - v2.y) * (v0.x - v2.x) + (v2.x - v1.x) * (v0.y - v2.y);
    lambda1 = ((v1.y - v2.y) * (p.x - v2.x) + (v2.x - v1.x) * (p.y - v2.y)) / denominator;
    lambda2 = ((v2.y - v0.y) * (p.x - v2.x) + (v0.x - v2.x) * (p.y - v2.y)) / denominator;
    lambda3 = 1.0f - lambda1 - lambda2;
}