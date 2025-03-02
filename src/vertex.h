#pragma once

#include <vector>
#include "math.h"

struct Vertex {
    Vec3 position;
    Vec3 normal;
    Vec2 uv;
};

inline std::vector<float> cube_vertices = {
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


inline std::vector<uint32_t> cube_indices = {
    0, 2, 1, 0, 3, 2,
    4, 5, 6, 4, 6, 7,
    8, 9, 10, 8, 10, 11,
    12, 14, 13, 12, 15, 14,
    16, 17, 18, 16, 18, 19,
    20, 22, 21, 20, 23, 22
};

inline std::vector<float> quad_vertices = {
    -1.0f, -1.0f, 0.0f, 0.0f, 0.0f,
    1.0f, -1.0f, 0.0f, 1.0f, 0.0f,
    1.0f, 1.0f, 0.0f, 1.0f, 1.0f,
    -1.0f, 1.0f, 0.0f, 0.0f, 1.0f
};

inline std::vector<uint32_t> quad_indices = {
    0, 2, 1, 0, 3, 2
};

inline void generateSphere(float radius, int segments, int rings, std::vector<float>& vertices,
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
