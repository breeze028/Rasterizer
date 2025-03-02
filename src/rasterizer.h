#pragma once

#include <iostream>
#include <omp.h>
#include "vertex.h"
#include "framebuffer.h"

constexpr int tileSize = 32;

inline bool isBackFace(const Vec3& cameraPosition, const Vec3& v0, const Vec3& v1, const Vec3& v2) {
    Vec3 normal = computeNormal(v0, v1, v2);
    Vec3 viewDir = (cameraPosition - (v0 + v1 + v2) / 3).normalize();
    return normal.dot(viewDir) <= 0.0f;
}

inline void backFaceCulling(const std::vector<Vertex>& vertex_buffer, std::vector<uint32_t>& indices, const Vec3& cameraPosition) {
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

template <typename T>
T perspectiveCorrectInterpolate(const T& v0, const T& v1, const T& v2, float lambda1, float lambda2, float lambda3,
    float z0, float z1, float z2) {
    float w0 = lambda1 / z0;
    float w1 = lambda2 / z1;
    float w2 = lambda3 / z2;
    float w_sum = w0 + w1 + w2;

    return (v0 * w0 + v1 * w1 + v2 * w2) / w_sum;
}

inline void gaussianFilter(ColorBuffer& buffer) {
    float weight[] = { 0.4026f, 0.2442f, 0.0545f };

    ColorBuffer new_buffer(buffer.width + 4, buffer.height + 4);
    #pragma omp parallel for collapse(2)
    for (int y = 2; y < buffer.height + 2; y++)
        for (int x = 2; x < buffer.width + 2; x++)
            new_buffer.getColor(x, y) = buffer.getColor(x - 2, y - 2);

    // X-axis
    #pragma omp parallel for collapse(2)
    for (int y = 2; y < buffer.height + 2; y++) {
        for (int x = 2; x < buffer.width + 2; x++) {
            Vec3 sum = Vec3(new_buffer.getColor(x, y) * weight[0]);
            for (int i = 1; i < 3; i++) {
                sum += Vec3(new_buffer.getColor(x + i, y) * weight[i]);
                sum += Vec3(new_buffer.getColor(x - i, y) * weight[i]);
            }
            new_buffer.getColor(x, y) = Vec4(sum, 1);
        }
    }

    // Y-axis
    #pragma omp parallel for collapse(2)
    for (int y = 2; y < buffer.height + 2; y++) {
        for (int x = 2; x < buffer.width + 2; x++) {
            Vec3 sum = Vec3(new_buffer.getColor(x, y) * weight[0]);
            for (int i = 1; i < 3; i++) {
                sum += Vec3(new_buffer.getColor(x, y + i) * weight[i]);
                sum += Vec3(new_buffer.getColor(x, y - i) * weight[i]);
            }
            new_buffer.getColor(x, y) = Vec4(sum, 1);
        }
    }

    #pragma omp parallel for collapse(2)
    for (int y = 2; y < buffer.height + 2; y++)
        for (int x = 2; x < buffer.width + 2; x++)
            buffer.getColor(x - 2, y - 2) = new_buffer.getColor(x, y);
}

inline void gaussianFilterNTimes(ColorBuffer& buffer, uint32_t N) {
    for (uint32_t i = 0; i < N; i++)
        gaussianFilter(buffer);
}

// only support downscaling
inline void nearestScaling(ColorBuffer& buffer, float scale_width, float scale_height) {
    if (scale_width > 1 || scale_height > 1) return;
    uint32_t src_width = buffer.width;
    uint32_t src_height = buffer.height;
    uint32_t dst_width = static_cast<uint32_t>(src_width * scale_width);
    uint32_t dst_height = static_cast<uint32_t>(src_height * scale_height);

    ColorBuffer new_buffer(dst_width, dst_height);
    #pragma omp parallel for collapse(2)
    for (int y = 0; y < dst_height; y++) {
        for (int x = 0; x < dst_width; x++) {
            uint32_t src_x = clamp(x / scale_width, 0, src_width - 1);
            uint32_t src_y = clamp(y / scale_height, 0, src_height - 1);
            new_buffer.getColor(x, y) = buffer.getColor(src_x, src_y);
        }
    }

    buffer.resize(dst_width, dst_height);
    #pragma omp parallel for collapse(2)
    for (int y = 0; y < dst_height; y++)
        for (int x = 0; x < dst_width; x++)
            buffer.getColor(x, y) = new_buffer.getColor(x, y);
}

inline void displayImage(ColorBuffer& buffer) {
    std::cout << "P3\n" << buffer.width << ' ' << buffer.height << "\n255\n";
    for (uint32_t y = 0; y < buffer.height; y++) {
        std::clog << "\rScanlines remaining: " << buffer.height - y << ' ' << std::flush;
        for (uint32_t x = 0; x < buffer.width; x++) {
            std::cout << static_cast<int>(clamp(buffer.getColor(x, buffer.height - y - 1).x * 255, 0.0f, 255.0f)) << ' '
                << static_cast<int>(clamp(buffer.getColor(x, buffer.height - y - 1).y * 255, 0.0f, 255.0f)) << ' '
                << static_cast<int>(clamp(buffer.getColor(x, buffer.height - y - 1).z * 255, 0.0f, 255.0f)) << '\n';
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
    #pragma omp parallel for schedule(dynamic)
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

        int tileXStart = (xMin / tileSize) * tileSize;
        int tileXEnd = ((xMax + tileSize) / tileSize) * tileSize;
        int tileYStart = (yMin / tileSize) * tileSize;
        int tileYEnd = ((yMax + tileSize) / tileSize) * tileSize;

        #pragma omp parallel for collapse(2)
        for (int tileY = tileYStart; tileY < tileYEnd; tileY += tileSize) {
            for (int tileX = tileXStart; tileX < tileXEnd; tileX += tileSize) {
                int localXMin = std::max(xMin, tileX);
                int localXMax = std::min(xMax, tileX + tileSize - 1);
                int localYMin = std::max(yMin, tileY);
                int localYMax = std::min(yMax, tileY + tileSize - 1);

                #pragma omp parallel for collapse(2)
                for (int y = localYMin; y <= localYMax; y++) {
                    for (int x = localXMin; x <= localXMax; x++) {
                        float lambda1, lambda2, lambda3;
                        computeBarycentricCoords(v0, v1, v2, Vec2(x + 0.5f, y + 0.5f),
                            lambda1, lambda2, lambda3);

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
                                    attr = perspectiveCorrectInterpolate(v0_attr, v1_attr, v2_attr,
                                        lambda1, lambda2, lambda3,
                                        v2fs[idx0].gl_ZCamera,
                                        v2fs[idx1].gl_ZCamera,
                                        v2fs[idx2].gl_ZCamera);
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

    }
}