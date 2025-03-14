#pragma once

#include <random>
#include "common.h"

struct SSAOMaterial {
    // 0: uv
    using V2F = GenericV2F<Vec2>;

    SSAOMaterial() : randomFloats(std::uniform_real_distribution<float>(0.0f, 1.0f)) {
        generateSampleKernel();
        generateNoise();
    }

    V2F vert(const Vertex& i) {
        V2F o;
        o.gl_Position = Vec4(i.position, 1);
        o.gl_ZCamera = 1;

        // Vec2 uv
        std::get<0>(o.attributes) = i.uv;
        return o;
    }

    std::array<Vec4, 4> frag(const V2F& fs_in) {
        uint32_t kernelSize = 64;
        float radius = 0.5f;
        float bias = 0.0f;

        Vec2 uv = std::get<0>(fs_in.attributes);
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

            float sampleDepth = texture2D(&(gbuffer->colorBuffers[2]), ndc.x, ndc.y, FILTERMODE::BILINEAR).z;
            float rangeCheck = smoothstep(0.0f, 1.0f, radius / (abs(fragPos.z - sampleDepth) + 1e-7));
            occlusion += (sampleDepth <= samplePos.z + bias ? 1.0f : 0.0f) * rangeCheck;
        }
        occlusion = 1 - occlusion / kernelSize;
        return { Vec4(occlusion), Vec4(), Vec4(), Vec4() };
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
        return { Vec4(result / 16), Vec4(), Vec4(), Vec4() };
    }

public:
    Mat4 view;
    FrameBuffer* ssao;

    Mat4 model; // Identity matrix, for back-face culling, actually useless
};