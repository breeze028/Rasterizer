#pragma once

#include <random>
#include "common.h"

// https://github.com/RoundedGlint585/ScreenSpaceReflection/blob/master/shaders/SSRFragment.glsl

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

        Vec3 position(gbuffer->getColori(2, x, y));
        // exponential fog
        Vec3 worldPos = Vec3(invView * Vec4(position, 1));
        float distance = (worldPos - cameraPos).length();
        float distanceFactor = 1 - exp(-fogDensity * distance);
        float heightFactor = clamp(exp(-fogDensity * (worldPos.y + fogHmin)), 0.0f, 0.6f);
        float fogFactor = distanceFactor * heightFactor;

        if (!enableSSR || objectID != floorID) {
            target = frame->getColori(0, x, y);
            //target = lerp(target, fogColor, fogFactor);
            return { target, Vec4(), Vec4(), Vec4() };
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
        }
        else {
            SSRColor = trace(position, reflection);
        }

        target = lerp(SSRColor, frame->getColori(0, x, y), 0.9f);       
        //target = lerp(target, fogColor, fogFactor);
        return { target, Vec4(), Vec4(), Vec4() };
    }

public:
    FrameBuffer* frame;
    FrameBuffer* gbuffer;

    Mat4 proj;
    Mat4 invProj;
    Mat4 view;
    Mat4 invView;
    Vec3 cameraPos;

    uint32_t floorID;
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
