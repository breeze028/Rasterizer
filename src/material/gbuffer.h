#pragma once

#include "common.h"

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
            // uint32_t coord_u = clamp(uv.x * normalMap->width, 0.0f, normalMap->width - 1);
            // uint32_t coord_v = clamp(uv.y * normalMap->height, 0.0f, normalMap->height - 1);
            // Vec3 normal = Vec3(normalMap->getColor(coord_u, coord_v) * 2 - 1).normalize();
            Vec3 normal = Vec3(texture2D(normalMap, uv.x, uv.y, FILTERMODE::BILINEAR));
            normal = (normal * 2 - 1).normalize();
            viewSpaceNormal = (normal2View * normal).normalize();
        }

        Vec4 target0(uv.x, uv.y, objectID, smoothness);
        Vec4 target1(albedo, metallic);
        Vec4 target2(viewSpacePosition, 1);
        Vec4 target3(viewSpaceNormal, 0);
        return { target0, target1, target2, target3 };
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