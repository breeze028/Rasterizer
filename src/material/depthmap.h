#pragma once

#include "common.h"

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
        return { target0, target1, target2, Vec4() };
    }
public:
    Mat4 lightSpaceMatrix;
    Mat4 model;
    Vec3 albedo;
    bool useAlbedoMap = false;
    ColorBuffer* albedoMap;
    PointLight light;
};
