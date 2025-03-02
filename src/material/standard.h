#pragma once

#include <random>
#include <unordered_map>
#include "common.h"

struct StandardMaterial {
    using V2F = GenericV2F<>;

    StandardMaterial(uint32_t _rsmSampleNums) : u(std::uniform_real_distribution<float>(0, 1)),
        rsmSampleNums(_rsmSampleNums) {
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
        if (albedoMap.count(objectID)) {
            ColorBuffer* albedoTexture = albedoMap[objectID];
            uint32_t coord_u = clamp(u * albedoTexture->width, 0.0f, albedoTexture->width - 1);
            uint32_t coord_v = clamp(v * albedoTexture->height, 0.0f, albedoTexture->height - 1);
            albedo = Vec3(albedoTexture->getColor(coord_u, coord_v));
        }
        else {
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
            if (rsmSampleNums)
                result += computeIndirectLight(worldPos, N, albedo, depthMaps[i]) * ao;
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
    std::unordered_map<uint32_t, ColorBuffer*> albedoMap;
    std::vector<FrameBuffer*> depthMaps;
    std::vector<PointLight> pointLights;
    uint32_t rsmSampleNums;
    float shadowBias;

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
            sum += linearizeDepth(z) - shadowBias > linearizeDepth(pcfDepth) ? 1.0f : 0.0f;
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
            uint32_t x = clamp((uv.x + 0.2 * sampleCoordsAndWeights[i].x) * depthMap->width, 0.0f, static_cast<float>(depthMap->width - 1));
            uint32_t y = clamp((uv.y + 0.2 * sampleCoordsAndWeights[i].y) * depthMap->height, 0.0f, static_cast<float>(depthMap->height - 1));

            Vec3 lightPos = Vec3(depthMap->getColori(0, x, y));
            Vec3 lightNormal = Vec3(depthMap->getColori(1, x, y));
            Vec3 flux = Vec3(depthMap->getColori(2, x, y));
            Vec3 L = (lightPos - worldPos).normalize();
            float NdotL = std::max(N.dot(L), 0.0f);

            float cosP = NdotL;
            float cosQ = std::max(lightNormal.dot(-1 * L), 0.0f);
            float weight = sampleCoordsAndWeights[i].z;
            Vec3 diff = lightPos - worldPos;
            float distSq = diff.dot(diff);
            float invDistSq = 1.0f / distSq;
            float scale = cosP * cosQ * weight * invDistSq;
            indirect += (albedo / PI) * scale * flux;
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