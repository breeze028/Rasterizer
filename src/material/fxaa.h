#pragma once

#include <algorithm>
#include "common.h"

// https://github.com/Raphael2048/AntiAliasing/blob/main/Assets/Shaders/FXAA/FXAASelf.shader

struct FXAAMaterial {
    // 0: uv
    using V2F = GenericV2F<Vec2>;

    V2F vert(const Vertex& i) {
        V2F o;
        o.gl_Position = Vec4(i.position, 1);
        o.gl_ZCamera = 1;

        // Vec2 uv
        std::get<0>(o.attributes) = i.uv;
        return o;
    }

    std::array<Vec4, 4> frag(const V2F& fs_in) {
        Vec2 uv = std::get<0>(fs_in.attributes);
        Vec2 texelSize(1.0f / frame->width, 1.0f / frame->height);
        Vec4 origin = texture2D(frame, uv, FILTERMODE::BILINEAR);
        float M  = luminance(origin);
        float E  = luminance(texture2D(frame, uv + Vec2( texelSize.x,            0), FILTERMODE::BILINEAR));
        float N  = luminance(texture2D(frame, uv + Vec2(           0,  texelSize.y), FILTERMODE::BILINEAR));
        float W  = luminance(texture2D(frame, uv + Vec2(-texelSize.x,            0), FILTERMODE::BILINEAR));
        float S  = luminance(texture2D(frame, uv + Vec2(           0, -texelSize.y), FILTERMODE::BILINEAR));
        float NW = luminance(texture2D(frame, uv + Vec2(-texelSize.x,  texelSize.y), FILTERMODE::BILINEAR));
        float NE = luminance(texture2D(frame, uv + Vec2( texelSize.x,  texelSize.y), FILTERMODE::BILINEAR));
        float SW = luminance(texture2D(frame, uv + Vec2(-texelSize.x, -texelSize.y), FILTERMODE::BILINEAR));
        float SE = luminance(texture2D(frame, uv + Vec2( texelSize.x, -texelSize.y), FILTERMODE::BILINEAR));

        // calculate the contrast
        float maxLuma = std::max(std::max(std::max(N, E), std::max(W, S)), M);
        float minLuma = std::min(std::min(std::min(N, E), std::min(W, S)), M);
        float contrast = std::max(maxLuma - minLuma, 1e-5f);

        //if (contrast < std::max(contrastThreshold, maxLuma * relativeThreshold)) {
        //    return { origin, Vec4(), Vec4(), Vec4() };
        //}

        // jaggy's direction
        float vertical = abs(N + S - 2 * M) * 2 + abs(NE + SE - 2 * E) + abs(NW + SW - 2 * W);
        float horizontal = abs(E + W - 2 * M) * 2 + abs(NE + NW - 2 * N) + abs(SE + SW - 2 * S);
        bool isHorizontal = vertical > horizontal;

        // blending direction
        Vec2 pixelStep = isHorizontal ? Vec2(0, texelSize.y) : Vec2(texelSize.x, 0);

        float positive = abs((isHorizontal ? N : E) - M);
        float negative = abs((isHorizontal ? S : W) - M);

        float gradient, oppositeLuminance;
        if (positive > negative) {
            gradient = positive;
            oppositeLuminance = isHorizontal ? N : E;
        } else {
            pixelStep = -pixelStep;
            gradient = negative;
            oppositeLuminance = isHorizontal ? S : W;
        }

        // blending coefficient based on luminance
        float filter = 2 * (N + E + S + W) + NE + NW + SE + SW;
        filter /= 12;
        filter = abs(filter - M);
        filter = saturate(filter / contrast);
        float pixelBlend = smoothstep(0, 1, filter);
        pixelBlend = pixelBlend * pixelBlend;

        // blending coefficient based on edge
        Vec2 uvEdge = uv;
        uvEdge = uvEdge + pixelStep * 0.5f;
        Vec2 edgeStep = isHorizontal ? Vec2(texelSize.x, 0) : Vec2(0, texelSize.y);

        const int searchSteps = 15;
        const int guess = 8;

        float edgeLuminance = (M + oppositeLuminance) * 0.5f;
        float gradientThreshold = gradient * 0.25f;
        float pLuminanceDelta, nLuminanceDelta, pDistance, nDistance;
        int i;
        for (i = 1; i <= searchSteps; i++) {
            pLuminanceDelta = luminance(texture2D(frame, uvEdge + i * edgeStep, FILTERMODE::BILINEAR)) - edgeLuminance;
            if (abs(pLuminanceDelta) > gradientThreshold) {
                pDistance = i * (isHorizontal ? edgeStep.x : edgeStep.y);
                break;
            }
        }
        if (i == searchSteps + 1) {
            pDistance = guess * (isHorizontal ? edgeStep.x : edgeStep.y);
        }

        for (i = 1; i <= searchSteps; i++) {
            nLuminanceDelta = luminance(texture2D(frame, uvEdge - i * edgeStep, FILTERMODE::BILINEAR)) - edgeLuminance;
            if (abs(nLuminanceDelta) > gradientThreshold) {
                nDistance = i * (isHorizontal ? edgeStep.x : edgeStep.y);
                break;
            }
        }
        if (i == searchSteps + 1) {
            nDistance = guess * (isHorizontal ? edgeStep.x : edgeStep.y);
        }

        float edgeBlend;
        if (pDistance < nDistance) {
            if(pLuminanceDelta > 0 == (M - edgeLuminance) > 0) {
                edgeBlend = 0;
            } else {
                edgeBlend = 0.5f - pDistance / (pDistance + nDistance);
            }
        } else {
            if(nLuminanceDelta > 0 == (M - edgeLuminance) > 0) {
                edgeBlend = 0;
            } else {
                edgeBlend = 0.5f - nDistance / (pDistance + nDistance);
            }
        }

        float finalBlend = std::max(pixelBlend, edgeBlend);
        Vec4 result = texture2D(frame, uv + pixelStep * finalBlend, FILTERMODE::BILINEAR);
        return { result, Vec4(), Vec4(), Vec4() };
    }

public:
    ColorBuffer* frame;
    float contrastThreshold;
    float relativeThreshold;

    Mat4 model; // Identity matrix, for back-face culling, actually useless
private:
    float luminance(Vec4 rgba) {
        return 0.213f * rgba.x + 0.715f * rgba.y + 0.072f * rgba.z;
    }
};