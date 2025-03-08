#pragma once

#include <iostream>
#include "framebuffer.h"

inline void generateCheckerboradTexture(ColorBuffer& texture, uint32_t checkSize) {
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

inline void generateStarTexture(ColorBuffer& texture, uint32_t starSize) {
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

inline void generateCircleTexture(ColorBuffer& texture, uint32_t radius) {
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

inline float randomNoise(float x, float y) {
    float value = std::sin(x * 12.9898f + y * 78.233f) * 43758.5453f;
    return value - std::floor(value);
}

inline void generateWoodTexture(ColorBuffer& texture, float ringSpacing, float noiseStrength, float directionX, float directionY) {
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

inline void generateWaveNormalMap(ColorBuffer& texture, float waveScale, float amplitude) {
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