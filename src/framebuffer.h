#pragma once

#include <vector>
#include "math.h"

struct ColorBuffer {
    ColorBuffer(uint32_t width, uint32_t height, const Vec4& colorDefault = Vec4(0.0f)) : width(width), height(height) {
        m_colorBuffer.resize(height, std::vector<Vec4>(width, colorDefault));
    }

    Vec4& getColor(uint32_t x, uint32_t y) {
        return m_colorBuffer[height - y - 1][x];
    }

    void resize(uint32_t width, uint32_t height, const Vec4& colorDefault = Vec4(0.0f)) {
        this->width = width;
        this->height = height;
        m_colorBuffer.clear();
        m_colorBuffer.resize(height, std::vector<Vec4>(width, colorDefault));
    }
public:
    uint32_t width;
    uint32_t height;
private:
    std::vector<std::vector<Vec4>> m_colorBuffer;
};

struct DepthBuffer {
    DepthBuffer(uint32_t width, uint32_t height, float depthDefault = 1.0f) : width(width), height(height) {
        m_depthBuffer.resize(height, std::vector<float>(width, depthDefault));
    }

    float& getDepth(uint32_t x, uint32_t y) {
        return m_depthBuffer[height - y - 1][x];
    }
public:
    uint32_t width;
    uint32_t height;
private:
    std::vector<std::vector<float>> m_depthBuffer;
};

struct FrameBuffer {
    FrameBuffer(uint32_t width, uint32_t height, const Vec4& colorDefault = Vec4(0), float depthDefault = 1.0f)
        : width(width), height(height),
        colorBuffers{ ColorBuffer(width, height, colorDefault), ColorBuffer(width, height, colorDefault),
                      ColorBuffer(width, height, colorDefault), ColorBuffer(width, height, colorDefault) },
        depthBuffer(width, height, depthDefault) {}

    Vec4& getColori(uint32_t i, uint32_t x, uint32_t y) {
        return colorBuffers[i].getColor(x, y);
    }

    float& getDepth(uint32_t x, uint32_t y) {
        return depthBuffer.getDepth(x, y);
    }
public:
    uint32_t width;
    uint32_t height;
    ColorBuffer colorBuffers[4];
    DepthBuffer depthBuffer;
};