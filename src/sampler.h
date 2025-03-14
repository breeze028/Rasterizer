#pragma once

#include "framebuffer.h"

enum class FILTERMODE { POINT, BILINEAR };

inline Vec4 texture2D(ColorBuffer* texture, float u, float v, FILTERMODE filter = FILTERMODE::POINT) {
    int width = texture->width;
    int height = texture->height;

    float u_float = u * (texture->width - 1);
    float v_float = v * (texture->height - 1);

    u_float = clamp(u_float, 0.0f, static_cast<float>(texture->width - 1));
    v_float = clamp(v_float, 0.0f, static_cast<float>(texture->height - 1));

    if (filter == FILTERMODE::POINT) {
        return texture->getColor(u_float, v_float);
    }

    if (filter == FILTERMODE::BILINEAR) {
        int x0 = static_cast<int>(floor(u_float));
        int y0 = static_cast<int>(floor(v_float));
        int x1 = static_cast<int>(ceil(u_float));
        int y1 = static_cast<int>(ceil(v_float));

        x0 = clamp(x0, 0, texture->width - 1);
        x1 = clamp(x1, 0, texture->width - 1);
        y0 = clamp(y0, 0, texture->height - 1);
        y1 = clamp(y1, 0, texture->height - 1);

        float frac_u = u_float - x0;
        float frac_v = v_float - y0;

        Vec4 c00 = texture->getColor(x0, y0);
        Vec4 c10 = texture->getColor(x1, y0);
        Vec4 c01 = texture->getColor(x0, y1);
        Vec4 c11 = texture->getColor(x1, y1);

        Vec4 h0 = c00 * (1.0f - frac_u) + c10 * frac_u;
        Vec4 h1 = c01 * (1.0f - frac_u) + c11 * frac_u;

        Vec4 final_color = h0 * (1.0f - frac_v) + h1 * frac_v;
        return final_color;
    }

    // never gonna happen
    return {};
}

inline Vec4 texture2D(ColorBuffer* texture, Vec2 uv, FILTERMODE filter = FILTERMODE::POINT) {
    return texture2D(texture, uv.x, uv.y, filter);
}