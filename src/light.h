#pragma once

#include "math.h"

struct PointLight {
    float intensity;
    float radius;
    Vec3 color;
    Vec3 position;
};
