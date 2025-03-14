#pragma once

#include <chrono>
#include "../material/depthmap.h"
#include "../material/gbuffer.h"
#include "../material/ssao.h"
#include "../material/ssr.h"
#include "../material/fxaa.h"
#include "../material/standard.h"
#include "../procedural_texture.h"

inline void box(bool use4xSSAA) {
    uint32_t SCREEN_WIDTH  = use4xSSAA ? 4 * 960 : 960;
    uint32_t SCREEN_HEIGHT = use4xSSAA ? 4 * 540 : 540;

    if (use4xSSAA)
        std::clog << "4x SSAA: On\n";  
    else
        std::clog << "4x SSAA: Off\n";
    
    auto start = std::chrono::high_resolution_clock::now();

    // Application Set Up
    Renderer renderer;
    FrameBuffer depth_map(1024, 1024);
    FrameBuffer g_buffer(SCREEN_WIDTH, SCREEN_HEIGHT);
    FrameBuffer ao(SCREEN_WIDTH, SCREEN_HEIGHT);
    FrameBuffer ao_final(SCREEN_WIDTH, SCREEN_HEIGHT);
    FrameBuffer frame(SCREEN_WIDTH, SCREEN_HEIGHT);
    std::vector<Vertex> vertex_buffer;
    Vec3 light_position(0, 8.4f, -5.49f);
    PointLight light{ 300, 20, Vec3(1, 1, 1), light_position};
    Vec3 camera_position(0, 6.22f, -8.03f);
    Vec3 lookat_position(0, 0, 5);
    
    std::vector<float> sphere_vertices;
    std::vector<uint32_t> sphere_indices;
    generateSphere(1, 40, 40, sphere_vertices, sphere_indices);

    ColorBuffer star(256, 256);
    generateStarTexture(star, 128);
    ColorBuffer checkerborad(256, 256);
    generateCheckerboradTexture(checkerborad, 20);
    ColorBuffer wood(512, 512);
    generateWoodTexture(wood, 50.0f, 0.6f, 1.0f, 0.0f);
    ColorBuffer normal_map(512, 512);
    generateWaveNormalMap(normal_map, 30, 0.15f);

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> serial_time = end - start;
    std::clog << "Application Set Up: " << serial_time.count() << " seconds\n";

    // Shadow Pass : Cube Right
    for (uint32_t i = 0; i < cube_vertices.size(); i += 8) {
        Vertex v;
        v.position = Vec3(cube_vertices[i], cube_vertices[i + 1], cube_vertices[i + 2]);
        v.normal = Vec3(cube_vertices[i + 3], cube_vertices[i + 4], cube_vertices[i + 5]);
        v.uv = Vec2(cube_vertices[i + 6], cube_vertices[i + 7]);
        vertex_buffer.push_back(v);
    }

    DepthMapMaterial cube_depth;
    cube_depth.model = createModelMatrix(Vec3(1.82f, 1.5f, -0.04f), Vec3(0, 1, 0), 15, Vec3(1.5f));
    cube_depth.lightSpaceMatrix = createPerspectiveMatrix(90, 1, 0.1f, 100) * createLookAtMatrix(light_position, Vec3(0, 0, 5));
    cube_depth.albedo = Vec3(1);
    cube_depth.light = light;

    renderer.vertex_buffer = std::move(vertex_buffer);
    renderer.index_buffer = cube_indices;
    renderer.cameraPos = light_position;
    renderer.render(cube_depth, depth_map);

    // Shadow Pass : Cube Left
    for (uint32_t i = 0; i < cube_vertices.size(); i += 8) {
        Vertex v;
        v.position = Vec3(cube_vertices[i], cube_vertices[i + 1], cube_vertices[i + 2]);
        v.normal = Vec3(cube_vertices[i + 3], cube_vertices[i + 4], cube_vertices[i + 5]);
        v.uv = Vec2(cube_vertices[i + 6], cube_vertices[i + 7]);
        vertex_buffer.push_back(v);
    }

    cube_depth.model = createModelMatrix(Vec3(-2.35f, 3, 1.86f), Vec3(0, 1, 0), -15, Vec3(1.5f, 3, 1.5f));
    cube_depth.lightSpaceMatrix = createPerspectiveMatrix(90, 1, 0.1f, 100) * createLookAtMatrix(light_position, Vec3(0, 0, 5));
    cube_depth.albedo = Vec3(1);
    cube_depth.light = light;

    renderer.vertex_buffer = std::move(vertex_buffer);
    renderer.index_buffer = cube_indices;
    renderer.render(cube_depth, depth_map);

    // Shadow Pass : Quad_Down
    for (uint32_t i = 0; i < quad_vertices.size(); i += 5) {
        Vertex v;
        v.position = Vec3(quad_vertices[i], quad_vertices[i + 1], quad_vertices[i + 2]);
        v.uv = Vec2(quad_vertices[i + 3], quad_vertices[i + 4]);
        v.normal = Vec3(0, 0, -1);
        vertex_buffer.push_back(v);
    }

    DepthMapMaterial quad_depth;
    quad_depth.model = createModelMatrix(Vec3(0), Vec3(1, 0, 0), 90, Vec3(5, 5, 1));
    quad_depth.lightSpaceMatrix = createPerspectiveMatrix(90, 1, 0.1f, 100) * createLookAtMatrix(light_position, Vec3(0, 0, 5));
    quad_depth.albedo = Vec3(0.94f, 1, 0);
    quad_depth.light = light;
    //quad_depth.useAlbedoMap = true;
    quad_depth.albedoMap = &wood;

    renderer.vertex_buffer = vertex_buffer;
    renderer.index_buffer = quad_indices;
    renderer.render(quad_depth, depth_map);
    quad_depth.useAlbedoMap = false;

    // Shadow Pass : Quad_Back
    quad_depth.model = createModelMatrix(Vec3(0, 5, 5), Vec3(1, 0, 0), 0, Vec3(5, 5, 1));
    quad_depth.albedo = Vec3(0.65f, 0.05f, 0.05f);

    renderer.vertex_buffer = vertex_buffer;
    renderer.index_buffer = quad_indices;
    renderer.render(quad_depth, depth_map);

    // Shadow Pass : Quad_Left
    quad_depth.model = createModelMatrix(Vec3(-5, 5, 0), Vec3(0, 1, 0), -90, Vec3(5, 5, 1));
    quad_depth.albedo = Vec3(0.12f, 0.45f, 0.15f);

    renderer.writeDepth = false;
    renderer.vertex_buffer = vertex_buffer;
    renderer.index_buffer = quad_indices;
    renderer.render(quad_depth, depth_map);
    renderer.writeDepth = true;

    // Shadow Pass : Quad_Right
    quad_depth.model = createModelMatrix(Vec3(5, 5, 0), Vec3(0, 1, 0), 90, Vec3(5, 5, 1));
    quad_depth.albedo = Vec3(0, (float)249 / 255, 1);

    renderer.vertex_buffer = std::move(vertex_buffer);
    renderer.index_buffer = quad_indices;
    renderer.render(quad_depth, depth_map);

    end = std::chrono::high_resolution_clock::now();
    serial_time = end - start;
    std::clog << "Shadow Passes:      " << serial_time.count() << " seconds\n";

    // Geometry Pass : Cube_Right
    for (uint32_t i = 0; i < cube_vertices.size(); i += 8) {
        Vertex v;
        v.position = Vec3(cube_vertices[i], cube_vertices[i + 1], cube_vertices[i + 2]);
        v.normal = Vec3(cube_vertices[i + 3], cube_vertices[i + 4], cube_vertices[i + 5]);
        v.uv = Vec2(cube_vertices[i + 6], cube_vertices[i + 7]);
        vertex_buffer.push_back(v);
    }

    GBufferMaterial cube_gbuffer;
    cube_gbuffer.model = createModelMatrix(Vec3(1.82f, 1.5f, -0.04f), Vec3(0, 1, 0), 15, Vec3(1.5f));
    cube_gbuffer.view = createLookAtMatrix(camera_position, lookat_position);
    cube_gbuffer.projection = createPerspectiveMatrix(60, (float)SCREEN_WIDTH / SCREEN_HEIGHT, 0.1f, 100);
    cube_gbuffer.albedo = Vec3(1);
    cube_gbuffer.emissive = Vec3(0);
    cube_gbuffer.metallic = 0.5f;
    cube_gbuffer.smoothness = 0.8f;
    cube_gbuffer.objectID = 1;

    renderer.vertex_buffer = std::move(vertex_buffer);
    renderer.index_buffer = cube_indices;
    renderer.cameraPos = camera_position;
    renderer.render(cube_gbuffer, g_buffer);

    // Geometry Pass : Cube
    for (uint32_t i = 0; i < cube_vertices.size(); i += 8) {
        Vertex v;
        v.position = Vec3(cube_vertices[i], cube_vertices[i + 1], cube_vertices[i + 2]);
        v.normal = Vec3(cube_vertices[i + 3], cube_vertices[i + 4], cube_vertices[i + 5]);
        v.uv = Vec2(cube_vertices[i + 6], cube_vertices[i + 7]);
        vertex_buffer.push_back(v);
    }

    cube_gbuffer.model = createModelMatrix(Vec3(-2.35f, 3, 1.86f), Vec3(0, 1, 0), -15, Vec3(1.5f, 3, 1.5f));
    cube_gbuffer.view = createLookAtMatrix(camera_position, lookat_position);
    cube_gbuffer.projection = createPerspectiveMatrix(60, (float)SCREEN_WIDTH / SCREEN_HEIGHT, 0.1f, 100);
    cube_gbuffer.albedo = Vec3(1);
    cube_gbuffer.emissive = Vec3(0);
    cube_gbuffer.metallic = 0.5f;
    cube_gbuffer.smoothness = 0.8f;
    cube_gbuffer.objectID = 0;

    renderer.vertex_buffer = std::move(vertex_buffer);
    renderer.index_buffer = cube_indices;
    renderer.render(cube_gbuffer, g_buffer);

    // Geometry Pass : Quad_Down
    for (uint32_t i = 0; i < quad_vertices.size(); i += 5) {
        Vertex v;
        v.position = Vec3(quad_vertices[i], quad_vertices[i + 1], quad_vertices[i + 2]);
        v.uv = Vec2(quad_vertices[i + 3], quad_vertices[i + 4]);
        v.normal = Vec3(0, 0, -1);
        vertex_buffer.push_back(v);
    }

    GBufferMaterial quad_gbuffer;
    quad_gbuffer.model = createModelMatrix(Vec3(0), Vec3(1, 0, 0), 90, Vec3(5, 5, 1));
    quad_gbuffer.view = createLookAtMatrix(camera_position, lookat_position);
    quad_gbuffer.projection = createPerspectiveMatrix(60, (float)SCREEN_WIDTH / SCREEN_HEIGHT, 0.1f, 100);
    quad_gbuffer.albedo = Vec3(0.94f, 1, 0);
    quad_gbuffer.emissive = Vec3(0);
    quad_gbuffer.metallic = 0.5f;
    quad_gbuffer.smoothness = 0;
    quad_gbuffer.objectID = 2;
    //quad_gbuffer.useNormalMap = true;
    quad_gbuffer.normalMap = &normal_map;

    renderer.vertex_buffer = vertex_buffer;
    renderer.index_buffer = quad_indices;
    renderer.render(quad_gbuffer, g_buffer);
    quad_gbuffer.useNormalMap = false;

    // Geometry Pass : Quad_Back
    quad_gbuffer.model = createModelMatrix(Vec3(0, 5, 5), Vec3(1, 0, 0), 0, Vec3(5, 5, 1));
    quad_gbuffer.albedo = Vec3(0.65f, 0.05f, 0.05f);
    quad_gbuffer.objectID = 3;

    renderer.vertex_buffer = vertex_buffer;
    renderer.index_buffer = quad_indices;
    renderer.render(quad_gbuffer, g_buffer);

    // Geometry Pass : Quad_Left
    quad_gbuffer.model = createModelMatrix(Vec3(-5, 5, 0), Vec3(0, 1, 0), -90, Vec3(5, 5, 1));
    quad_gbuffer.albedo = Vec3(0.12f, 0.45f, 0.15f);
    quad_gbuffer.objectID = 4;

    renderer.vertex_buffer = vertex_buffer;
    renderer.index_buffer = quad_indices;
    renderer.render(quad_gbuffer, g_buffer);

    // Geometry Pass : Quad_Right
    quad_gbuffer.model = createModelMatrix(Vec3(5, 5, 0), Vec3(0, 1, 0), 90, Vec3(5, 5, 1));
    quad_gbuffer.albedo = Vec3(0, (float)249 / 255, 1);
    quad_gbuffer.objectID = 5;

    renderer.vertex_buffer = vertex_buffer;
    renderer.index_buffer = quad_indices;
    renderer.render(quad_gbuffer, g_buffer);

    end = std::chrono::high_resolution_clock::now();
    serial_time = end - start;
    std::clog << "Geometry Passes:    " << serial_time.count() << " seconds\n";

    // SSAO Pass
    SSAOMaterial ssao;
    ssao.gbuffer = &g_buffer;
    ssao.view = createLookAtMatrix(camera_position, lookat_position);
    ssao.projection = createPerspectiveMatrix(60, (float)SCREEN_WIDTH / SCREEN_HEIGHT, 0.1f, 100);

    renderer.vertex_buffer = vertex_buffer;
    renderer.index_buffer = quad_indices;
    renderer.render(ssao, ao);

    // SSAO-Blur Pass
    SSAOBlurMaterial ssao_blur;
    ssao_blur.ssao = &ao;
    ssao_blur.view = createLookAtMatrix(camera_position, lookat_position);

    renderer.vertex_buffer = vertex_buffer;
    renderer.index_buffer = quad_indices;
    renderer.render(ssao_blur, ao_final);

    end = std::chrono::high_resolution_clock::now();
    serial_time = end - start;
    std::clog << "SSAO Passes:        " << serial_time.count() << " seconds\n";

    // Shading Pass
    Mat4 view = createLookAtMatrix(camera_position, lookat_position);
    Mat4 inv_view = view;
    inv_view.inverse();

    StandardMaterial standard_shading(0);
    standard_shading.albedoMap[0] = &star;         // cube left
    standard_shading.albedoMap[1] = &checkerborad; // cube right
    //standard_shading.albedoMap[2] = &wood;         // quad down
    standard_shading.gbuffer = &g_buffer;
    standard_shading.ssao = &ao_final;
    standard_shading.cameraPos = camera_position;
    standard_shading.view = view;
    standard_shading.invView = inv_view;
    standard_shading.lightSpaceMatrix = createPerspectiveMatrix(90, 1, 0.1f, 100) * createLookAtMatrix(light_position, Vec3(0, 0, 5));
    standard_shading.depthMaps.push_back(&depth_map);
    standard_shading.pointLights.push_back(light);
    standard_shading.shadowBias = 0.3f;
    
    renderer.vertex_buffer = vertex_buffer;
    renderer.index_buffer = quad_indices;
    renderer.render(standard_shading, frame);

    end = std::chrono::high_resolution_clock::now();
    serial_time = end - start;
    std::clog << "Shading Pass:       " << serial_time.count() << " seconds\n";

    // SSR Pass
    Mat4 proj = createPerspectiveMatrix(60, (float)SCREEN_WIDTH / SCREEN_HEIGHT, 0.1f, 100);
    Mat4 inv_proj = proj;
    inv_proj.inverse();
    
    SSRMaterial ssr;
    ssr.frame = &frame;
    ssr.gbuffer = &g_buffer;
    ssr.floorID = 2; // quad down
    ssr.view = view;
    ssr.invView = inv_view;
    ssr.proj = proj;
    ssr.invProj = inv_proj;
    ssr.cameraPos = camera_position;

    renderer.vertex_buffer = vertex_buffer;
    renderer.index_buffer = quad_indices;
    renderer.render(ssr, frame);

    end = std::chrono::high_resolution_clock::now();
    serial_time = end - start;
    std::clog << "SSR Pass:           " << serial_time.count() << " seconds\n";

    if (!use4xSSAA) {
        FXAAMaterial fxaa;
        fxaa.frame = &(frame.colorBuffers[0]);
        fxaa.contrastThreshold = 0.005f;
        fxaa.relativeThreshold = 0.0f;

        renderer.vertex_buffer = vertex_buffer;
        renderer.index_buffer = quad_indices;
        renderer.render(fxaa, frame);
    
        end = std::chrono::high_resolution_clock::now();
        serial_time = end - start;
        std::clog << "FXAA Pass:          " << serial_time.count() << " seconds\n";
    }

    // Image Output
    if (use4xSSAA) {
        gaussianFilterNTimes(frame.colorBuffers[0], 3);
        nearestScaling(frame.colorBuffers[0], 0.25f, 0.25f);

        end = std::chrono::high_resolution_clock::now();
        serial_time = end - start;
        std::clog << "DownSampling:       " << serial_time.count() << " seconds\n";
    }
    
    displayImage(frame.colorBuffers[0]);
}