#include "shader.h"

Vec4f Uchar2Float(Color c) {
    return Vec4f(c.r/255., c.g/255., c.b/255., c.a/255.);
}

Color Float2Uchar(Vec4f f) {
    float x, y, z, w;
    x = std::max(0.f, std::min(1.f, f.x));
    y = std::max(0.f, std::min(1.f, f.y));
    z = std::max(0.f, std::min(1.f, f.z));
    w = std::max(0.f, std::min(1.f, f.w));
    return Color((unsigned char)(x*255+.5f), (unsigned char)(y*255+.5f), (unsigned char)(z*255+.5f), (unsigned char)(w*255+.5f));
}


Shader::Shader(){}
Shader::~Shader(){}

TextureShader::TextureShader(Texture &tex) {
    setTexture(tex);
}

void TextureShader::setTexture(Texture &tex) {
    texture = &tex;
}

Color TextureShader::shadeFragment(Fragment &frag) {
    return texture->get_color(frag.tex[0], frag.tex[1]);
}

Texture* TextureShader::getTexturePtr() {
    return texture;
}

Light* TextureShader::getLightPtr() {
    return nullptr;
}

PhongShader::PhongShader(){}
PhongShader::PhongShader(Texture &tex, Light &l) {
    setTexture(tex);
    setLight(l);
}

void PhongShader::setTexture(Texture &tex) {
    texture = &tex;
}

void PhongShader::setLight(Light &l) {
    light = &l;
}

Texture* PhongShader::getTexturePtr() {
    return texture;
}

Light* PhongShader::getLightPtr() {
    return light;
}

Color PhongShader::shadeFragment(Fragment &frag) {
    Vec3f albedo = Uchar2Float(texture->get_color(frag.tex[0], frag.tex[1])).value();
    albedo = Vec3f(powf(albedo.x, 2.2f), powf(albedo.y, 2.2f), powf(albedo.z, 2.2f));
    Vec3f ka = Vec3f(0.8f, 0.8f, 0.8f);
    Vec3f kd = Vec3f(0.2f, 0.2f, 0.2f);
    Vec3f ks = Vec3f(1.f, 1.f, 1.f);

    Vec3f ambient(ka.x*albedo.x, ka.y*albedo.y, ka.z*albedo.z);
    
    Vec3f light_intensity = light->color;    
    Vec3f vec_l = -1 * light->dir;
    float diff = std::max(0.f, vec_l*frag.norm);
    Vec3f diffuse(albedo.x*light_intensity.x*kd.x, albedo.y*light_intensity.y*kd.y, albedo.z*light_intensity.z*kd.z);
    diffuse = diffuse * diff;

    // view coords eye_pos = Vec3f(0.f,0.f,0.f)
    float p = 128.f;
    Vec3f vec_v = -1 * frag.pos_view;
    Vec3f vec_h = (vec_l + vec_v).normalize();
    float spec = std::max(0.f, std::powf(frag.norm*vec_h, p));
    // std::cout << spec << std::endl;
    Vec3f specular(ks.x*light_intensity.x, ks.y*light_intensity.y, ks.z*light_intensity.z);
    specular = specular * spec;

    Vec3f all_shade = ambient+diffuse+specular;
    all_shade = Vec3f(powf(all_shade.x, 0.45f), powf(all_shade.y, 0.45f), powf(all_shade.z, 0.45f));
    
    return Float2Uchar(Vec4f(all_shade, 1.0f));
}
