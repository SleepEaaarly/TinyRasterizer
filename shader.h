#ifndef SHADER_H
#define SHADER_H

#include "texture.h"
#include "primitive.h"
#include "light.h"
#include <cmath>

Vec4f Uchar2Float(Color c) {
    return Vec4f(c.r/255., c.g/255., c.b/255., c.a/255.);
}

Color Float2Uchar(Vec4f f) {
    float x, y, z, w;
    x = std::max(0.f, std::min(1.f, f.x));
    y = std::max(0.f, std::min(1.f, f.y));
    z = std::max(0.f, std::min(1.f, f.z));
    w = std::max(0.f, std::min(1.f, f.w));
    return Color((unsigned char)(x*255+.49f), (unsigned char)(y*255+.49f), (unsigned char)(z*255+.49f), (unsigned char)(w*255+.49f));
}

class Shader
{
public:
    Shader(){}
    ~Shader(){}
    void setTexture(Texture &texture);
    virtual Color shadeFragment(Fragment &frag);
    // Color shadeNormal();
};

Color Shader::shadeFragment(Fragment &frag) {
    // return Color(255, 0, 0, 255);
    return Color(rand()%255, rand()%255, rand()%255, 255);
}

class TextureShader : public Shader {
protected:
    Texture *texture;
public:
    TextureShader(Texture &tex) {
        setTexture(tex);
    }
    void setTexture(Texture &tex);
    Color shadeFragment(Fragment &frag) override;
};

void TextureShader::setTexture(Texture &tex) {
    texture = &tex;
}

Color TextureShader::shadeFragment(Fragment &frag) {
    return texture->get_color(frag.tex[0], frag.tex[1]);
}


class PhongShader : public Shader {     // Blinn-Phong
protected:
    Texture *texture;
    Light *light;
public:
    PhongShader() {}
    PhongShader(Texture &tex, Light &l) {
        setTexture(tex);
        setLight(l);
    }

    void setLight(Light &l);
    void setTexture(Texture &texture);
    Color shadeFragment(Fragment &frag) override;
};

void PhongShader::setTexture(Texture &tex) {
    texture = &tex;
}

void PhongShader::setLight(Light &l) {
    light = &l;
}

Color PhongShader::shadeFragment(Fragment &frag) {
    Vec3f ka = Vec3f(0.005, 0.005, 0.005);
    Vec3f kd = Uchar2Float(texture->get_color(frag.tex[0], frag.tex[1])).value();
    Vec3f ks = Vec3f(0.5, 0.5, 0.5);

    Vec3f light_intensity = light->color;
    Vec3f ambient(ka.x*light_intensity.x, ka.y*light_intensity.y, ka.z*light_intensity.z);
    
    Vec3f vec_l = -1 * light->dir;
    float diff = std::max(0.f, vec_l*frag.norm);
    Vec3f diffuse(kd.x*light_intensity.x, kd.y*light_intensity.y, kd.z*light_intensity.z);
    diffuse = diffuse * diff;

    // view coords eye_pos = Vec3f(0.f,0.f,0.f)
    float p = 128.f;
    Vec3f vec_v = -1 * frag.pos_view;
    Vec3f vec_h = (vec_l + vec_v).normalize();
    float spec = std::max(0.f, std::pow(frag.norm*vec_h, p));
    Vec3f specular(ks.x*light_intensity.x, ks.y*light_intensity.y, ks.z*light_intensity.z);
    specular = specular * spec;

    Vec3f all_shade = ambient+diffuse+specular;

    return Float2Uchar(Vec4f(all_shade, 1.0f));
}

#endif