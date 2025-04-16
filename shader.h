#ifndef SHADER_H
#define SHADER_H

#include "texture.h"
#include "primitive.h"
#include "light.h"
#include <cmath>

class Shader
{
public:
    Shader();
    ~Shader();
    virtual void setTexture(Texture &texture) = 0;
    virtual Color shadeFragment(Fragment &frag) = 0;
    virtual Texture* getTexturePtr() = 0;
    virtual Light* getLightPtr() = 0;
    // Color shadeNormal();
};


class TextureShader : public Shader {
protected:
    Texture *texture;
public:
    TextureShader(Texture &tex);
    void setTexture(Texture &tex) override;
    Color shadeFragment(Fragment &frag) override;
    Texture* getTexturePtr() override;
    Light* getLightPtr() override;
};


class PhongShader : public Shader {     // Blinn-Phong
protected:
    Texture *texture;
    Light *light;
public:
    PhongShader();
    PhongShader(Texture &tex, Light &l);

    void setLight(Light &l);
    void setTexture(Texture &texture) override;
    Color shadeFragment(Fragment &frag) override;
    Texture* getTexturePtr() override;
    Light* getLightPtr() override;
};

#endif