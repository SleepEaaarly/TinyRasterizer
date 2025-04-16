#ifndef PRIMITIVE_CUDA_CUH
#define PRIMITIVE_CUDA_CUH

struct Vec2f_cuda {
    float x, y;
    __device__ Vec2f_cuda() {}
    __device__ Vec2f_cuda(float x_in, float y_in) { x=x_in; y=y_in; }
};

struct Vec3f_cuda {
    float x, y, z;
    __device__ Vec3f_cuda() {}
    __device__ Vec3f_cuda(float x_in, float y_in, float z_in) { x=x_in; y=y_in; z=z_in; }
    __device__ Vec3f_cuda normalize() { 
        float rev_norm = rsqrtf(x*x+y*y+z*z);
        return Vec3f_cuda(x*rev_norm, y*rev_norm, z*rev_norm);
    }  
};

struct Vec4f_cuda {
    float x, y, z, w;
    __device__ Vec4f_cuda() {}
    __device__ Vec4f_cuda(float x_in, float y_in, float z_in, float w_in) { x=x_in; y=y_in; z=z_in; w=w_in; }
    __device__ Vec3f_cuda value() { return Vec3f_cuda(x, y, z); }
};

struct Vert_cuda {
    Vec4f_cuda pos;
    Vec3f_cuda norm;
    Vec2f_cuda tex;
    Vec3f_cuda pos_view;
    __device__ Vert_cuda() {}
};

struct Vec2i_cuda {
    int x;
    int y;
    __device__ Vec2i_cuda() {}
    __device__ Vec2i_cuda(int x_in, int y_in) : x(x_in), y(y_in) {}
};

struct Fragment_cuda {
    Vec2i_cuda pos;
    Vec2f_cuda tex;
    Vec3f_cuda norm;
    Vec3f_cuda pos_view;
    float z;
    __device__ Fragment_cuda() {}
    __device__ Fragment_cuda(Vec2i_cuda &pos, Vec2f_cuda &tex, Vec3f_cuda &norm, Vec3f_cuda &pos_view, float z) {
        this->pos = pos;
        this->tex = tex;
        this->norm = norm;
        this->pos_view = pos_view;
        this->z = z;
    }
};

struct Color_cuda {
	union {
		struct {
			unsigned char b, g, r, a;
		};
		unsigned char raw[4];
		unsigned int val;
	};
    __device__ Color_cuda() : val(0) {
	}

	__device__ Color_cuda(unsigned char R, unsigned char G, unsigned char B, unsigned char A) : b(B), g(G), r(R), a(A) {
	}
};

struct Triangle_cuda {
    Vec3f_cuda poses[3];
    Vec2f_cuda texs[3];
    Vec3f_cuda norms[3];
    Vec3f_cuda poses_view[3];
    __device__ Triangle_cuda() {}
    
};

#endif