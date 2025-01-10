#define GOOGLE_CUDA

#ifdef GOOGLE_CUDA
#define EIGEN_USE_GPU

#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/core/util/gpu_kernel_helper.h"
#include "tensorflow/core/util/gpu_launch_config.h"
#include "cuda_runtime.h"
#include "vector_types.h"

using namespace tensorflow;

using GPUDevice = Eigen::GpuDevice;

#define IMAD(a, b, c) (__mul24((a), (b)) + (c))
#define JS_EP 1e-5f

inline int iDivUp(int a, int b)
{
    return ((a % b) != 0) ? (a / b + 1) : (a / b);
}

__device__ float fracf(float x)
{
    return x - floorf(x);
}

__device__ int2 operator<(const int2 &a, const int2 &b)
{
    return make_int2(a.x < b.x, a.y < b.y);
}

__device__ int2 operator>(const int2 &a, const int2 &b)
{
    return make_int2(a.x > b.x, a.y > b.y);
}

__device__ int2 operator+(const int2 &a, const int2 &b)
{
    return make_int2(a.x + b.x, a.y + b.y);
}

__device__ int2 operator-(const int2 &a, const int2 &b)
{
    return make_int2(a.x - b.x, a.y - b.y);
}

__device__ float2 operator*(const float2 &a, const int2 &b)
{
    return make_float2(a.x * b.x, a.y * b.y);
}

__device__ float2 operator+(const float2 &a, const float2 &b)
{
    return make_float2(a.x + b.x, a.y + b.y);
}

__device__ float4 operator-(float4 a, float4 b) {
    return make_float4(a.x - b.x, a.y - b.y, a.z - b.z,  a.w - b.w);
}

__device__ float4 operator*(float4 a, float4 b)
{
    return make_float4(a.x * b.x, a.y * b.y, a.z * b.z,  a.w * b.w);
}

__device__ float4 operator*(float4 a, float b) {
    return make_float4(a.x * b, a.y * b, a.z * b,  a.w * b);
}

__device__ float4 operator*(float b, float4 a) {
    return make_float4(b * a.x, b * a.y, b * a.z, b * a.w);
}

__device__ void operator+=(float4 &a, float4 b) {
    a.x += b.x;
    a.y += b.y;
    a.z += b.z;
    a.w += b.w;
}

__device__ float4 operator/(float4 a, float4 b) {
    return make_float4(a.x / b.x, a.y / b.y, a.z / b.z,  a.w / b.w);
}

__device__ float4 operator/(float4 a, float b) {
    return make_float4(a.x / b, a.y / b, a.z / b,  a.w / b);
}

__device__ int2 float2int(const float2 in)
{
    return make_int2(in.x, in.y);
}

__device__ float2 floor(const float2 a)
{
    return make_float2(floor(a.x), floor(a.y));
}

__device__ bool any(const int2 &a)
{
    return a.x || a.y;
}

__device__ float dot(const float3 &a, const float3 &b)
{
    return (a.x * b.x + a.y * b.y + a.z * b.z);
}

__device__ float Dot(const float4& a, const float4& b) 
{
	return (a.x * b.x + a.y * b.y + a.z * b.z);
}

__device__ float3 operator-(const float3 &a, const float3 &b)
{
    return make_float3(a.x - b.x, a.y - b.y, a.z - b.z);
}

__device__ float4 fmaxf(float4 a, float4 b) 
{
    return make_float4(fmaxf(a.x,b.x), fmaxf(a.y,b.y), fmaxf(a.z,b.z), fmaxf(a.w,b.w));
}

__forceinline__ __device__ float Avg(const float4 &a)
{
    return (a.x + a.y + a.z) / 3.f;
}

__device__ float distance(const float3 &a, const float3 &b)
{
    return sqrtf(dot((a - b), (a - b)));
}

__device__ float luminance(const float4 &rgb)
{
    return dot(make_float3(rgb.x, rgb.y, rgb.z), make_float3(0.2126f, 0.7152f, 0.0722f));
}

__device__ float norm2(const float4 &a)
{
    return (a.x)*(a.x) + (a.y)*(a.y) + (a.z)*(a.z);
}

__device__ bool isReprojValid(const int2 &imageDim, const int2 &coord,                                  //
                              const float &Z, const float &Zprev, const float &fwidthZ,                 //
                              const float3 &normal, const float3 &normalPrev, const float &fwidthNormal //
)
{
    // check whether reprojected pixel is inside of the screen
    if (any(coord < make_int2(1, 1)) || any(coord > (imageDim - make_int2(1, 1))))
        return false;

    // check if deviation of depths is acceptable
    if (abs(Zprev - Z) / (fwidthZ + 1e-2f) > 10.f)
        return false;

    // check normals for compatibility
    if (distance(normal, normalPrev) / (fwidthNormal + 1e-2) > 4.0f)
        return false;

    return true;
}

__global__ void OutlierRemovalKernel(float *_outImg, const float *_rand, int height, int width)
{
    const int cx = IMAD(blockDim.x, blockIdx.x, threadIdx.x);
    const int cy = IMAD(blockDim.y, blockIdx.y, threadIdx.y);

    if (cx >= width || cy >= height)
        return;

    const int halfWinSize = 2;

    int sx = MAX(0, cx - halfWinSize);
    int ex = MIN(width - 1, cx + halfWinSize);
    int sy = MAX(0, cy - halfWinSize);
    int ey = MIN(height - 1, cy + halfWinSize);

    int numPixels = 0;
    float4 accCol2 = make_float4(0.f, 0.f, 0.f, 0.f);
    float4 accCol = make_float4(0.f, 0.f, 0.f, 0.f);
    for (int iy = sy; iy <= ey; ++iy)
    {
        for (int ix = sx; ix <= ex; ++ix)
        {
            int idx = iy * width + ix;
            const float4 &iCol = make_float4(_rand[idx * 3 + 0], _rand[idx * 3 + 1], _rand[idx * 3 + 2], 0.f);
            accCol += iCol;
            accCol2 += iCol * iCol;
            ++numPixels;
        }
    }
    float invNumPixels = 1.f / (float)numPixels;
    float4 meanCol = accCol * invNumPixels;
    float4 meanSqCol = accCol2 * invNumPixels;
    float stdDev = sqrtf(fmaxf(0.f, Avg(meanSqCol - meanCol * meanCol)));

    const int cIdx = cy * width + cx;
    const float4 &cCol = make_float4(_rand[cIdx * 3 + 0], _rand[cIdx * 3 + 1], _rand[cIdx * 3 + 2], 0.f);
    if (Avg(cCol) - Avg(meanCol) > 3.f * stdDev + 0.1f)
    {
        float4 robustMean = (accCol - cCol) / (float)(numPixels - 1);
        _outImg[cIdx * 3 + 0] = robustMean.x;
        _outImg[cIdx * 3 + 1] = robustMean.y;
        _outImg[cIdx * 3 + 2] = robustMean.z;
    }
    else
    {
        _outImg[cIdx * 3 + 0] = cCol.x;
        _outImg[cIdx * 3 + 1] = cCol.y;
        _outImg[cIdx * 3 + 2] = cCol.z;
    }
}

__global__ void WeightAvgKernel(
    const float* img,
    const float* wgt,
    float* out,
    const int width, const int height, const int winSize)
{
    const int cx = IMAD(blockDim.x, blockIdx.x, threadIdx.x);
    const int cy = IMAD(blockDim.y, blockIdx.y, threadIdx.y);

    if (cx >= width || cy >= height)
        return;

    const int halfWinSize = winSize / 2;
	const int winSizeSqr = winSize * winSize;
    const int cIdx = cy * width + cx;
    
    int sx = MAX(0, cx - halfWinSize);
    int ex = MIN(width - 1, cx + halfWinSize);
    int sy = MAX(0, cy - halfWinSize);
    int ey = MIN(height - 1, cy + halfWinSize);

    float4 accCol = make_float4(0.f, 0.f, 0.f, 0.f);
    int n = 0;
    float sumW = 0.f;

     for (int iy = sy; iy <= ey; ++iy)
    {
        for (int ix = sx; ix <= ex; ++ix)
        {
            int idx = iy * width + ix;
            const float4 &iCol = make_float4(img[idx * 3 + 0], img[idx * 3 + 1], img[idx * 3 + 2], 0.f);
            const float& weight = wgt[cIdx * winSizeSqr + n];
            ++n;

            accCol += weight * iCol;
            sumW += weight;	
        }
    }
    float invSumW = 1.f / fmaxf(sumW, JS_EP);
    float4 outCol = invSumW * accCol;

    out[cIdx * 3 + 0] = outCol.x;
    out[cIdx * 3 + 1] = outCol.y;
    out[cIdx * 3 + 2] = outCol.z;
}

__global__ void BoxFilterFeatureKernel(
    const float *img,
    const float *albedo,
    const float *normal,
    float *outImg,
    int height, 
    int width, 
    int winSize)
{
    const int cx = IMAD(blockDim.x, blockIdx.x, threadIdx.x);
    const int cy = IMAD(blockDim.y, blockIdx.y, threadIdx.y);

    if (cx >= width || cy >= height)
        return;

    int pIdx = cy * width + cx;
    outImg[pIdx * 3 + 0] = outImg[pIdx * 3 + 1] = outImg[pIdx * 3 + 2] = 0;

    const float4 cAlbedo = make_float4(albedo[pIdx * 3 + 0], albedo[pIdx * 3 + 1], albedo[pIdx * 3 + 2], 0.f);
    const float4 cNormal = make_float4(normal[pIdx * 3 + 0], normal[pIdx * 3 + 1], normal[pIdx * 3 + 2], 0.f);

    const int halfWinSize = winSize / 2;

    int sx = MAX(0, cx - halfWinSize);
    int ex = MIN(width - 1, cx + halfWinSize);
    int sy = MAX(0, cy - halfWinSize);
    int ey = MIN(height - 1, cy + halfWinSize);

    float sum[3] = {0.f};
    float num = 0.f;
    for (int iy = sy; iy <= ey; ++iy)
    {
        for (int ix = sx; ix <= ex; ++ix)
        {
            int idx = iy * width + ix;

            float4 iAlbedo = make_float4(albedo[idx * 3 + 0], albedo[idx * 3 + 1], albedo[idx * 3 + 2], 0.f);
            float4 iNormal = make_float4(normal[idx * 3 + 0], normal[idx * 3 + 1], normal[idx * 3 + 2], 0.f);

            float distSqAlbedo = norm2(iAlbedo - cAlbedo);
            float distSqNormal = norm2(iNormal - cNormal);

            if (distSqAlbedo > 0.1f || distSqNormal > 0.1f)
                continue;

            for (int c = 0; c < 3; ++c)
                sum[c] += img[idx * 3 + c];
            num += 1.f;
        }
    }

    outImg[pIdx * 3 + 0] = sum[0] / num;
    outImg[pIdx * 3 + 1] = sum[1] / num;
    outImg[pIdx * 3 + 2] = sum[2] / num;
}

__global__ void Reproject(
    bool *success,     // [1, H, W, 1]
    float **output,    // [1, H, W, ?]: data to be reprojected
    const float *mvec, //
    const int *dims,
    const float **input, //
    const float *linearZ,      //
    const float *prevLinearZ,  //
    const float *normal,       //
    const float *prevNormal,   //
    const float *pnFwidth,     //
    const int width, const int height, const int num_dims)
{
    const int cx = IMAD(blockDim.x, blockIdx.x, threadIdx.x);
    const int cy = IMAD(blockDim.y, blockIdx.y, threadIdx.y);

    if (cx >= width || cy >= height)
        return;

    const int cIdx = cy * width + cx;
    const int2 ipos = make_int2(cx, cy);
    const int2 imageDim = make_int2(width, height);

    for (int d = 0; d < num_dims; ++d)
    {
        int dim = dims[d];
        for (int c = 0; c < dim; ++c)
            output[d][cIdx * dim + c] = 0.f;
    }

    const float2 posH = make_float2(cx, cy);
    const float2 motion = make_float2(mvec[cIdx * 3 + 0], mvec[cIdx * 3 + 1]);
    const float normalFwidth = pnFwidth[cIdx * 3 + 1];

    // +0.5 to account for texel center offset
    const int2 iposPrev = float2int(posH + motion * imageDim + make_float2(0.5f, 0.5f));

    const float2 depth = make_float2(linearZ[cIdx * 3 + 0], linearZ[cIdx * 3 + 1]);
    const float3 norm = make_float3(normal[cIdx * 3 + 0], normal[cIdx * 3 + 1], normal[cIdx * 3 + 2]);

    bool v[4];
    const float2 posPrev = floor(posH) + motion * imageDim;
    const int2 offset[4] = {make_int2(0, 0), make_int2(1, 0), make_int2(0, 1), make_int2(1, 1)};

    // check for all 4 taps of the bilinear filter for validity
    bool valid = false;
    for (int sampleIdx = 0; sampleIdx < 4; sampleIdx++)
    {
        float2 depthPrev;
        float3 normalPrev;
        const int2 loc = float2int(posPrev) + offset[sampleIdx];
        const int locIdx = loc.y * imageDim.x + loc.x;
        if (loc.x >= imageDim.x || loc.x < 0 || loc.y >= imageDim.y || loc.y < 0)
        {
            v[sampleIdx] = false;
        }
        else
        {
            depthPrev = make_float2(prevLinearZ[locIdx * 3 + 0], prevLinearZ[locIdx * 3 + 1]);
            normalPrev = make_float3(prevNormal[locIdx * 3 + 0], prevNormal[locIdx * 3 + 1], prevNormal[locIdx * 3 + 2]);
            v[sampleIdx] = isReprojValid(imageDim, iposPrev, depth.x, depthPrev.x, depth.y, norm, normalPrev, normalFwidth);
            valid = valid || v[sampleIdx];
        }
    }

    if (valid)
    {
        float sumw = 0;
        float x = fracf(posPrev.x);
        float y = fracf(posPrev.y);

        // bilinear weights
        const float w[4] = {(1 - x) * (1 - y),
                            x * (1 - y),
                            (1 - x) * y,
                            x * y};

        // perform the actual bilinear interpolation
        for (int sampleIdx = 0; sampleIdx < 4; sampleIdx++)
        {
            const int2 loc = float2int(posPrev) + offset[sampleIdx];
            const int locIdx = loc.y * imageDim.x + loc.x;

            if (v[sampleIdx])
            {
                for (int d = 0; d < num_dims; ++d)
                {
                    int dim = dims[d];
                    for (int c = 0; c < dim; ++c)
                        output[d][cIdx * dim + c] += w[sampleIdx] * input[d][locIdx * dim + c];
                }
                sumw += w[sampleIdx];
            }
        }

        // redistribute weights in case not all taps were used
        valid = (sumw >= 0.01f);
        if (valid)
        {
            for (int d = 0; d < num_dims; ++d)
            {
                int dim = dims[d];
                for (size_t c = 0; c < dim; ++c)
                    output[d][cIdx * dim + c] /= sumw;
            }
        }
        else
        {
            for (int d = 0; d < num_dims; ++d)
            {
                int dim = dims[d];
                for (size_t c = 0; c < dim; ++c)
                    output[d][cIdx * dim + c] = 0.f;
            }
        }
    }

    if (!valid) // perform cross-bilateral filter in the hope to find some suitable samples somewhere
    {
        float nValid = 0.0;

        // this code performs a binary descision for each tap of the cross-bilateral filter
        const int radius = 1;
        for (int yy = -radius; yy <= radius; yy++)
        {
            for (int xx = -radius; xx <= radius; xx++)
            {
                const int2 p = iposPrev + make_int2(xx, yy);
                const int pIdx = p.y * imageDim.x + p.x;
                if (p.x >= imageDim.x || p.x < 0 || p.y >= imageDim.y || p.y < 0)
                {
                    // Outside window
                }
                else
                {
                    // Inside window
                    float2 depthPrev = make_float2(prevLinearZ[pIdx * 3 + 0], prevLinearZ[pIdx * 3 + 1]);
                    float3 normalPrev = make_float3(prevNormal[pIdx * 3 + 0], prevNormal[pIdx * 3 + 1], prevNormal[pIdx * 3 + 2]);

                    if (isReprojValid(imageDim, iposPrev, depth.x, depthPrev.x, depth.y, norm, normalPrev, normalFwidth))
                    {
                        for (int d = 0; d < num_dims; ++d)
                        {
                            int dim = dims[d];
                            for (size_t c = 0; c < dim; ++c)
                                output[d][cIdx * dim + c] += input[d][pIdx * dim + c];
                        }
                        nValid += 1.f;
                    }
                }
            }
        }

        if (nValid > 0)
        {
            valid = true;
            for (int d = 0; d < num_dims; ++d)
            {
                int dim = dims[d];
                for (size_t c = 0; c < dim; ++c)
                    output[d][cIdx * dim + c] /= nValid;
            }
        }
    }

    if (!valid)
    {
        for (int d = 0; d < num_dims; ++d)
        {
            int dim = dims[d];
            for (size_t c = 0; c < dim; ++c)
                output[d][cIdx * dim + c] = 0.f;
        }
    }

    success[cIdx] = valid;
}

__global__ void ReprojectVariance(
    bool *success,     // [1, H, W, 1]
    float **output,    // [1, H, W, ?]: data to be reprojected
    float *output_variance,
    const float *mvec, //
    const int *dims,
    const float **input, //
    const float* input_variance,
    const float *linearZ,      //
    const float *prevLinearZ,  //
    const float *normal,       //
    const float *prevNormal,   //
    const float *pnFwidth,     //
    const int width, const int height, const int num_dims)
{
    const int cx = IMAD(blockDim.x, blockIdx.x, threadIdx.x);
    const int cy = IMAD(blockDim.y, blockIdx.y, threadIdx.y);

    if (cx >= width || cy >= height)
        return;

    const int cIdx = cy * width + cx;
    const int2 ipos = make_int2(cx, cy);
    const int2 imageDim = make_int2(width, height);

    output_variance[cIdx * 3 + 0] = 0.f;
    output_variance[cIdx * 3 + 1] = 0.f;
    output_variance[cIdx * 3 + 2] = 0.f; 

    for (int d = 0; d < num_dims; ++d)
    {
        int dim = dims[d];
        for (int c = 0; c < dim; ++c)
            output[d][cIdx * dim + c] = 0.f;
    }

    const float2 posH = make_float2(cx, cy);
    const float2 motion = make_float2(mvec[cIdx * 3 + 0], mvec[cIdx * 3 + 1]);
    const float normalFwidth = pnFwidth[cIdx * 3 + 1];

    // +0.5 to account for texel center offset
    const int2 iposPrev = float2int(posH + motion * imageDim + make_float2(0.5f, 0.5f));

    const float2 depth = make_float2(linearZ[cIdx * 3 + 0], linearZ[cIdx * 3 + 1]);
    const float3 norm = make_float3(normal[cIdx * 3 + 0], normal[cIdx * 3 + 1], normal[cIdx * 3 + 2]);

    bool v[4];
    const float2 posPrev = floor(posH) + motion * imageDim;
    const int2 offset[4] = {make_int2(0, 0), make_int2(1, 0), make_int2(0, 1), make_int2(1, 1)};

    // check for all 4 taps of the bilinear filter for validity
    bool valid = false;
    for (int sampleIdx = 0; sampleIdx < 4; sampleIdx++)
    {
        float2 depthPrev;
        float3 normalPrev;
        const int2 loc = float2int(posPrev) + offset[sampleIdx];
        const int locIdx = loc.y * imageDim.x + loc.x;
        if (loc.x >= imageDim.x || loc.x < 0 || loc.y >= imageDim.y || loc.y < 0)
        {
            v[sampleIdx] = false;
        }
        else
        {
            depthPrev = make_float2(prevLinearZ[locIdx * 3 + 0], prevLinearZ[locIdx * 3 + 1]);
            normalPrev = make_float3(prevNormal[locIdx * 3 + 0], prevNormal[locIdx * 3 + 1], prevNormal[locIdx * 3 + 2]);
            v[sampleIdx] = isReprojValid(imageDim, iposPrev, depth.x, depthPrev.x, depth.y, norm, normalPrev, normalFwidth);
            valid = valid || v[sampleIdx];
        }
    }

    if (valid)
    {
        float sumw = 0;
        float x = fracf(posPrev.x);
        float y = fracf(posPrev.y);

        // bilinear weights
        const float w[4] = {(1 - x) * (1 - y),
                            x * (1 - y),
                            (1 - x) * y,
                            x * y};

        // perform the actual bilinear interpolation
        for (int sampleIdx = 0; sampleIdx < 4; sampleIdx++)
        {
            const int2 loc = float2int(posPrev) + offset[sampleIdx];
            const int locIdx = loc.y * imageDim.x + loc.x;

            if (v[sampleIdx])
            {
                for (int d = 0; d < num_dims; ++d)
                {
                    int dim = dims[d];
                    for (int c = 0; c < dim; ++c)
                        output[d][cIdx * dim + c] += w[sampleIdx] * input[d][locIdx * dim + c];
                }
                output_variance[cIdx * 3 + 0] += w[sampleIdx] * input_variance[locIdx * 3 + 0];
                output_variance[cIdx * 3 + 1] += w[sampleIdx] * input_variance[locIdx * 3 + 1];
                output_variance[cIdx * 3 + 2] += w[sampleIdx] * input_variance[locIdx * 3 + 2];
                sumw += w[sampleIdx];
            }
        }

        // redistribute weights in case not all taps were used
        valid = (sumw >= 0.01f);
        if (valid)
        {
            for (int d = 0; d < num_dims; ++d)
            {
                int dim = dims[d];
                for (size_t c = 0; c < dim; ++c)
                    output[d][cIdx * dim + c] /= sumw;
            }
            output_variance[cIdx * 3 + 0] /= sumw;
            output_variance[cIdx * 3 + 1] /= sumw;
            output_variance[cIdx * 3 + 2] /= sumw;
        }
        else
        {
            for (int d = 0; d < num_dims; ++d)
            {
                int dim = dims[d];
                for (size_t c = 0; c < dim; ++c)
                    output[d][cIdx * dim + c] = 0.f;
            }
            output_variance[cIdx * 3 + 0] = 0.f;
            output_variance[cIdx * 3 + 1] = 0.f;
            output_variance[cIdx * 3 + 2] = 0.f;
        }
    }

    if (!valid) // perform cross-bilateral filter in the hope to find some suitable samples somewhere
    {
        float nValid = 0.0;

        // this code performs a binary descision for each tap of the cross-bilateral filter
        const int radius = 1;
        for (int yy = -radius; yy <= radius; yy++)
        {
            for (int xx = -radius; xx <= radius; xx++)
            {
                const int2 p = iposPrev + make_int2(xx, yy);
                const int pIdx = p.y * imageDim.x + p.x;
                if (p.x >= imageDim.x || p.x < 0 || p.y >= imageDim.y || p.y < 0)
                {
                    // Outside window
                }
                else
                {
                    // Inside window
                    float2 depthPrev = make_float2(prevLinearZ[pIdx * 3 + 0], prevLinearZ[pIdx * 3 + 1]);
                    float3 normalPrev = make_float3(prevNormal[pIdx * 3 + 0], prevNormal[pIdx * 3 + 1], prevNormal[pIdx * 3 + 2]);

                    if (isReprojValid(imageDim, iposPrev, depth.x, depthPrev.x, depth.y, norm, normalPrev, normalFwidth))
                    {
                        for (int d = 0; d < num_dims; ++d)
                        {
                            int dim = dims[d];
                            for (size_t c = 0; c < dim; ++c)
                                output[d][cIdx * dim + c] += input[d][pIdx * dim + c];
                        }
                        output_variance[cIdx * 3 + 0] += input_variance[pIdx * 3 + 0];
                        output_variance[cIdx * 3 + 1] += input_variance[pIdx * 3 + 1];
                        output_variance[cIdx * 3 + 1] += input_variance[pIdx * 3 + 2];
                        nValid += 1.f;
                    }
                }
            }
        }

        if (nValid > 0)
        {
            valid = true;
            for (int d = 0; d < num_dims; ++d)
            {
                int dim = dims[d];
                for (size_t c = 0; c < dim; ++c)
                    output[d][cIdx * dim + c] /= nValid;
            }
            output_variance[cIdx * 3 + 0] /= nValid;
            output_variance[cIdx * 3 + 1] /= nValid;
            output_variance[cIdx * 3 + 2] /= nValid;
        }
    }

    if (!valid)
    {
        for (int d = 0; d < num_dims; ++d)
        {
            int dim = dims[d];
            for (size_t c = 0; c < dim; ++c)
                output[d][cIdx * dim + c] = 0.f;
        }
        output_variance[cIdx * 3 + 0] = 0.f;
        output_variance[cIdx * 3 + 1] = 0.f;
        output_variance[cIdx * 3 + 2] = 0.f;
    }

    success[cIdx] = valid;
}

__global__ void CalShrinkage(
    const float* current, 
    const float* history, 
    const float* var, 
    const float *albedo,
    const float *normal,
    float* shrinkage, 
    float* denominator,
    float* term,
    const int height, 
    const int width, 
    const int winSize)
{
    const int cx = IMAD(blockDim.x, blockIdx.x, threadIdx.x);
	const int cy = IMAD(blockDim.y, blockIdx.y, threadIdx.y);

	if (cx >= width || cy >= height)
		return;

	const int halfWinSize = winSize / 2;
	const int winSizeSqr = winSize * winSize;
    const int cIdx = cy * width + cx;

    int numPixels = 0;
    float4 accCol2 = make_float4(0.f, 0.f, 0.f, 0.f);
    float4 accCol = make_float4(0.f, 0.f, 0.f, 0.f);

    const float4 cAlbedo = make_float4(albedo[cIdx * 3 + 0], albedo[cIdx * 3 + 1], albedo[cIdx * 3 + 2], 0.f);
    const float4 cNormal = make_float4(normal[cIdx * 3 + 0], normal[cIdx * 3 + 1], normal[cIdx * 3 + 2], 0.f);

    int sx = MAX(0, cx - halfWinSize);
    int ex = MIN(width - 1, cx + halfWinSize);
    int sy = MAX(0, cy - halfWinSize);
    int ey = MIN(height - 1, cy + halfWinSize);

    //float4 sse = make_float4(0.f, 0.f, 0.f, 0.f);
    float sse = 0.f;
    int cnt = 0;
	
    for (int iy = sy; iy <= ey; ++iy)
    {
        for (int ix = sx; ix <= ex; ++ix)
        {
            int idx = iy * width + ix;
            float4 iCurrent = make_float4(current[idx * 3 + 0], current[idx * 3 + 1], current[idx * 3 + 2], 0.f);
            float4 iHistory = make_float4(history[idx * 3 + 0], history[idx * 3 + 1], history[idx * 3 + 2], 0.f);

            float4 iAlbedo = make_float4(albedo[idx * 3 + 0], albedo[idx * 3 + 1], albedo[idx * 3 + 2], 0.f);
            float4 iNormal = make_float4(normal[idx * 3 + 0], normal[idx * 3 + 1], normal[idx * 3 + 2], 0.f);
            float distSqAlbedo = norm2(iAlbedo - cAlbedo);
            float distSqNormal = norm2(iNormal - cNormal);

            //luminance 
            const float currLuminance = luminance(iCurrent);
            const float historyLuminance = luminance(iHistory);
            sse += (currLuminance - historyLuminance) * (currLuminance - historyLuminance);

            //if (distSqAlbedo > 0.1f || distSqNormal > 0.1f)
            //    continue;

            if(distSqNormal > 0.1f){
                cnt += 1;
            }
            
            accCol += iCurrent;
            accCol2 += (iCurrent * iCurrent);
            ++numPixels;         
        }
    }
    float df = (float)(winSizeSqr - 2);
    //float df = 1.f;
    float variance = var[cIdx * 3 + 0];
    float4 Current = make_float4(current[cIdx * 3 + 0], current[cIdx * 3 + 1], current[cIdx * 3 + 2], 0.f);
    float4 History = make_float4(history[cIdx * 3 + 0], history[cIdx * 3 + 1], history[cIdx * 3 + 2], 0.f);

    const float currLuminance = luminance(Current);
    const float historyLuminance = luminance(History);
    float diff = currLuminance - historyLuminance;

    float4 cCol = make_float4(current[cIdx * 3 + 0], current[cIdx * 3 + 1], current[cIdx * 3 + 2], 0.f);
    
    float4 alpha;
    alpha.x = fmaxf(0.f, 1.f-(df * variance) / fmaxf(sse, JS_EP));
    alpha.y = fmaxf(0.f, 1.f-(df * variance) / fmaxf(sse, JS_EP));
    alpha.z = fmaxf(0.f, 1.f-(df * variance) / fmaxf(sse, JS_EP));

    float invNumPixels = 1.f / (float)numPixels;
    float4 meanCol = accCol * invNumPixels;
    float4 meanSqCol = accCol2 * invNumPixels;
    float stdDev = sqrtf(fmaxf(0.f, Avg(meanSqCol - meanCol * meanCol)));

    //bool outlier = (Avg(cCol) - Avg(meanCol) > 2.5f * stdDev + 0.1f);
    //if(outlier != true  && variance < 1.0f && alpha.x > 0.7f){
    /*if((diff > 0.5f) || ((sse - diff > 1.0f) && cnt < 1)){
        alpha.x = 0.0f;
        alpha.y = 0.0f;
        alpha.z = 0.0f; 
        /*variance *= 1000.0f;
        alpha.x = fmaxf(0.f, 1.f-(df * variance) / fmaxf(sse, JS_EP));
        alpha.y = fmaxf(0.f, 1.f-(df * variance) / fmaxf(sse, JS_EP));
        alpha.z = fmaxf(0.f, 1.f-(df * variance) / fmaxf(sse, JS_EP));
    }*/
    
    /*if (sse > 10.f){
        alpha.x = 0.1f;
        alpha.y = 0.1f;
        alpha.z = 0.1f; 
    }*/
    /*if(alpha.x == 0.f){
        alpha.x = 0.1f;
        alpha.y = 0.1f;
        alpha.z = 0.1f;
    }*/

    shrinkage[cIdx * 3 + 0] = alpha.x;
    shrinkage[cIdx * 3 + 1] = alpha.y;
    shrinkage[cIdx * 3 + 2] = alpha.z;

    denominator[cIdx * 3 + 0] = sse;
    denominator[cIdx * 3 + 1] = cnt;
    denominator[cIdx * 3 + 2] = diff;
}

__global__ void AvgShrinkage(
    const float* albedo, 
    const float* normal, 
    const float *shrinkage, 
    //float* output,
    float* outAlpha,
    float* output_pixel,
    const int height, 
    const int width, 
    const int winSize)
{
    const int cx = IMAD(blockDim.x, blockIdx.x, threadIdx.x);
	const int cy = IMAD(blockDim.y, blockIdx.y, threadIdx.y);

	if (cx >= width || cy >= height)
		return;

	const int halfWinSize = winSize / 2;
	const int winSizeSqr = winSize * winSize;
    const int cIdx = cy * width + cx;

    int sx = MAX(0, cx - halfWinSize);
    int ex = MIN(width - 1, cx + halfWinSize);
    int sy = MAX(0, cy - halfWinSize);
    int ey = MIN(height - 1, cy + halfWinSize);

    float4 avgShrinkage = make_float4(0.f, 0.f, 0.f, 0.f);
    const float4 cAlbedo = make_float4(albedo[cIdx * 3 + 0], albedo[cIdx * 3 + 1], albedo[cIdx * 3 + 2], 0.f);
    const float4 cNormal = make_float4(normal[cIdx * 3 + 0], normal[cIdx * 3 + 1], normal[cIdx * 3 + 2], 0.f);
    float num = 0.f;

    for (int iy = sy; iy <= ey; ++iy)
    {
        for (int ix = sx; ix <= ex; ++ix)
        {
            int idx = iy * width + ix;

            const float4& _shrinkage = make_float4(shrinkage[idx * 3 + 0], shrinkage[idx * 3 + 1], shrinkage[idx * 3 + 2], 0.f);
            float4 iAlbedo = make_float4(albedo[idx * 3 + 0], albedo[idx * 3 + 1], albedo[idx * 3 + 2], 0.f);
            float4 iNormal = make_float4(normal[idx * 3 + 0], normal[idx * 3 + 1], normal[idx * 3 + 2], 0.f);

            float distSqAlbedo = norm2(iAlbedo - cAlbedo);
            float distSqNormal = norm2(iNormal - cNormal);

            if (distSqAlbedo > 0.1f || distSqNormal > 0.1f)
                continue;

            num += 1.f;
            avgShrinkage += _shrinkage;
        }
    }
    
    avgShrinkage = avgShrinkage / num;

    outAlpha[cIdx * 3 + 0] = avgShrinkage.x;
    outAlpha[cIdx * 3 + 1] = avgShrinkage.y;
    outAlpha[cIdx * 3 + 2] = avgShrinkage.z;

    //for debugging
    output_pixel[cIdx * 3 + 0] = shrinkage[cIdx * 3 + 0];
    output_pixel[cIdx * 3 + 1] = shrinkage[cIdx * 3 + 1];
    output_pixel[cIdx * 3 + 2] = shrinkage[cIdx * 3 + 2];
}

void CUDASynchronizeFunc()
{
    cudaDeviceSynchronize();
}

void OutlierRemovalFunc(const GPUDevice &_dev, const float *_rand, float *_out, int nBatch, int height, int width)
{
    const int blockDim = 8;
    dim3 threads(blockDim, blockDim);
    dim3 grid(iDivUp(width, blockDim), iDivUp(height, blockDim));

    OutlierRemovalKernel<<<grid, threads, 0, _dev.stream()>>>(_out, _rand, height, width);
}

void ReprojectVarianceFunc(
    const GPUDevice &_dev,
    const float *mvec,
    const int *dims,
    const float **input_list,
    const float *input_variance,
    const float *linearZ,
    const float *prevLinearZ,
    const float *normal,
    const float *prevNormal,
    const float *pnFwidth,
    float **output,
    bool *success,
    float *output_variance,
    const int num_dims, const int height, const int width)
{
    const int blockDim = 8;
    dim3 threads(blockDim, blockDim);
    dim3 grid(iDivUp(width, blockDim), iDivUp(height, blockDim));
    
    ReprojectVariance<<<grid, threads, 0, _dev.stream()>>>(
        // output
        success, output, output_variance,
        // input
        mvec, dims, input_list, input_variance, linearZ, prevLinearZ, normal, prevNormal, pnFwidth,
        // misc.
        width, height, num_dims
        //
    );
}

void ReprojectFunc(
    const GPUDevice &_dev,
    const float *mvec,
    const int *dims,
    const float **input_list,
    const float *linearZ,
    const float *prevLinearZ,
    const float *normal,
    const float *prevNormal,
    const float *pnFwidth,
    float **output,
    bool *success,
    const int num_dims, const int height, const int width)
{
    const int blockDim = 8;
    dim3 threads(blockDim, blockDim);
    dim3 grid(iDivUp(width, blockDim), iDivUp(height, blockDim));

    Reproject<<<grid, threads, 0, _dev.stream()>>>(
        // output
        success, output,
        // input
        mvec, dims, input_list, linearZ, prevLinearZ, normal, prevNormal, pnFwidth,
        // misc.
        width, height, num_dims
        //
    );
}

void CalShrinkageFunc(
    const GPUDevice &_dev,
    const float* current, 
    const float* history, 
    const float *albedo,
    const float *normal,
    const float* var, 
    float* shrinkage, 
    float* denominator,
    float* term,
    const int height, 
    const int width, 
    const int winSize
)
{
    const int blockDim = 8;
    dim3 threads(blockDim, blockDim);
    dim3 grid(iDivUp(width, blockDim), iDivUp(height, blockDim));

    CalShrinkage<<<grid, threads, 0, _dev.stream()>>>(
        current, history, albedo, normal, var,
        shrinkage, denominator, term,
        height, width, winSize
    );
}

void AvgShrinkageFunc(
    const GPUDevice &_dev,
    const float* albedo, 
    const float* normal, 
    const float *shrinkage, 
    //float* output,
    float* outAlpha,
    float* output_pixel,
    const int height, 
    const int width, 
    const int winSize
)
{
    const int blockDim = 8;
    dim3 threads(blockDim, blockDim);
    dim3 grid(iDivUp(width, blockDim), iDivUp(height, blockDim));

    AvgShrinkage<<<grid, threads, 0, _dev.stream()>>>(
        albedo, normal, shrinkage,
        outAlpha, output_pixel,
        height, width, winSize
    );

}

void BoxFilterFunc(
    const GPUDevice &_dev,
    const float *img,
    const float *albedo,
    const float *normal,
    float *outImg,
    int height, 
    int width, 
    int winSize
)
{
    const int blockDim = 8;
    dim3 threads(blockDim, blockDim);
    dim3 grid(iDivUp(width, blockDim), iDivUp(height, blockDim));

    BoxFilterFeatureKernel<<<grid, threads, 0, _dev.stream()>>>(
        img, albedo, normal,
        outImg,
        height, width, winSize
    );

}

void WeightAvgFunc(
    const GPUDevice &_dev,
    const float *img,
    const float* wgt,
    float* out,
    const int width, 
    const int height, 
    const int winSize
)
{
    const int blockDim = 8;
    dim3 threads(blockDim, blockDim);
    dim3 grid(iDivUp(width, blockDim), iDivUp(height, blockDim));
    WeightAvgKernel<<<grid, threads, 0, _dev.stream()>>>(
        img, wgt,
        out,
        width, height, winSize
    );
}
#endif // GOOGLE_CUDA