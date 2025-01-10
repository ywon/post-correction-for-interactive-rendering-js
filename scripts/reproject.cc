
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "cuda_runtime.h"

using namespace tensorflow;

// Some debug macros
#define CUDA_CHECK(val)                                                                                       \
    {                                                                                                         \
        if (val != cudaSuccess)                                                                               \
        {                                                                                                     \
            fprintf(stderr, "Error %s at line %d in file %s\n", cudaGetErrorString(val), __LINE__, __FILE__); \
            exit(1);                                                                                          \
        }                                                                                                     \
    }

using GPUDevice = Eigen::GpuDevice;

REGISTER_OP("CUDASynchronize")
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext *c)
                { return Status(); }
    );

REGISTER_OP("WeightAvg")
    .Input("img: float")
    .Input("wgt: float")
    .Output("out: float")
    .Attr("img_height: int >= 1")
    .Attr("img_width: int >= 1")
    .Attr("winSize: int >= 1")
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext *c)
                {
                    c->set_output(0, c->input(0));
                    return Status();
                }
    );

REGISTER_OP("OutlierRemoval")
    .Input("rand: float")
    .Output("out: float")
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext *c)
                {
    c->set_output(0, c->input(0));
    return Status(); }
    );

REGISTER_OP("BoxFilter")
    .Input("img: float")
    .Input("albedo: float")
    .Input("normal: float")
    .Output("output: float")
    .Attr("img_height: int >= 1")
    .Attr("img_width: int >= 1")
    .Attr("winSize: int >= 1")
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext *c)
                {
                    c->set_output(0, c->input(0));
                    c->set_output(1, c->input(0));
                    c->set_output(2, c->input(0));
                    return Status();
                }
    );

REGISTER_OP("ReprojectVariance")
    .Attr("T: list(type)")
    .Input("mvec: float")
    .Input("input_list: T")
    .Input("input_variance: float")
    .Input("linear_z: float")
    .Input("prev_linear_z: float")
    .Input("normal: float")
    .Input("prev_normal: float")
    .Input("pn_fwidth: float")
    .Output("success: bool")
    .Output("output: T")
    .Output("output_variance: float")
    .Attr("img_height: int >= 1")
    .Attr("img_width: int >= 1")
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext *c)
                {
                    // Set output
                    c->set_output(1, c->input(1));
                    c->set_output(2, c->input(2));
                    // Set success
                    auto B = c->Dim(c->input(0), 0);
                    auto H = c->Dim(c->input(0), 1);
                    auto W = c->Dim(c->input(0), 2);
                    c->set_output(0, c->MakeShape({B, H, W, 1}));
                    return Status(); //
                }                    //
    );

REGISTER_OP("Reproject")
    .Attr("T: list(type)")
    .Input("mvec: float")
    .Input("input_list: T")
    .Input("linear_z: float")
    .Input("prev_linear_z: float")
    .Input("normal: float")
    .Input("prev_normal: float")
    .Input("pn_fwidth: float")
    .Output("success: bool")
    .Output("output: T")
    .Attr("img_height: int >= 1")
    .Attr("img_width: int >= 1")
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext *c)
                {
                    // Set output
                    c->set_output(1, c->input(1));
                    // Set success
                    auto B = c->Dim(c->input(0), 0);
                    auto H = c->Dim(c->input(0), 1);
                    auto W = c->Dim(c->input(0), 2);
                    c->set_output(0, c->MakeShape({B, H, W, 1}));
                    return Status(); //
                }                    //
    );

REGISTER_OP("CalShrinkage")
    .Input("current: float")
    .Input("history: float")
    .Input("albedo: float")
    .Input("normal: float")
    .Input("var: float")
    .Output("shrinkage: float")
    .Output("denominator: float")
    .Output("term: float")
    .Attr("img_height: int >= 1")
    .Attr("img_width: int >= 1")
    .Attr("winSize: int >= 1")
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext *c)
                {
                    c->set_output(0, c->input(0));
                    c->set_output(1, c->input(0));
                    c->set_output(2, c->input(0));
                    return Status();
                }
    );


REGISTER_OP("AvgShrinkage")
    .Input("albedo: float")
    .Input("normal: float")
    .Input("shrinkage: float")
    //.Output("output: float")
    .Output("out_alpha: float")
    .Output("output_pixel: float")
    .Attr("img_height: int >= 1")
    .Attr("img_width: int >= 1")
    .Attr("winSize: int >= 1")
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext *c)
                {
                    c->set_output(1, c->input(0));
                    c->set_output(0, c->input(0)); 
                    //c->set_output(2, c->input(0));
                    
                    return Status();
                }
    );

void CUDASynchronizeFunc();

void WeightAvgFunc(
    const GPUDevice &_dev,
    const float* img,
    const float* wgt,
    float* out,
    const int width, 
    const int height, 
    const int winSize);

void OutlierRemovalFunc(
    const GPUDevice &_dev, 
    const float *_rand, 
    float *_out, 
    int nBatch, 
    int height, 
    int width);

void BoxFilterFunc(
    const GPUDevice &_dev,
    const float *img,
    const float *albedo,
    const float *normal,
    float *outImg,
    int height, int width, int winSize);

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
    const int num_dims, const int height, const int width);

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
    const int num_dims, const int height, const int width);

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
    const int winSize);

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
    const int winSize);

class CUDASynchronizeOp : public OpKernel
{
public:
    explicit CUDASynchronizeOp(OpKernelConstruction *context) : OpKernel(context)
    {
    }
    void Compute(OpKernelContext *context) override
    {
        CUDASynchronizeFunc();
    }

private:
};

class WeightAvgOp : public OpKernel
{
public:
    explicit WeightAvgOp(OpKernelConstruction *context) : OpKernel(context)
    {
        context->GetAttr("img_height", &imgHeight);
        context->GetAttr("img_width", &imgWidth);
        context->GetAttr("winSize", &winSize);
    }
    void Compute(OpKernelContext *context) override
    {
        const Tensor& img = context->input(0);
	    const Tensor& wgt = context->input(1);

        const TensorShape& img_shape = img.shape();    
    
        TensorShape output_shape;
        output_shape.AddDim(img_shape.dim_size(0));
        output_shape.AddDim(img_shape.dim_size(1));
        output_shape.AddDim(img_shape.dim_size(2));
        output_shape.AddDim(img_shape.dim_size(3));

        Tensor* out_tensor = NULL;
        OP_REQUIRES_OK(context, context->allocate_output(0, output_shape, &out_tensor));
        auto out_mat = out_tensor->tensor<float, 4>();

        WeightAvgFunc(context->eigen_device<GPUDevice>(),
	                      img.flat<float>().data(), 
                          wgt.flat<float>().data(), 
                          out_mat.data(),
                          imgHeight, imgWidth, winSize);
    }   

private:
    int imgHeight, imgWidth, winSize;
};

class OutlierRemovalOp : public OpKernel
{
public:
    explicit OutlierRemovalOp(OpKernelConstruction *context) : OpKernel(context)
    {
    }
    void Compute(OpKernelContext *context) override
    {
        const Tensor &rand = context->input(0);

        const TensorShape &input_shape = rand.shape();

        Tensor *out_img = NULL;
        OP_REQUIRES_OK(context, context->allocate_output(0, input_shape, &out_img));
        auto out_mat = out_img->tensor<float, 4>();

        OutlierRemovalFunc(context->eigen_device<GPUDevice>(),
                           rand.flat<float>().data(), 
                           out_mat.data(), 
                           input_shape.dim_size(0), 
                           input_shape.dim_size(1), 
                           input_shape.dim_size(2));
    }

private:
};

class BoxFilterOp : public OpKernel
{
public:
    explicit BoxFilterOp(OpKernelConstruction *context) : OpKernel(context)
    {
        context->GetAttr("img_height", &imgHeight);
        context->GetAttr("img_width", &imgWidth);
        context->GetAttr("winSize", &winSize);
    }
    void Compute(OpKernelContext *context) override
    {
        const Tensor& img = context->input(0);
        const Tensor& albedo = context->input(1);
        const Tensor& normal = context->input(2);

        const TensorShape& input_shape = img.shape();
        TensorShape output_shape;
        output_shape.AddDim(input_shape.dim_size(0));
        output_shape.AddDim(input_shape.dim_size(1));
        output_shape.AddDim(input_shape.dim_size(2));
        output_shape.AddDim(input_shape.dim_size(3));

        Tensor *out_tensor = NULL;
        OP_REQUIRES_OK(context, context->allocate_output(0, output_shape, &out_tensor));
        auto out_mat = out_tensor->tensor<float, 4>();

        BoxFilterFunc(context->eigen_device<GPUDevice>(),
                      img.flat<float>().data(),
                      albedo.flat<float>().data(),
                      normal.flat<float>().data(),
                      out_mat.data(),
                      imgHeight, imgWidth, winSize);
    }
    private:
        int imgHeight, imgWidth, winSize;
};

class ReprojectVarianceOp : public OpKernel
{
public:
    explicit ReprojectVarianceOp(OpKernelConstruction *context) : OpKernel(context)
    {
        context->GetAttr("img_height", &imgHeight);
        context->GetAttr("img_width", &imgWidth);
    }
    void Compute(OpKernelContext *context) override
    {
        // Get input tensors (from REGISTER_OP)
        const Tensor *mvec, *linearZ, *prevLinearZ, *normal, *prevNormal, *pnFwidth, *input_variance;
        OP_REQUIRES_OK(context, context->input("mvec", &mvec));
        OP_REQUIRES_OK(context, context->input("input_variance", &input_variance));
        OP_REQUIRES_OK(context, context->input("linear_z", &linearZ));
        OP_REQUIRES_OK(context, context->input("prev_linear_z", &prevLinearZ));
        OP_REQUIRES_OK(context, context->input("normal", &normal));
        OP_REQUIRES_OK(context, context->input("prev_normal", &prevNormal));
        OP_REQUIRES_OK(context, context->input("pn_fwidth", &pnFwidth));

        // Get list of tensors
        OpInputList input_list;
        OP_REQUIRES_OK(context, context->input_list("input_list", &input_list));
        int num_dims = input_list.size();
        std::vector<const float *> input_list_ptr(num_dims);
        std::vector<int> dims(num_dims);
        for (int i = 0; i < num_dims; ++i)
        {
            input_list_ptr[i] = input_list[i].flat<float>().data();
            dims[i] = input_list[i].shape().dim_size(3);
        }

        // Shape of input
        TensorShape input_shape = mvec->shape();
        TensorShape input_variance_shape = input_variance->shape();
        int B = input_shape.dim_size(0);
        int H = input_shape.dim_size(1);
        int W = input_shape.dim_size(2);

        // Output success
        Tensor *success_img = NULL;
        OP_REQUIRES_OK(context, context->allocate_output(0, TensorShape{B, H, W, 1}, &success_img));
        auto success_mat = success_img->tensor<bool, 4>();

        Tensor *output_variance = NULL;
        OP_REQUIRES_OK(context, context->allocate_output(3, input_variance_shape ,&output_variance));
        auto out_variance = output_variance->tensor<float, 4>();

        // Output list
        OpOutputList output_list;
        OP_REQUIRES_OK(context, context->output_list("output", &output_list));
        std::vector<Tensor *> out_imgs(num_dims, NULL);
        //context->output_list("output", &output_list);
        std::vector<float *> out_mats(num_dims);
        for (int i = 0; i < num_dims; ++i)
        {
            OP_REQUIRES_OK(context, output_list.allocate(i, TensorShape{B, H, W, dims[i]}, &out_imgs[i]));
            out_mats[i] = out_imgs[i]->tensor<float, 4>().data();
        }

        if (!initialized)
        {
            // Allocate memory for dims in GPU
            CUDA_CHECK(cudaMalloc((void **)&dims_gpu_ptr, sizeof(int) * num_dims));
            // Allocate memory for pointer of input/output list in GPU
            CUDA_CHECK(cudaMalloc((void **)&input_gpu_ptr, sizeof(float *) * num_dims));
            CUDA_CHECK(cudaMalloc((void **)&out_gpu_ptr, sizeof(float *) * num_dims));
            initialized = true;
        }

        CUDA_CHECK(cudaMemcpy(dims_gpu_ptr, dims.data(), sizeof(int) * num_dims, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(input_gpu_ptr, input_list_ptr.data(), sizeof(float *) * num_dims, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(out_gpu_ptr, out_mats.data(), sizeof(float *) * num_dims, cudaMemcpyHostToDevice));

        ReprojectVarianceFunc(context->eigen_device<GPUDevice>(),
                      mvec->flat<float>().data(),
                      dims_gpu_ptr,
                      input_gpu_ptr,
                      input_variance->flat<float>().data(),
                      linearZ->flat<float>().data(),
                      prevLinearZ->flat<float>().data(),
                      normal->flat<float>().data(),
                      prevNormal->flat<float>().data(),
                      pnFwidth->flat<float>().data(),
                      out_gpu_ptr,
                      success_mat.data(),
                      out_variance.data(),
                      num_dims, input_shape.dim_size(1), input_shape.dim_size(2));

        // CUDA_CHECK(cudaFree(out_gpu_ptr));
    }

private:
    int imgHeight, imgWidth;
    static bool initialized;
    static int *dims_gpu_ptr;
    static const float **input_gpu_ptr;
    static float **out_gpu_ptr;
};
bool ReprojectVarianceOp::initialized = false;
int *ReprojectVarianceOp::dims_gpu_ptr = nullptr;
const float **ReprojectVarianceOp::input_gpu_ptr = nullptr;
float **ReprojectVarianceOp::out_gpu_ptr = nullptr;

class ReprojectOp : public OpKernel
{
public:
    explicit ReprojectOp(OpKernelConstruction *context) : OpKernel(context)
    {
        context->GetAttr("img_height", &imgHeight);
        context->GetAttr("img_width", &imgWidth);
    }
    void Compute(OpKernelContext *context) override
    {
        // Get input tensors (from REGISTER_OP)
        const Tensor *mvec, *linearZ, *prevLinearZ, *normal, *prevNormal, *pnFwidth;
        OP_REQUIRES_OK(context, context->input("mvec", &mvec));
        OP_REQUIRES_OK(context, context->input("linear_z", &linearZ));
        OP_REQUIRES_OK(context, context->input("prev_linear_z", &prevLinearZ));
        OP_REQUIRES_OK(context, context->input("normal", &normal));
        OP_REQUIRES_OK(context, context->input("prev_normal", &prevNormal));
        OP_REQUIRES_OK(context, context->input("pn_fwidth", &pnFwidth));

        // Get list of tensors
        OpInputList input_list;
        OP_REQUIRES_OK(context, context->input_list("input_list", &input_list));
        int num_dims = input_list.size();
        std::vector<const float *> input_list_ptr(num_dims);
        std::vector<int> dims(num_dims);
        for (int i = 0; i < num_dims; ++i)
        {
            input_list_ptr[i] = input_list[i].flat<float>().data();
            dims[i] = input_list[i].shape().dim_size(3);
        }

        // Shape of input
        TensorShape input_shape = mvec->shape();
        int B = input_shape.dim_size(0);
        int H = input_shape.dim_size(1);
        int W = input_shape.dim_size(2);

        // Output success
        Tensor *success_img = NULL;
        OP_REQUIRES_OK(context, context->allocate_output(0, TensorShape{B, H, W, 1}, &success_img));
        auto success_mat = success_img->tensor<bool, 4>();

        // Output list
        OpOutputList output_list(context, 1, num_dims + 1);
        std::vector<Tensor *> out_imgs(num_dims, NULL);
        context->output_list("output", &output_list);
        std::vector<float *> out_mats(num_dims);
        for (int i = 0; i < num_dims; ++i)
        {
            OP_REQUIRES_OK(context, output_list.allocate(i, TensorShape{B, H, W, dims[i]}, &out_imgs[i]));
            out_mats[i] = out_imgs[i]->tensor<float, 4>().data();
        }

        if (!initialized)
        {
            // Allocate memory for dims in GPU
            CUDA_CHECK(cudaMalloc((void **)&dims_gpu_ptr, sizeof(int) * num_dims));
            // Allocate memory for pointer of input/output list in GPU
            CUDA_CHECK(cudaMalloc((void **)&input_gpu_ptr, sizeof(float *) * num_dims));
            CUDA_CHECK(cudaMalloc((void **)&out_gpu_ptr, sizeof(float *) * num_dims));
            initialized = true;
        }

        CUDA_CHECK(cudaMemcpy(dims_gpu_ptr, dims.data(), sizeof(int) * num_dims, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(input_gpu_ptr, input_list_ptr.data(), sizeof(float *) * num_dims, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(out_gpu_ptr, out_mats.data(), sizeof(float *) * num_dims, cudaMemcpyHostToDevice));

        ReprojectFunc(context->eigen_device<GPUDevice>(),
                      mvec->flat<float>().data(),
                      dims_gpu_ptr,
                      input_gpu_ptr,
                      linearZ->flat<float>().data(),
                      prevLinearZ->flat<float>().data(),
                      normal->flat<float>().data(),
                      prevNormal->flat<float>().data(),
                      pnFwidth->flat<float>().data(),
                      out_gpu_ptr,
                      success_mat.data(),
                      num_dims, input_shape.dim_size(1), input_shape.dim_size(2));

        // CUDA_CHECK(cudaFree(out_gpu_ptr));
    }
private:
    int imgHeight, imgWidth;
    static bool initialized;
    static int *dims_gpu_ptr;
    static const float **input_gpu_ptr;
    static float **out_gpu_ptr;
};
bool ReprojectOp::initialized = false;
int *ReprojectOp::dims_gpu_ptr = nullptr;
const float **ReprojectOp::input_gpu_ptr = nullptr;
float **ReprojectOp::out_gpu_ptr = nullptr;

class CalShrinkageOp : public OpKernel
{
public:
    explicit CalShrinkageOp(OpKernelConstruction *context) : OpKernel(context)
    {
        context->GetAttr("img_height", &imgHeight);
        context->GetAttr("img_width", &imgWidth);
        context->GetAttr("winSize", &winSize);
    }
    void Compute(OpKernelContext *context) override
    {
        const Tensor& current = context->input(0);
        const Tensor& history = context->input(1);
        const Tensor& albedo = context->input(2);
        const Tensor& normal = context->input(3);
        const Tensor& var = context->input(4);

        const TensorShape& input_shape = current.shape();

        TensorShape output_shape;
        output_shape.AddDim(input_shape.dim_size(0));
        output_shape.AddDim(input_shape.dim_size(1));
        output_shape.AddDim(input_shape.dim_size(2));
        output_shape.AddDim(input_shape.dim_size(3));

        Tensor *out_tensor = NULL;
        OP_REQUIRES_OK(context, context->allocate_output(0, output_shape, &out_tensor));
        auto out_mat = out_tensor->tensor<float, 4>();

        Tensor *out_denominator = NULL;
        OP_REQUIRES_OK(context, context->allocate_output(1, input_shape, &out_denominator));
        auto out_denominator_mat = out_denominator->tensor<float, 4>();

        Tensor *out_term = NULL;
        OP_REQUIRES_OK(context, context->allocate_output(2, input_shape, &out_term));
        auto out_term_mat = out_term->tensor<float, 4>();

        CalShrinkageFunc(context->eigen_device<GPUDevice>(),
                         current.flat<float>().data(), 
                         history.flat<float>().data(), 
                         albedo.flat<float>().data(),
                         normal.flat<float>().data(),
                         var.flat<float>().data(),
                         out_mat.data(), 
                         out_denominator_mat.data(),
                         out_term_mat.data(),
                         imgHeight, imgWidth, winSize); 
    }

private:
    int imgHeight, imgWidth, winSize;
};


class AvgShrinkageOp : public OpKernel
{
public:
    explicit AvgShrinkageOp(OpKernelConstruction *context) : OpKernel(context)
    {
        context->GetAttr("img_height", &imgHeight);
        context->GetAttr("img_width", &imgWidth);
        context->GetAttr("winSize", &winSize);
    }
    void Compute(OpKernelContext *context) override
    {
        const Tensor& albedo = context->input(0);
        const Tensor& normal = context->input(1);
        const Tensor& shrinkage = context->input(2);

        const TensorShape& input_shape = albedo.shape();
        const TensorShape& alpha_shape = normal.shape();

        TensorShape output_shape, output_alpha_shape;
        output_shape.AddDim(input_shape.dim_size(0));
        output_shape.AddDim(input_shape.dim_size(1));
        output_shape.AddDim(input_shape.dim_size(2));
        output_shape.AddDim(input_shape.dim_size(3));
        output_alpha_shape.AddDim(alpha_shape.dim_size(0));
        output_alpha_shape.AddDim(alpha_shape.dim_size(1));
        output_alpha_shape.AddDim(alpha_shape.dim_size(2));
        output_alpha_shape.AddDim(alpha_shape.dim_size(3));

        //Tensor *out_tensor = NULL;
        Tensor *out_alpha_tensor = NULL;

        //OP_REQUIRES_OK(context, context->allocate_output(0, output_shape, &out_tensor));
        OP_REQUIRES_OK(context, context->allocate_output(0, output_alpha_shape, &out_alpha_tensor));

        //auto out_mat = out_tensor->tensor<float, 4>();
        auto out_alpha_mat = out_alpha_tensor->tensor<float, 4>();

        Tensor *out_pixel = NULL;
        OP_REQUIRES_OK(context, context->allocate_output(1, input_shape, &out_pixel));
        auto out_pixel_mat = out_pixel->tensor<float, 4>();

        AvgShrinkageFunc(context->eigen_device<GPUDevice>(),
                         albedo.flat<float>().data(), 
                         normal.flat<float>().data(), 
                         shrinkage.flat<float>().data(),
                         //out_mat.data(), 
                         out_alpha_mat.data(), 
                         out_pixel_mat.data(),
                         imgHeight, imgWidth, winSize); 
    }

private:
    int imgHeight, imgWidth, winSize;
};
REGISTER_KERNEL_BUILDER(Name("CUDASynchronize").Device(DEVICE_GPU), CUDASynchronizeOp);
REGISTER_KERNEL_BUILDER(Name("OutlierRemoval").Device(DEVICE_GPU), OutlierRemovalOp);
REGISTER_KERNEL_BUILDER(Name("BoxFilter").Device(DEVICE_GPU), BoxFilterOp);
REGISTER_KERNEL_BUILDER(Name("Reproject").Device(DEVICE_GPU), ReprojectOp);
REGISTER_KERNEL_BUILDER(Name("ReprojectVariance").Device(DEVICE_GPU), ReprojectVarianceOp);
REGISTER_KERNEL_BUILDER(Name("CalShrinkage").Device(DEVICE_GPU), CalShrinkageOp);
REGISTER_KERNEL_BUILDER(Name("AvgShrinkage").Device(DEVICE_GPU), AvgShrinkageOp);
REGISTER_KERNEL_BUILDER(Name("WeightAvg").Device(DEVICE_GPU), WeightAvgOp)
