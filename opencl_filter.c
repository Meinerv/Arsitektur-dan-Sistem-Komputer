#ifdef __APPLE__
  #include <OpenCL/opencl.h>
#else
  #include <CL/cl.h>
#endif

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

/* ── Image struct ── */
typedef struct {
    int   width, height, maxval;
    unsigned char *data;
} Image;

/* ─────────────────────────────────────────────
   PGM I/O
   ───────────────────────────────────────────── */
Image *read_pgm(const char *filename)
{
    FILE *fp = fopen(filename, "rb");
    if (!fp) { fprintf(stderr, "Cannot open %s\n", filename); return NULL; }

    Image *img = (Image *)malloc(sizeof(Image));
    char magic[3];
    fscanf(fp, "%2s", magic);
    if (strcmp(magic, "P5") != 0) {
        fprintf(stderr, "Only binary PGM (P5) supported.\n");
        free(img); fclose(fp); return NULL;
    }

    int c = fgetc(fp);
    while (c == '\n' || c == ' ') c = fgetc(fp);
    while (c == '#') { while (fgetc(fp) != '\n'); c = fgetc(fp); }
    ungetc(c, fp);

    fscanf(fp, "%d %d %d", &img->width, &img->height, &img->maxval);
    fgetc(fp);

    size_t sz = (size_t)img->width * img->height;
    img->data = (unsigned char *)malloc(sz);
    if (fread(img->data, 1, sz, fp) != sz) {
        fprintf(stderr, "Error reading pixel data.\n");
        free(img->data); free(img); fclose(fp); return NULL;
    }
    fclose(fp);
    return img;
}

int write_pgm(const char *filename, const Image *img)
{
    FILE *fp = fopen(filename, "wb");
    if (!fp) { fprintf(stderr, "Cannot write %s\n", filename); return -1; }
    fprintf(fp, "P5\n%d %d\n%d\n", img->width, img->height, img->maxval);
    fwrite(img->data, 1, (size_t)img->width * img->height, fp);
    fclose(fp);
    return 0;
}

void free_image(Image *img)
{
    if (img) { free(img->data); free(img); }
}


//   Salt-and-Pepper Noise
void add_salt_pepper_noise(Image *img, double noise_ratio)
{
    srand((unsigned)time(NULL));
    size_t total = (size_t)img->width * img->height;
    size_t n     = (size_t)(total * noise_ratio);

    for (size_t i = 0; i < n / 2; i++)
        img->data[(size_t)rand() % total] = 255;
    for (size_t i = 0; i < n / 2; i++)
        img->data[(size_t)rand() % total] = 0;

    printf("[Noise] Salt-and-pepper noise added (ratio=%.2f%%)\n",
           noise_ratio * 100.0);
}

/* ─────────────────────────────────────────────
   PSNR metric
   ───────────────────────────────────────────── */
double compute_psnr(const Image *orig, const Image *filtered)
{
    size_t total = (size_t)orig->width * orig->height;
    double mse = 0.0;
    for (size_t i = 0; i < total; i++) {
        double d = (double)orig->data[i] - (double)filtered->data[i];
        mse += d * d;
    }
    mse /= (double)total;
    if (mse == 0.0) return 99.99;
    return 10.0 * log10((255.0 * 255.0) / mse);
}

/* ─────────────────────────────────────────────
   Timer helper
   ───────────────────────────────────────────── */
static double now_ms(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1000.0 + ts.tv_nsec / 1e6;
}

/* ─────────────────────────────────────────────
   OpenCL kernel source (embedded string)
   Supports kernel_size up to 11 (max window = 121)
   ───────────────────────────────────────────── */
static const char *OCL_KERNEL_SRC =
"__kernel void median_filter(\n"
"    __global const uchar *src,\n"
"    __global       uchar *dst,\n"
"    int width, int height, int radius, int kernel_size)\n" /* <-- UBAH 'half' JADI 'radius' */
"{\n"
"    int x = get_global_id(0);\n"
"    int y = get_global_id(1);\n"
"    if (x >= width || y >= height) return;\n"
"\n"
"    uchar win[121];\n"          /* max 11x11 */
"    int cnt = 0;\n"
"    for (int ky = -radius; ky <= radius; ky++) {\n"        /* <-- UBAH DI SINI */
"        for (int kx = -radius; kx <= radius; kx++) {\n"    /* <-- UBAH DI SINI */
"            int ny = y + ky, nx = x + kx;\n"
"            if (ny < 0) ny = -ny;\n"
"            if (nx < 0) nx = -nx;\n"
"            if (ny >= height) ny = 2*height - ny - 2;\n"
"            if (nx >= width)  nx = 2*width  - nx - 2;\n"
"            win[cnt++] = src[ny * width + nx];\n"
"        }\n"
"    }\n"
"    /* insertion sort */\n"
"    int n = kernel_size * kernel_size;\n"
"    for (int i = 1; i < n; i++) {\n"
"        uchar key = win[i]; int j = i - 1;\n"
"        while (j >= 0 && win[j] > key) { win[j+1] = win[j]; j--; }\n"
"        win[j+1] = key;\n"
"    }\n"
"    dst[y * width + x] = win[n / 2];\n"
"}\n";


// OpenCL error helper
static void ocl_check(cl_int err, const char *msg)
{
    if (err != CL_SUCCESS) {
        fprintf(stderr, "[OpenCL Error] %s: %d\n", msg, err);
        exit(EXIT_FAILURE);
    }
}

// OpenCL Median Filter
Image *median_filter_opencl(const Image *src, int kernel_size,
                            double *gpu_exec_ms, double *total_ms)
{
    cl_int err;
    int half = kernel_size / 2;
    int W = src->width, H = src->height;
    size_t total_px = (size_t)W * H;

    // ── Platform & device ── 
    cl_platform_id platform;
    cl_uint num_platforms;
    err = clGetPlatformIDs(1, &platform, &num_platforms);
    ocl_check(err, "clGetPlatformIDs");

    cl_device_id device;
    err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, NULL);
    if (err == CL_DEVICE_NOT_FOUND) {
        printf("[OpenCL] No GPU found, falling back to CPU device.\n");
        err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_CPU, 1, &device, NULL);
    }
    ocl_check(err, "clGetDeviceIDs");

    char dev_name[256];
    clGetDeviceInfo(device, CL_DEVICE_NAME, sizeof(dev_name), dev_name, NULL);
    printf("[OpenCL] Using device: %s\n", dev_name);

    // ── Context & queue (with profiling) ── 
    cl_context context = clCreateContext(NULL, 1, &device, NULL, NULL, &err);
    ocl_check(err, "clCreateContext");

    cl_command_queue queue = clCreateCommandQueue(context, device,
                                CL_QUEUE_PROFILING_ENABLE, &err);
    ocl_check(err, "clCreateCommandQueue");

    // ── Program ── 
    cl_program program = clCreateProgramWithSource(context, 1,
                             &OCL_KERNEL_SRC, NULL, &err);
    ocl_check(err, "clCreateProgramWithSource");

    err = clBuildProgram(program, 1, &device, "-cl-fast-relaxed-math", NULL, NULL);
    if (err != CL_SUCCESS) {
        size_t log_sz;
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG,
                              0, NULL, &log_sz);
        char *log = (char *)malloc(log_sz);
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG,
                              log_sz, log, NULL);
        fprintf(stderr, "Build log:\n%s\n", log);
        free(log);
        exit(EXIT_FAILURE);
    }

    cl_kernel kernel = clCreateKernel(program, "median_filter", &err);
    ocl_check(err, "clCreateKernel");

    // ── Buffers ── 
    double t_host_start = now_ms();

    cl_mem buf_src = clCreateBuffer(context, CL_MEM_READ_ONLY  | CL_MEM_COPY_HOST_PTR,
                                    total_px, src->data, &err);
    ocl_check(err, "clCreateBuffer src");

    cl_mem buf_dst = clCreateBuffer(context, CL_MEM_WRITE_ONLY,
                                    total_px, NULL, &err);
    ocl_check(err, "clCreateBuffer dst");

    //── Kernel args ── 
    err  = clSetKernelArg(kernel, 0, sizeof(cl_mem), &buf_src);
    err |= clSetKernelArg(kernel, 1, sizeof(cl_mem), &buf_dst);
    err |= clSetKernelArg(kernel, 2, sizeof(int),    &W);
    err |= clSetKernelArg(kernel, 3, sizeof(int),    &H);
    err |= clSetKernelArg(kernel, 4, sizeof(int),    &half);
    err |= clSetKernelArg(kernel, 5, sizeof(int),    &kernel_size);
    ocl_check(err, "clSetKernelArg");

    // ── Enqueue NDRange ── 
    size_t gws[2] = { (size_t)W, (size_t)H };
    // Align to 16x16 work-group 
    size_t lws[2] = { 16, 16 };
    // round up global work size 
    gws[0] = ((gws[0] + 15) / 16) * 16;
    gws[1] = ((gws[1] + 15) / 16) * 16;

    cl_event ev;
    err = clEnqueueNDRangeKernel(queue, kernel, 2, NULL, gws, lws, 0, NULL, &ev);
    ocl_check(err, "clEnqueueNDRangeKernel");
    clWaitForEvents(1, &ev);

    // ── GPU kernel time via profiling ──
    cl_ulong t_start_ns, t_end_ns;
    clGetEventProfilingInfo(ev, CL_PROFILING_COMMAND_START,
                            sizeof(cl_ulong), &t_start_ns, NULL);
    clGetEventProfilingInfo(ev, CL_PROFILING_COMMAND_END,
                            sizeof(cl_ulong), &t_end_ns,   NULL);
    *gpu_exec_ms = (double)(t_end_ns - t_start_ns) / 1e6;
    clReleaseEvent(ev);

    // ── Read result ── 
    Image *dst_img = (Image *)malloc(sizeof(Image));
    dst_img->width  = W;
    dst_img->height = H;
    dst_img->maxval = src->maxval;
    dst_img->data   = (unsigned char *)malloc(total_px);

    err = clEnqueueReadBuffer(queue, buf_dst, CL_TRUE, 0,
                              total_px, dst_img->data, 0, NULL, NULL);
    ocl_check(err, "clEnqueueReadBuffer");

    *total_ms = now_ms() - t_host_start;

    // ── Cleanup ── 
    clReleaseMemObject(buf_src);
    clReleaseMemObject(buf_dst);
    clReleaseKernel(kernel);
    clReleaseProgram(program);
    clReleaseCommandQueue(queue);
    clReleaseContext(context);

    return dst_img;
}


//   MAIN
int main(int argc, char *argv[])
{
    if (argc < 5) {
        fprintf(stderr,
            "Usage: %s <input.pgm> <output.pgm> <noise_ratio> <kernel_size>\n"
            "  noise_ratio : 0.0 – 1.0  (e.g. 0.05 = 5%%)\n"
            "  kernel_size : odd integer, max 11 (e.g. 3, 5, 7)\n",
            argv[0]);
        return EXIT_FAILURE;
    }

    const char *in_file  = argv[1];
    const char *out_file = argv[2];
    double noise_ratio   = atof(argv[3]);
    int    kernel_size   = atoi(argv[4]);

    if (kernel_size % 2 == 0) { kernel_size++; printf("[Warn] kernel_size adjusted to %d\n", kernel_size); }
    if (kernel_size > 11)     { kernel_size = 11; printf("[Warn] kernel_size capped at 11\n"); }

    printf("=== OpenCL Median Filter ===\n");
    printf("Input        : %s\n", in_file);
    printf("Output       : %s\n", out_file);
    printf("Noise ratio  : %.1f%%\n", noise_ratio * 100.0);
    printf("Kernel size  : %dx%d\n\n", kernel_size, kernel_size);

    // 1. Read 
    double t0 = now_ms();
    Image *orig = read_pgm(in_file);
    if (!orig) return EXIT_FAILURE;
    printf("[Step 1] Image loaded: %dx%d pixels (%.2f ms)\n",
           orig->width, orig->height, now_ms() - t0);

    // 2. Clone & add noise 
    Image *noisy = (Image *)malloc(sizeof(Image));
    noisy->width  = orig->width;
    noisy->height = orig->height;
    noisy->maxval = orig->maxval;
    noisy->data   = (unsigned char *)malloc((size_t)orig->width * orig->height);
    memcpy(noisy->data, orig->data, (size_t)orig->width * orig->height);

    t0 = now_ms();
    add_salt_pepper_noise(noisy, noise_ratio);
    printf("[Step 2] Noise added (%.2f ms)\n\n", now_ms() - t0);

    // 3. Save noisy image 
    {
        char noisy_name[512];
        snprintf(noisy_name, sizeof(noisy_name), "noisy_%s", out_file);
        write_pgm(noisy_name, noisy);
        printf("[Step 3] Noisy image saved: %s\n", noisy_name);
    }

    // 4. OpenCL filter 
    printf("\n[Step 4] Running OpenCL Median Filter...\n");
    double gpu_exec_ms, total_ms;
    Image *filtered = median_filter_opencl(noisy, kernel_size,
                                           &gpu_exec_ms, &total_ms);

    // 5. Save result 
    write_pgm(out_file, filtered);

    // 6. PSNR 
    double psnr_noisy    = compute_psnr(orig, noisy);
    double psnr_filtered = compute_psnr(orig, filtered);

    printf("\n\n");
    printf("  RESULTS - OpenCL\n");
    printf("------------------------------\n");
    printf("  Image size        : %d x %d\n", orig->width, orig->height);
    printf("  Kernel size       : %d x %d\n", kernel_size, kernel_size);
    printf("  Noise ratio       : %.1f%%\n", noise_ratio * 100.0);
    printf("  GPU kernel time   : %.4f ms  (profiling)\n", gpu_exec_ms);
    printf("  Total time (H2D+kernel+D2H) : %.4f ms\n", total_ms);
    printf("  PSNR (noisy)      : %.2f dB\n", psnr_noisy);
    printf("  PSNR (filter)     : %.2f dB\n", psnr_filtered);
    printf("  Output saved      : %s\n", out_file);
    printf("------------------------------\n");

    free_image(orig);
    free_image(noisy);
    free_image(filtered);
    return EXIT_SUCCESS;
}
