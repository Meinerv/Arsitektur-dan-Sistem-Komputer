#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

// Image struct
typedef struct {
    int   width, height, maxval;
    unsigned char *data;
} Image;

// PGM I/O
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


// Salt-and-Pepper Noise
void add_salt_pepper_noise(Image *img, double noise_ratio)
{
    srand((unsigned)time(NULL));
    size_t total = (size_t)img->width * img->height;
    size_t n     = (size_t)(total * noise_ratio);

    for (size_t i = 0; i < n / 2; i++) {   // salt 
        size_t idx = (size_t)rand() % total;
        img->data[idx] = 255;
    }
    for (size_t i = 0; i < n / 2; i++) {   // pepper 
        size_t idx = (size_t)rand() % total;
        img->data[idx] = 0;
    }
    printf("[Noise] Salt-and-pepper noise added (ratio=%.2f%%)\n",
           noise_ratio * 100.0);
}

// Insertion sort on a small window array
static void insertion_sort(unsigned char *a, int n)
{
    for (int i = 1; i < n; i++) {
        unsigned char key = a[i];
        int j = i - 1;
        while (j >= 0 && a[j] > key) { a[j + 1] = a[j]; j--; }
        a[j + 1] = key;
    }
}


//   Sequential Median Filter
Image *median_filter_sequential(const Image *src, int kernel_size)
{
    int half = kernel_size / 2;
    int W = src->width, H = src->height;
    int win_sz = kernel_size * kernel_size;

    Image *dst = (Image *)malloc(sizeof(Image));
    dst->width  = W;
    dst->height = H;
    dst->maxval = src->maxval;
    dst->data   = (unsigned char *)malloc((size_t)W * H);

    unsigned char *win = (unsigned char *)malloc(win_sz);

    for (int y = 0; y < H; y++) {
        for (int x = 0; x < W; x++) {
            int cnt = 0;
            for (int ky = -half; ky <= half; ky++) {
                for (int kx = -half; kx <= half; kx++) {
                    int ny = y + ky, nx = x + kx;

                    if (ny < 0) ny = -ny;
                    if (nx < 0) nx = -nx;
                    if (ny >= H) ny = 2 * H - ny - 2;
                    if (nx >= W) nx = 2 * W - nx - 2;
                    win[cnt++] = src->data[ny * W + nx];
                }
            }
            insertion_sort(win, win_sz);
            dst->data[y * W + x] = win[win_sz / 2];
        }
    }

    free(win);
    return dst;
}

//   PSNR metric
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


//   Timer helper
static double now_ms(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1000.0 + ts.tv_nsec / 1e6;
}


//   MAIN
int main(int argc, char *argv[])
{
    if (argc < 5) {
        fprintf(stderr,
            "Usage: %s <input.pgm> <output.pgm> <noise_ratio> <kernel_size>\n"
            "  noise_ratio  : 0.0 – 1.0  (e.g. 0.05 = 5%%)\n"
            "  kernel_size  : odd integer (e.g. 3, 5, 7)\n",
            argv[0]);
        return EXIT_FAILURE;
    }

    const char *in_file  = argv[1];
    const char *out_file = argv[2];
    double noise_ratio   = atof(argv[3]);
    int    kernel_size   = atoi(argv[4]);

    if (kernel_size % 2 == 0) { kernel_size++; printf("[Warn] kernel_size adjusted to %d\n", kernel_size); }

    printf("=== Sequential Median Filter ===\n");
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

    // 4. Sequential filter 
    printf("[Step 4] Running Sequential Median Filter...\n");
    t0 = now_ms();
    Image *filtered = median_filter_sequential(noisy, kernel_size);
    double elapsed = now_ms() - t0;

    // 5. Save result 
    write_pgm(out_file, filtered);

    // 6. PSNR 
    double psnr_noisy    = compute_psnr(orig, noisy);
    double psnr_filtered = compute_psnr(orig, filtered);

    printf("\n\n");
    printf("  RESULTS - Sequential\n");
    printf("------------------------------\n");
    printf("  Image size   : %d x %d\n", orig->width, orig->height);
    printf("  Kernel size  : %d x %d\n", kernel_size, kernel_size);
    printf("  Noise ratio  : %.1f%%\n", noise_ratio * 100.0);
    printf("  Exec time    : %.4f ms\n", elapsed);
    printf("  PSNR (noisy) : %.2f dB\n", psnr_noisy);
    printf("  PSNR (filter): %.2f dB\n", psnr_filtered);
    printf("  Output saved : %s\n", out_file);
    printf("------------------------------\n");

    free_image(orig);
    free_image(noisy);
    free_image(filtered);
    return EXIT_SUCCESS;
}
