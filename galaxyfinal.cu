#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <Windows.h>
#include <stdint.h>
#ifndef M_PI
#define M_PI 3.1415926535f
#endif

#define totaldegrees 180
#define binsperdegree 4
#define threadsperblock 1024

// data for the real galaxies will be read into these arrays
float *ra_real, *decl_real;
// number of real galaxies
int NoofReal;

// data for the simulated random galaxies will be read into these arrays
float *ra_sim, *decl_sim;
// number of simulated random galaxies
int NoofSim;

unsigned int *histogramDR, *histogramDD, *histogramRR;
unsigned int *d_histogram;

__device__ float calculateAngularSeparation(float ra1, float dec1, float ra2, float dec2) {
    // Convert angles from arcminutes to radians
    float dec1_rad = dec1 * M_PI / (180.0f * 60);
    float dec2_rad = dec2 * M_PI / (180.0f * 60);
    float ra_diff_rad = (ra1 - ra2) * M_PI / (180.0f * 60);

    // Compute the argument for the acos function
    float acos_arg = sinf(dec1_rad) * sinf(dec2_rad) + cosf(dec1_rad) * cosf(dec2_rad) * cosf(ra_diff_rad);

    // Clamp the argument to be within [-1.0f, +1.0f]
    acos_arg = fminf(fmaxf(acos_arg, -1.0f), 1.0f);

    // Compute the angular separation and convert it back to degrees
    float angle = acosf(acos_arg);
    return angle * 180.0f / M_PI;
}

__global__ void calculateHistograms(float *ra1, float *dec1, int size1, float *ra2, float *dec2, int size2, unsigned int *histogram, int numBins)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < size1)
    {
        for (int j = 0; j < size2; j++)
        {
            float deltaRA = fabsf(ra1[idx] - ra2[j]);
            float deltaDec = fabsf(dec1[idx] - dec2[j]);
            float angularSeparation = calculateAngularSeparation(ra1[idx], dec1[idx], ra2[j], dec2[j]);

            // Determine the histogram bin (this is a simple example, adjust as needed)
            int bin = (int)(angularSeparation * binsperdegree);
            if (bin < numBins)
            {
                atomicAdd(&histogram[bin], 1); // Use atomicAdd to avoid race conditions
            }
        }
    }
}

int main(int argc, char *argv[])
{
    int i;
    int noofblocks;
    int readdata(char *argv1, char *argv2);
    int getDevice(int deviceno);
    long int histogramDRsum, histogramDDsum, histogramRRsum;
    double w;
    double kerneltime;
	LARGE_INTEGER frequency;
    LARGE_INTEGER start, end;
    cudaError_t myError;
    FILE *outfil;
    long long int histsum = 0;

    if (argc != 4)
    {
        printf("Usage: a.out real_data random_data output_data\n");
        return (-1);
    }

    if (getDevice(0) != 0)
        return (-1);

    if (readdata(argv[1], argv[2]) != 0)
        return (-1);

    // allocate mameory on the GPU
    float *d_ra_real, *d_decl_real;
    float *d_ra_sim, *d_decl_sim;

    cudaMalloc(&d_ra_real, NoofReal * sizeof(float));
    cudaMalloc(&d_decl_real, NoofReal * sizeof(float));
    cudaMalloc(&d_ra_sim, NoofSim * sizeof(float));
    cudaMalloc(&d_decl_sim, NoofSim * sizeof(float));

    QueryPerformanceFrequency(&frequency);
    kerneltime = 0.0f;
    QueryPerformanceCounter(&start);

    // copy data to the GPU
    cudaMemcpy(d_ra_real, ra_real, NoofReal * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_decl_real, decl_real, NoofReal * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_ra_sim, ra_sim, NoofSim * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_decl_sim, decl_sim, NoofSim * sizeof(float), cudaMemcpyHostToDevice);

    // Calculate the total number of bins
    int numBins = totaldegrees * binsperdegree;

    // Allocate memory for the histograms on the GPU
    unsigned int *d_histogramDD, *d_histogramDR, *d_histogramRR;
    cudaMallocManaged(&d_histogramDD, numBins * sizeof(unsigned int));
    cudaMallocManaged(&d_histogramDR, numBins * sizeof(unsigned int));
    cudaMallocManaged(&d_histogramRR, numBins * sizeof(unsigned int));

    // Initialize histograms to zero
    cudaMemset(d_histogramDD, 0, numBins * sizeof(unsigned int));
    cudaMemset(d_histogramDR, 0, numBins * sizeof(unsigned int));
    cudaMemset(d_histogramRR, 0, numBins * sizeof(unsigned int));

    // Launch the kernels
    // DD
    calculateHistograms<<<(NoofReal + threadsperblock - 1) / threadsperblock, threadsperblock>>>(d_ra_real, d_decl_real, NoofReal, d_ra_real, d_decl_real, NoofReal, d_histogramDD, numBins);
    cudaDeviceSynchronize();

    // DR
    calculateHistograms<<<(NoofReal + threadsperblock - 1) / threadsperblock, threadsperblock>>>(d_ra_real, d_decl_real, NoofReal, d_ra_sim, d_decl_sim, NoofSim, d_histogramDR, numBins);
    cudaDeviceSynchronize();

    // RR
    calculateHistograms<<<(NoofSim + threadsperblock - 1) / threadsperblock, threadsperblock>>>(d_ra_sim, d_decl_sim, NoofSim, d_ra_sim, d_decl_sim, NoofSim, d_histogramRR, numBins);
    cudaDeviceSynchronize();

    // copy the results back to the CPU
    // Allocate memory on the host for the histograms
    unsigned int *histogramDD = (unsigned int *)malloc(numBins * sizeof(unsigned int));
    unsigned int *histogramDR = (unsigned int *)malloc(numBins * sizeof(unsigned int));
    unsigned int *histogramRR = (unsigned int *)malloc(numBins * sizeof(unsigned int));

    // Copy the histograms from device to host
    cudaMemcpy(histogramDD, d_histogramDD, numBins * sizeof(unsigned int), cudaMemcpyDeviceToHost);
    cudaMemcpy(histogramDR, d_histogramDR, numBins * sizeof(unsigned int), cudaMemcpyDeviceToHost);
    cudaMemcpy(histogramRR, d_histogramRR, numBins * sizeof(unsigned int), cudaMemcpyDeviceToHost);

    // calculate omega values on the CPU
    // Assuming omega calculation is based on the formula Wi(O) = (DDi - 2*DRi + RRi) / RRi
    float *omega = (float *)malloc(numBins * sizeof(float));

    for (int i = 0; i < numBins; i++)
    {
        if (histogramRR[i] != 0)
        { // Prevent division by zero
            omega[i] = (float)(histogramDD[i] - 2.0f * histogramDR[i] + histogramRR[i]) / histogramRR[i];
        }
        else
        {
            omega[i] = 0.0f;
        }
        histsum += (long long int)histogramDD[i];
        histsum += (long long int)histogramDR[i];
        histsum += (long long int)histogramRR[i];
        printf("%4d: %f %u %u %u\n", i, omega[i], histogramDD[i], histogramDR[i], histogramRR[i]);

    }
    printf("Histogram DD Sum: %lld\n", histogramDRsum);
    printf("Histogram DR Sum: %lld\n", histogramDDsum);
    printf("Histogram RR Sum: %lld\n", histogramRRsum);
    printf("Histogram Sum: %lld\n", histsum);
    if (histsum == 10000000000)
    {
        printf("The histogram sum is exactly 10 billion.\n");
    }
    else
    {
        printf("The histogram sum is not 10 billion.\n");
    }

    // Omega values are now calculated and stored in omega array

     QueryPerformanceCounter(&end);

    // Calculate the elapsed time in seconds
    kerneltime = (double)(end.QuadPart - start.QuadPart) / frequency.QuadPart;

    printf("Total Kernel Execution Time: %f seconds\n", kerneltime);

    // Free GPU memory
    cudaFree(d_ra_real);
    cudaFree(d_decl_real);
    cudaFree(d_ra_sim);
    cudaFree(d_decl_sim);
    cudaFree(d_histogramDD);
    cudaFree(d_histogramDR);
    cudaFree(d_histogramRR);

    // Free CPU memory
    free(histogramDD);
    free(histogramDR);
    free(histogramRR);
    free(omega);

    return (0);
}

int readdata(char *argv1, char *argv2)
{
    int i, linecount;
    char inbuf[180];
    double ra, dec, phi, theta, dpi;
    FILE *infil;

    printf("   Assuming input data is given in arc minutes!\n");
    // spherical coordinates phi and theta:
    // phi   = ra/60.0 * dpi/180.0;
    // theta = (90.0-dec/60.0)*dpi/180.0;

    dpi = acos(-1.0);
    infil = fopen(argv1, "r");
    if (infil == NULL)
    {
        printf("Cannot open input file %s\n", argv1);
        return (-1);
    }

    // read the number of galaxies in the input file
    int announcednumber;
    if (fscanf(infil, "%d\n", &announcednumber) != 1)
    {
        printf(" cannot read file %s\n", argv1);
        return (-1);
    }
    linecount = 0;
    while (fgets(inbuf, 180, infil) != NULL)
        ++linecount;
    rewind(infil);

    if (linecount == announcednumber)
        printf("   %s contains %d galaxies\n", argv1, linecount);
    else
    {
        printf("   %s does not contain %d galaxies but %d\n", argv1, announcednumber, linecount);
        return (-1);
    }

    NoofReal = linecount;
    ra_real = (float *)calloc(NoofReal, sizeof(float));
    decl_real = (float *)calloc(NoofReal, sizeof(float));

    // skip the number of galaxies in the input file
    if (fgets(inbuf, 180, infil) == NULL)
        return (-1);
    i = 0;
    while (fgets(inbuf, 80, infil) != NULL)
    {
        if (sscanf(inbuf, "%lf %lf", &ra, &dec) != 2)
        {
            printf("   Cannot read line %d in %s\n", i + 1, argv1);
            fclose(infil);
            return (-1);
        }
        ra_real[i] = (float)ra;
        decl_real[i] = (float)dec;
        ++i;
    }

    fclose(infil);

    if (i != NoofReal)
    {
        printf("   Cannot read %s correctly\n", argv1);
        return (-1);
    }

    infil = fopen(argv2, "r");
    if (infil == NULL)
    {
        printf("Cannot open input file %s\n", argv2);
        return (-1);
    }

    if (fscanf(infil, "%d\n", &announcednumber) != 1)
    {
        printf(" cannot read file %s\n", argv2);
        return (-1);
    }
    linecount = 0;
    while (fgets(inbuf, 80, infil) != NULL)
        ++linecount;
    rewind(infil);

    if (linecount == announcednumber)
        printf("   %s contains %d galaxies\n", argv2, linecount);
    else
    {
        printf("   %s does not contain %d galaxies but %d\n", argv2, announcednumber, linecount);
        return (-1);
    }

    NoofSim = linecount;
    ra_sim = (float *)calloc(NoofSim, sizeof(float));
    decl_sim = (float *)calloc(NoofSim, sizeof(float));

    // skip the number of galaxies in the input file
    if (fgets(inbuf, 180, infil) == NULL)
        return (-1);
    i = 0;
    while (fgets(inbuf, 80, infil) != NULL)
    {
        if (sscanf(inbuf, "%lf %lf", &ra, &dec) != 2)
        {
            printf("   Cannot read line %d in %s\n", i + 1, argv2);
            fclose(infil);
            return (-1);
        }
        ra_sim[i] = (float)ra;
        decl_sim[i] = (float)dec;
        ++i;
    }

    fclose(infil);

    if (i != NoofSim)
    {
        printf("   Cannot read %s correctly\n", argv2);
        return (-1);
    }

    return (0);
}

int getDevice(int deviceNo)
{

    int deviceCount;
    cudaGetDeviceCount(&deviceCount);
    printf("   Found %d CUDA devices\n", deviceCount);
    if (deviceCount < 0 || deviceCount > 128)
        return (-1);
    int device;
    for (device = 0; device < deviceCount; ++device)
    {
        cudaDeviceProp deviceProp;
        cudaGetDeviceProperties(&deviceProp, device);
        printf("      Device %s                  device %d\n", deviceProp.name, device);
        printf("         compute capability            =        %d.%d\n", deviceProp.major, deviceProp.minor);
        printf("         totalGlobalMemory             =       %.2lf GB\n", deviceProp.totalGlobalMem / 1000000000.0);
        printf("         l2CacheSize                   =   %8d B\n", deviceProp.l2CacheSize);
        printf("         regsPerBlock                  =   %8d\n", deviceProp.regsPerBlock);
        printf("         multiProcessorCount           =   %8d\n", deviceProp.multiProcessorCount);
        printf("         maxThreadsPerMultiprocessor   =   %8d\n", deviceProp.maxThreadsPerMultiProcessor);
        printf("         sharedMemPerBlock             =   %8d B\n", (int)deviceProp.sharedMemPerBlock);
        printf("         warpSize                      =   %8d\n", deviceProp.warpSize);
        printf("         clockRate                     =   %8.2lf MHz\n", deviceProp.clockRate / 1000.0);
        printf("         maxThreadsPerBlock            =   %8d\n", deviceProp.maxThreadsPerBlock);
        printf("         asyncEngineCount              =   %8d\n", deviceProp.asyncEngineCount);
        printf("         f to lf performance ratio     =   %8d\n", deviceProp.singleToDoublePrecisionPerfRatio);
        printf("         maxGridSize                   =   %d x %d x %d\n",
               deviceProp.maxGridSize[0], deviceProp.maxGridSize[1], deviceProp.maxGridSize[2]);
        printf("         maxThreadsDim in thread block =   %d x %d x %d\n",
               deviceProp.maxThreadsDim[0], deviceProp.maxThreadsDim[1], deviceProp.maxThreadsDim[2]);
        printf("         concurrentKernels             =   ");
        if (deviceProp.concurrentKernels == 1)
            printf("     yes\n");
        else
            printf("    no\n");
        printf("         deviceOverlap                 =   %8d\n", deviceProp.deviceOverlap);
        if (deviceProp.deviceOverlap == 1)
            printf("            Concurrently copy memory/execute kernel\n");
    }

    cudaSetDevice(deviceNo);
    cudaGetDevice(&device);
    if (device != 0)
        printf("   Unable to set device 0, using %d instead", device);
    else
        printf("   Using CUDA device %d\n\n", device);

    return (0);
}
