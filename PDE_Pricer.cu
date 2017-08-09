#include <iostream> 
#include <exception>
#include <device_launch_parameters.h>
#include <cuda_runtime.h>
#include <string>

// Threads per Block, 256 is shown to give optimal results for the Kepler architecture.
#define TPB 256 

// Used for SOR/PSOR Kernel, 6 iterations have shown to provide with good accuracy at moderate cost with Omega = 1.3.
#define PSOR_ITERATIONS 6
#define PSOR_OMEGA 1.3f

// Bench CUDA Kernels using a templated C++ function
template <class fn>
void Bench(int Repetitions,       // The number of times the benchmark has to be launched.
           std::string Message,   // A message to be appended to the output (can be the name of the module, or the params)
           fn&& Function) {       // The function.
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    float total_elapsed_time = 0.f;
    // Sequentially invoke the function.
    for (int i = 0; i < Repetitions; ++i) {
        cudaEventRecord(start);
        Function();
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        float iteration_elapsed_time = 0.f;
        cudaEventElapsedTime(&iteration_elapsed_time, start, stop);
        total_elapsed_time += iteration_elapsed_time;
    }
    // Display the average of running times
    std::cout << Message << ": Time Elapsed = " << total_elapsed_time / double(Repetitions) << " ms \n";
}

// Wrapper around cudaMalloc().
inline float * cudaAlloc(size_t n) {
    float* Ptr = 0;
    cudaMalloc(&Ptr, n * sizeof(float));
    if (!Ptr)
        throw std::bad_alloc();
    return Ptr;
}

// Kernel to initialize Saxis (Saxis[i] = Smin + i * deltaS)
__global__ void Saxis_Initialize(float * Saxis,
                                 const float Smin,
                                 const float deltaS,
                                 const int Length) {
    int i = threadIdx.x + blockDim.x * blockIdx.x;
    if (i < Length) {
        Saxis[i] = Smin + i * deltaS;
    }
}

// Computes P[i] = max(0, S[i] - K) for i in 0:(SizeS-1)
__global__ void Call(float * __restrict__ P,        // The payoff (output)
                     const float * __restrict__ S,  // The underlying level (input)
                     const float K,                 // The option Strike (input)
                     const int SizeS) {             // Size of the S-axis (input)
    int i = threadIdx.x + blockDim.x * blockIdx.x;
    if (i < SizeS) {
        P[i] = max(S[i] - K, 0.f);
    }
}

// Refer to Call.
__global__ void Put(float * __restrict__ P,
                    const float * __restrict__ S,
                    const float K,
                    const int SizeS) {
    int i = threadIdx.x + blockDim.x * blockIdx.x;
    if (i < SizeS) {
        P[i] = max(K - S[i], 0.f);
    }
}

// Computes the Dirichlet boundary of an European Call for the upper-bound of the S-axis, for t in 0:(SizeT-1).
__global__ void MaxBoundary_EuropeanCall(float * __restrict__ P,        // The Dirichlet Boundary (output)
                                         const float maxS,              // Maximum of S-axis (input, should be K + 6 * Vol * sqrt(T))
                                         const float * __restrict__ r,  // The interest rate (input)
                                         const float * __restrict__ q,  // The dividend yield (input)
                                         const float K,                 // The option strike (input)
                                         const int SizeT) {             // Size of the T-axis (input) 
    int i = threadIdx.x + blockDim.x * blockIdx.x;
    if (i < SizeT) {
        P[i] = maxS *__expf(q[i]) - K * __expf(r[i]);
    }
}

// Refer to MaxBoundary_EuropeanCall.
__global__ void MaxBoundary_AmericanCall(float * __restrict__ P,
                                         const float maxS,
                                         const float K,
                                         const int SizeT) {
    int i = threadIdx.x + blockDim.x * blockIdx.x;
    if (i < SizeT) {
        P[i] = maxS - K;
    }
}

// Refer to MaxBoundary_EuropeanCall.
__global__ void MinBoundary_EuropeanPut(float * __restrict__ P,
                                        const float minS,
                                        const float * __restrict__ r,
                                        const float * __restrict__ q,
                                        const float K,
                                        const int SizeT) {
    int i = threadIdx.x + blockDim.x * blockIdx.x;
    if (i < SizeT) {
        P[i] = -minS * __expf(q[i]) + K * __expf(r[i]);
    }
}

// Refer to MaxBoundary_EuropeanCall.
__global__ void MinBoundary_AmericanPut(float * __restrict__ P,
                                        const float minS,
                                        const float K,
                                        const int SizeT) {
    int i = threadIdx.x + blockDim.x * blockIdx.x;
    if (i < SizeT) {
        P[i] = -minS + K;
    }
}

// Handler for Explicit kernels. It returns the value of the derivative at T-1 given the value of the derivative at T and 
// local parameters.
__inline__ __device__ float ExplicitHandler(const float * __restrict__ pastV,   // Value of the derivative at T 
                                            const float S,                      // The underlying level
                                            const float r,                      // Instantaneous interest rate
                                            const float q,                      // Instantanous dividend yield 
                                            const float sigma,                  // Local volatility
                                            const float deltaT,                 // Time step
                                            const float deltaS,                 // Space step
                                            const float minBoundary,            // Min Dirichlet boundary at T
                                            const float maxBoundary,            // Max Dirichlet boundary at T
                                            const int i,                        // Space index
                                            const int SizeS) {                  // Size of S-axis
    // Computation of the BS-Dupire coefficients
    float Base = (deltaT * S) / (2.f * deltaS);
    float Order1 = Base * (r - q);
    float Order2 = (Base * sigma * sigma * S) / deltaS;
    float Alpha = Order1 - Order2;
    float Beta = 1.f + r * deltaT + 2.f * Order2;
    float Gamma = -Order1 - Order2;
    float Mid = pastV[i];
    float Down, Up;
    if (i == 0) {
        // Lower bound.
        Down = minBoundary;
        Up = pastV[i + 1];
    }
    else if (i == SizeS - 1) {
        // Upper bound.
        Down = pastV[i - 1];
        Up = maxBoundary;
    }
    else {
        // Tridiagonal case.
        Down = pastV[i - 1];
        Up = pastV[i + 1];
    }
    return (Alpha * Down + Beta * Mid + Gamma * Up);
}

// Kernel for explicit scheme applied to American options.
__global__ void ExplicitKernel_American(float * __restrict__ Grid,		// The PDE grid (output)
                                        const float * __restrict__ S,		// Value of asset
                                        const float * __restrict__ r,		// Risk-free rate
                                        const float * __restrict__ q,		// Dividend yield
                                        const float * __restrict__ sigma,		// Volatility
                                        const float deltaT,			// Time precision
                                        const float deltaS,			// Asset precision
                                        const float * __restrict__ minBoundary,         // Values of derivative for S = minS
                                        const float * __restrict__ maxBoundary,         // Values of derivative for S = maxS
                                        const int sizeS, const int sizeT) {		// The number of elements for S-axis and T-axis
    int i = threadIdx.x + blockDim.x * blockIdx.x;
    if (i < sizeS) {
        float* pastV = Grid;
        float* currV = pastV + sizeS;
        float* Payoff = Grid;
        // Iterate over time steps. j = 0 equals the maturity, and j = sizeT - 1 is today.
        for (int j = 1; j < sizeT; ++j) {
            float X = ExplicitHandler(pastV,
                                      S[i],
                                      r[j],
                                      q[j],
                                      sigma[i + j * sizeS],
                                      deltaT,
                                      deltaS,
                                      minBoundary[j],
                                      maxBoundary[j],
                                      i,
                                      sizeS);
            // Take the maximum value btw. Payoff and derivative value
            currV[i] = max(X, Payoff[i]);
            // Update currV, pastV and synchronize within block.
            pastV = currV;
            currV += sizeS;
            // Synchronize the block so all threads are in the same time step.
            __syncthreads();
        }
    }
}

// Kernel for explicit scheme applied to European options.
__global__ void ExplicitKernel_European(float * __restrict__ Grid,		// The PDE grid (output)
                                        const float * __restrict__ S,		// Value of asset
                                        const float * __restrict__ r,		// Risk-free rate
                                        const float * __restrict__ q,		// Dividend yield
                                        const float * __restrict__ sigma,		// Volatility
                                        const float deltaT,			// Time precision
                                        const float deltaS,			// Asset precision
                                        const float * __restrict__ minBoundary,         // Values of derivative for S = minS
                                        const float * __restrict__ maxBoundary,         // Values of derivative for S = maxS
                                        const int sizeS, const int sizeT) {		// The number of elements for S-axis and T-axis
    int i = threadIdx.x + blockDim.x * blockIdx.x;
    if (i < sizeS) {
        float* pastV = Grid;
        float* currV = pastV + sizeS;
        // Iterate over time steps. j = 0 equals the maturity, and j = sizeT - 1 is today.
        for (int j = 1; j < sizeT; ++j) {
            currV[i] = ExplicitHandler(pastV,
                                       S[i],
                                       r[j],
                                       q[j],
                                       sigma[i + j * sizeS],
                                       deltaT,
                                       deltaS,
                                       minBoundary[j],
                                       maxBoundary[j],
                                       i,
                                       sizeS);
            // Update currV, pastV and synchronize within block.
            pastV = currV;
            currV += sizeS;
            // Synchronize the block so all threads are in the same time step.
            __syncthreads();
        }
    }
}

// Handler for Implicit kernels. Basically computes the tridiagonal system given the value of the derivative at T and 
// local parameters and returns new_val
__inline__ __device__ float ImplicitHandler(float * __restrict__ newV,     // Is set to new_val. (output)
                                            float * __restrict__ Array,    // Shared memory. Is set to new_val (output)
                                            float & __restrict__ _LD,      // The lower diagonal element (output)
                                            float & __restrict__ _D,       // The diagonal element (output)
                                            float & __restrict__ _UD,      // The upper diagonal element (output)
                                            const float sigma,             // Local volatility
                                            const float new_val,           // The value of derivative at T
                                            const float S,                 // Underlying level
                                            const float deltaT,            // Time step
                                            const float deltaS,            // Space step
                                            const float _r,                // Instantaneous interest rate
                                            const float _q,                // Instantaneous dividend yield
                                            const float minBoundary,       // Min Dirichlet boundary at T
                                            const float maxBoundary,       // Max Dirichlet boundary at T
                                            const int i,                   // Space index
                                            const int SizeS) {             // Size of S-axis
    // Computation of the tridiagnonal value given the BS-Dupire parameters
    newV[i] = new_val;
    float old_val = new_val;
    float Base = (deltaT * S) / (2.f * deltaS);
    float Order1 = Base * (_r - _q);
    float Order2 = (Base * sigma * sigma * S) / deltaS;
    _LD = Order2 - Order1;
    _D = 1.f - _r * deltaT - 2.f * Order2;
    _UD = Order2 + Order1;
    // Apply Dirichlet boundaries.
    if (i == 0) {
        old_val -= _LD * minBoundary;
    }
    else if (i == SizeS - 1) {
        old_val -= _UD * maxBoundary;
    }
    // Shared memory optimization...
    Array[i] = new_val;
    return old_val;
}

// Computes an iteration of the SOR algorithm. 
__inline__ __device__ float SORHandler(const float * __restrict__ Array,     // "Neighbours" previous iteration values
                                       const float old_val,                  // Previous iteration value
                                       const float _LD,                      // The lower diagonal element
                                       const float _D,                       // The diagonal element
                                       const float _UD,                      // The upper diagonal element
                                       const int i,                          // Space index
                                       const int SizeS) {                    // Size of S-axis
    float Up = (i > SizeS - 2) ? 0.f : Array[i + 1];
    float Down = (i < 1) ? 0.f : Array[i - 1];
    float r = (1.f - PSOR_OMEGA) * Array[i] + (PSOR_OMEGA / _D) * (old_val - _LD * Down - _UD * Up);
    return r;
}

// Kernel for implicit scheme applied to European options.
__global__ void ImplicitKernel_European(float * __restrict__ NewValues,                   // The PDE grid, starting at T-1 (it is offset by SizeS) (output)
                                        const float * __restrict__ R,                     // Instantaneous interest rate
                                        const float * __restrict__ Q,                     // Instantaneous dividend yield
                                        const float * __restrict__ _Sigma,                // Local Volatility
                                        const float * __restrict__ _S,                    // S-axis
                                        const float * __restrict__ minBoundary,           // Min Dirichlet boundary at each time step
                                        const float * __restrict__ maxBoundary,           // Max Dirichlet boundary at each time step
                                        const float deltaT,                               // Time step
                                        const float deltaS,                               // Space step
                                        const int SizeS, const int SizeT) {               // Sizes of S- and T-axes
    extern __shared__ float Array[];
    int i = threadIdx.x + blockDim.x * blockIdx.x;
    float _D, _UD, _LD, old_val, S;
    if (i < SizeS) {
        S = _S[i];
        // Iterate over time steps. j = 0 equals the maturity, and j = sizeT - 1 is today.
        for (int j = 1; j < SizeT; ++j) {
            old_val = ImplicitHandler(NewValues, Array, _LD, _D, _UD,
                                      _Sigma[i], NewValues[i - SizeS], S,
                                      deltaT, deltaS, R[j], Q[j], minBoundary[j], 
                                      maxBoundary[j], i, SizeS);
            // This synchronization is used so all threads are in the same time step!
            __syncthreads();
            // Case "Red" : even threads
            if (!(i & 1)) {
                // Iterate PSOR_ITERATIONS times
                for (int k = 0; k < PSOR_ITERATIONS; ++k) {
                    Array[i] = SORHandler(Array, old_val, _LD, _D, _UD, i, SizeS);
                    // Synchronize Black and Red threads.
                    __syncthreads();
                    // While "Black" threads are computing, do nothing.
                    // Synchronize Black and Red threads.
                    __syncthreads();
                }
            }
            // Case "Black" : odd threads
            else {
                // Iterate PSOR_ITERATIONS times
                for (int k = 0; k < PSOR_ITERATIONS; ++k) {
                    // While "Red" threads are computing, do nothing.
                    // Synchronize Black and Red threads.
                    __syncthreads();
                    Array[i] = SORHandler(Array, old_val, _LD, _D, _UD, i, SizeS);
                    // Synchronize Black and Red threads.
                    __syncthreads();
                }
            }
            NewValues[i] = Array[i];
            NewValues += SizeS;
            _Sigma += SizeS;
        }
    }
}
        
// Kernel for implicit scheme applied to American options. Refer to ImplicitKernel_European for parameters and comments.
__global__ void ImplicitKernel_American(float * __restrict__ NewValues,
                                        const float * __restrict__ R,
                                        const float * __restrict__ Q,
                                        const float * __restrict__ _Sigma,
                                        const float * __restrict__ _S,
                                        const float * __restrict__ minBoundary,
                                        const float * __restrict__ maxBoundary,
                                        const float deltaT,
                                        const float deltaS,
                                        const int SizeS, const int SizeT) {
    extern __shared__ float Array[];
    int i = threadIdx.x + blockDim.x * blockIdx.x;
    float r, _Payoff, _D, _UD, _LD;
    float old_val, S;
    if (i < SizeS) {
        _Payoff = NewValues[i - SizeS];
        S = _S[i];
        // Iterate for each time step using an implicit PSOR scheme.
        for (int j = 1; j < SizeT; ++j) {
            old_val = ImplicitHandler(NewValues, Array, _LD, _D, _UD,
                                      _Sigma[i], NewValues[i - SizeS], S,
                                      deltaT, deltaS, R[j], Q[j], minBoundary[j], 
                                      maxBoundary[j], i, SizeS);
            __syncthreads();
            if (!(i & 1)) {
                for (int k = 0; k < PSOR_ITERATIONS; ++k) {
                    r = SORHandler(Array, old_val, _LD, _D, _UD, i, SizeS);
                    // This maximum condition guarantees the American exercise-style of the option !
                    r = max(_Payoff, r);
                    Array[i] = r;
                    __syncthreads();
                    __syncthreads();
                }
            }
            else {
                for (int k = 0; k < PSOR_ITERATIONS; ++k) {
                    __syncthreads();
                    r = SORHandler(Array, old_val, _LD, _D, _UD, i, SizeS);
                    // This maximum condition guarantees the American exercise-style of the option !
                    r = max(_Payoff, r);
                    Array[i] = r;
                    __syncthreads();
                }
            }
            NewValues[i] = Array[i];
            NewValues += SizeS;
            _Sigma += SizeS;
        }
    }
}

class BlackScholesPDE {
private:
    int sizeS;		// Dimension of grid (S-axis)
    int sizeT;		// Dimension of grid (T-axis)
    float minT;		// Minimum value of T
    float maxT;		// Maximum value of T
    float minS;		// Minimum value of S
    float maxS;		// Maximum value of S
    float* sigma;		// Local Volatility. 
                                 // Should be given as a row-ordered matrix with 
                                 // Rows = sizeT (1st row <=> t = T and last row <=> t = 0), Cols = sizeS
    float* r;		// Instantaneous Risk-free rate
                                 // Should be given as a vector of size sizeT, with 1st element <=> t = T and last element <=> t = 0
    float* q;	           // Instantaneous Dividend yield
                                 // Should be given as a vector of size sizeT, with 1st element <=> t = T and last element <=> t = 0
    float deltaT;		// Implied from minT, maxT and sizeT
    float deltaS;		// Implied from minS, maxS and sizeS
    float* Grid;		// Stores the PDE grid into the GPU.
    float* cpuGrid;		// Stores the PDE grid into the CPU.
    float* MinBoundary;	// Stores the Dirichlet boundary for S = minS.
    float* MaxBoundary;	// Stores the Dirichlet boundary for S = maxS.
    float* Saxis;		// Stores the values taken by the asset.
    float* Integral_r;	// Stores the integrated risk-free rate.
    float* Integral_q;	// Stores the integrated dividend yield.
    float* Integral_r_CPU;	// Stores the integrated risk-free rate in central memory.
    float* Integral_q_CPU;	// Stores the integrated dividend yield in central memory.
public:
    BlackScholesPDE(
        int _sizeT, int _sizeS,
        float _minT, float _maxT,
        float _minS, float _maxS,
        float* _r, float* _q, float* _sigma);
    ~BlackScholesPDE();
    void CopyToCPU();
    void Print_S(int timeIdx);
    void Print_T(int assetIdx);
    void EuropeanCall(float K);
    void EuropeanPut(float K);
    void AmericanCall(float K);
    void AmericanPut(float K);
    void ImplicitSolving_European();
    void ImplicitSolving_American();
    void ExplicitSolving_European();
    void ExplicitSolving_American();
};

// CTOR.
BlackScholesPDE::BlackScholesPDE(int _sizeT, int _sizeS,
                                 float _minT, float _maxT,
                                 float _minS, float _maxS,
                                 float* _r, float* _q, float* _CPULocalVol) :
    sizeT(_sizeT), sizeS(_sizeS),
    minT(_minT), maxT(_maxT),
    minS(_minS), maxS(_maxS) {
    // deltaT is made negative as the t-axis goes backward.
    deltaT = (minT - maxT) / (sizeT - 1);
    deltaS = (maxS - minS) / (sizeS - 1);
    // Allocate the grid on both GPU and CPU.
    Grid = cudaAlloc(sizeT * sizeS);
    cpuGrid = (float*)malloc(sizeT * sizeS * sizeof(float));
    // Storage for Dirichlet boundaries.
    MinBoundary = cudaAlloc(sizeT);
    MaxBoundary = cudaAlloc(sizeT);
    // Allocate and initialize S-axis.
    Saxis = cudaAlloc(sizeS);
    Saxis_Initialize<<<(sizeS + TPB - 1) / TPB, TPB>>>(Saxis, minS, deltaS, sizeS);
    // Same for local volatility and rates
    sigma = cudaAlloc(sizeS * sizeT);
    cudaMemcpy(sigma, _CPULocalVol, sizeS * sizeT * sizeof(float), cudaMemcpyHostToDevice);
    r = cudaAlloc(sizeT);
    cudaMemcpy(r, _r, sizeT * sizeof(float), cudaMemcpyHostToDevice);
    q = cudaAlloc(sizeT);
    cudaMemcpy(q, _q, sizeT * sizeof(float), cudaMemcpyHostToDevice);
    // r, q are instantanous rates. We have to integrate them to have the "correct" ZC curve.
    Integral_r_CPU = (float*)malloc(sizeT * sizeof(float));
    Integral_q_CPU = (float*)malloc(sizeT * sizeof(float));
    Integral_r = cudaAlloc(sizeT);
    Integral_q = cudaAlloc(sizeT);
    Integral_r_CPU[0] = 0.f;
    Integral_q_CPU[0] = 0.f;
    for (int i = 1; i < sizeT; ++i) {
        Integral_r_CPU[i] = Integral_r_CPU[i - 1] - deltaT * 0.5f * (_r[i] + _r[i - 1]);
        Integral_q_CPU[i] = Integral_q_CPU[i - 1] - deltaT * 0.5f * (_q[i] + _q[i - 1]);
    }
    cudaMemcpy(Integral_q, Integral_q_CPU, sizeT * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(Integral_r, Integral_r_CPU, sizeT * sizeof(float), cudaMemcpyHostToDevice);
}

// DTOR. Basically frees all memory.
BlackScholesPDE::~BlackScholesPDE() {
    cudaFree(Grid);
    cudaFree(MinBoundary);
    cudaFree(MaxBoundary);
    cudaFree(Saxis);
    cudaFree(sigma);
    cudaFree(q);
    cudaFree(r);
    cudaFree(Integral_r);
    cudaFree(Integral_q);
    free(cpuGrid);
    free(Integral_r_CPU);
    free(Integral_q_CPU);
}

// Initializes the PDE Solver with an European Call of strike K.
void BlackScholesPDE::EuropeanCall(float K) {
    // Case i = 0 solved by payoff function
    Call<<<(sizeS + TPB - 1) / TPB, TPB>>>(Grid, Saxis, K, sizeS);
    // Dirichlet boundaries.
    // For European Call, S(t) = minS => Call(t) = 0.
    cudaMemset(MinBoundary, 0, sizeT * sizeof(float));
    // For European Call, S(t) = maxS => Call(T) = maxS * exp(-qT) - K * exp(-rT)
    MaxBoundary_EuropeanCall<<<(sizeT + TPB - 1) / TPB, TPB>>>(MaxBoundary,
                                                               maxS,
                                                               Integral_r,
                                                               Integral_q,
                                                               K,
                                                               sizeT);
}

// Initializes the PDE Solver with an European Put of strike K.
void BlackScholesPDE::EuropeanPut(float K) {
    // Case i = 0 solved by payoff function
    Put<<<(sizeS + TPB - 1) / TPB, TPB>>>(Grid, Saxis, K, sizeS);
    // Dirichlet boundaries.
    // For European Put, S(t) = maxS => Put(t) = 0.
    cudaMemset(MaxBoundary, 0, sizeT * sizeof(float));
    // For European Put, S(t) = minS => Put(T) = - minS * exp(-qT) + K * exp(-rT)
    MinBoundary_EuropeanPut<<<(sizeT + TPB - 1) / TPB, TPB>>>(MinBoundary,
                                                              minS,
                                                              Integral_r,
                                                              Integral_q,
                                                              K,
                                                              sizeT);
}

// Initializes the PDE Solver with an American Call of strike K.
void BlackScholesPDE::AmericanCall(float K) {
    // Case i = 0 solved by payoff function
    Call<<<(sizeS + TPB - 1) / TPB, TPB>>>(Grid, Saxis, K, sizeS);
    // Dirichlet boundaries.
    // For American Call, S(t) = minS => Call(t) = 0.
    cudaMemset(MinBoundary, 0, sizeT * sizeof(float));
    // For American Call, S(t) = maxS => Call(t) = maxS - K
    MaxBoundary_AmericanCall<<<(sizeT + TPB - 1) / TPB, TPB>>>(MaxBoundary,
                                                               maxS,
                                                               K,
                                                               sizeT);
}

// Initializes the PDE Solver with an American Put of strike K.
void BlackScholesPDE::AmericanPut(float K) {
    // Case i = 0 solved by payoff function
    Put<<<(sizeS + TPB - 1) / TPB, TPB>>>(Grid, Saxis, K, sizeS);
    // Dirichlet boundaries.
    // For American Put, S(t) = maxS => Put(t) = 0.
    cudaMemset(MaxBoundary, 0, sizeT * sizeof(float));
    // For American Put, S(t) = minS => Put(t) = - minS + K
    MinBoundary_AmericanPut<<<(sizeT + TPB - 1) / TPB, TPB>>>(MinBoundary,
                                                              minS,
                                                              K,
                                                              sizeT);
}

// Copy the grid back to central memory.
void BlackScholesPDE::CopyToCPU() {
    // Copy the grid back to Central memory
    cudaMemcpy(cpuGrid, Grid, sizeS * sizeT * sizeof(float), cudaMemcpyDeviceToHost);
}

// Prints the derivative value at a fixed time specified by timeIdx.
void BlackScholesPDE::Print_S(int timeIdx) {
    for (int i = 0; i < sizeS; ++i) {
        // The timeIdx goes backward, ie. timeIdx = 0 at maturity, and = (sizeT - 1) at option issue.
        std::cout << minS + i * deltaS << "\t" << cpuGrid[timeIdx * sizeS + i] << "\n";
    }
}

// Prints the derivative value at a fixed asset level specified by assetIdx.
void BlackScholesPDE::Print_T(int assetIdx) {
    for (int i = 0; i < sizeT; ++i) {
        std::cout << minT + i * deltaT << "\t" << cpuGrid[i * sizeS + assetIdx] << "\n";
    }
}

void BlackScholesPDE::ExplicitSolving_European() {
    // 1 block is a limitation to synchronizing all the threads at each time step.
    ExplicitKernel_European<<<1, sizeS>>>(Grid,
                                          Saxis,
                                          r, q, sigma, deltaT, deltaS,
                                          MinBoundary, MaxBoundary, sizeS, sizeT);
}

void BlackScholesPDE::ExplicitSolving_American() {
    // 1 block is a limitation to synchronizing all the threads at each time step.
    ExplicitKernel_American<<<1, sizeS>>>(Grid,
                                          Saxis,
                                          r, q, sigma, deltaT, deltaS,
                                          MinBoundary, MaxBoundary, sizeS, sizeT);
}

void BlackScholesPDE::ImplicitSolving_European() {
    // 1 block is a limitation to synchronizing all the threads at each time step.
    ImplicitKernel_European<<<1, sizeS, sizeS * sizeof(float)>>>(Grid + sizeS, 
                                                                 r, q, sigma, Saxis,
                                                                 MinBoundary, MaxBoundary, 
                                                                 deltaT, deltaS, sizeS, sizeT);
}

void BlackScholesPDE::ImplicitSolving_American() {
    // 1 block is a limitation to synchronizing all the threads at each time step.
    ImplicitKernel_American<<<1, sizeS, sizeS * sizeof(float)>>>(Grid + sizeS, 
                                                                 r, q, sigma, Saxis,
                                                                 MinBoundary, MaxBoundary, 
                                                                 deltaT, deltaS, sizeS, sizeT);
}

// Main function.
int main() {
    // PDE Parameters.
    int sizeT = 20000;
    int sizeS = 5 * 190 + 1;
    float minT = 0.0f;
    float maxT = 1.0f;
    // Should be in the range [max(0, K-6*Sigma*sqrt(T)); K+6*Sigma*sqrt(T)]
    float minS = 0.0f;
    float maxS = 190.0f;

    // Market observed parameters.
    float* sigma = (float*)malloc(sizeT * sizeS * sizeof(float));
    float* r = (float*)malloc(sizeT * sizeof(float));
    float* q = (float*)malloc(sizeT * sizeof(float));
    for (int i = 0; i < sizeT; ++i) {
        r[i] = 0.064f;
        q[i] = 0.045f;
    }
    for (int i = 0; i < sizeT * sizeS; ++i) {
        sigma[i] = 0.15f;
    }

    // Solving part.
    BlackScholesPDE Solver(
        sizeT, sizeS,
        minT, maxT,
        minS, maxS,
        r, q, sigma);
    float Strike = 100.f;
    Solver.EuropeanPut(Strike);
    //Solver.AmericanPut(Strike);
    Bench(50, "ImplicitSolving_European", [&]() {
        //Solver.ExplicitSolving_European();
        //Solver.ExplicitSolving_American();
        Solver.ImplicitSolving_European();
        //Solver.ImplicitSolving_American();
        cudaDeviceSynchronize();
    });
    Solver.CopyToCPU();
    
    // Print at t = 0.
    Solver.Print_S(sizeT - 1);
    getchar();
    free(sigma);
    free(r);
    free(q);
    return 0;
}
