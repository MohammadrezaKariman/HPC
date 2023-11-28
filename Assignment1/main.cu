
#include <iostream>
#include <sstream>
#include <cmath>

#include <cuda_runtime.h>
#include <cublas_v2.h>

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/reduce.h>

#include <chrono>
#include <random>

#include <cblas.h>




//-----------------------------------------------------------------------------
// very useful debug macros!
#define FatalError(s) do {                                             \
    std::stringstream _where, _message;                                \
    _where << __FILE__ << ':' << __LINE__;                             \
    _message << std::string(s) + "\n" << __FILE__ << ':' << __LINE__;  \
    std::cerr << _message.str() << "\nAborting...\n";                  \
    cudaDeviceReset();                                                 \
    exit(1);                                                           \
} while(0)

#define checkCudaErrors(status) do {                                   \
    std::stringstream _error;                                          \
    if (status != 0) {                                                 \
      _error << "Cuda failure: " << status;                            \
      FatalError(_error.str());                                        \
    }                                                                  \
} while(0)

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess)
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}













void function_1 (double *a, double *b, const int N) {
  
  for (int i=0; i < N; i++) {
    b[i] = a[i];
    b[i]++;

  }
}

void function_2 (double *a, double *b, double *c, const int N) {
  
  for (int i=0; i < N; i++) {
    c[i] = a[i] + b[i];
}
}

void function_3 (double *a, double *b, const int N) {

  for (int i=0; i < N; i++){

      b[i] = log(a[i]);

  }
}

void function_4 (double *c, double *b, const int N) {

  for (int i=0; i < N; i++){

      c[i] = exp (b[i]);

  }
}

//-----------------------------------------------------------------------------
// cuda
//__device__

__global__ void function_1_gpu (double *a, double *b, const int N) 
{
  const int i = blockIdx.x * blockDim.x + threadIdx.x; 

  if (i < N)
  {
    b[i] = a[i];
    b[i]++ ;
  }
  
}
  
__global__ void function_2_gpu (double *a, double *b, double *c, const int N) 
{
  const int i = blockIdx.x * blockDim.x + threadIdx.x; 

  if (i < N)
  {
    c[i] = a[i] + b[i];
  }
    
}

__global__ void function_3_gpu (double *a, double *b, const int N) 
{
  const int i = blockIdx.x * blockDim.x + threadIdx.x; 

  if (i < N)
  {

      b[i] = log(a[i]);

  }
}



__global__ void function_4_gpu(double *c, double *b, const int N)
{
  const int i = blockIdx.x*blockDim.x + threadIdx.x;

  if (i < N)
    c[i] = exp(b[i]);
}


//-----------------------------------------------------------------------------
// evaluation functions

void res_check (double *res, double *ref, const int N)
{
  for (int i = 0; i < N; i++)
  {
    assert(abs(res[i] - ref[i]) < 1e-6);
  }

  std::cout <<"results on gpu = results on cpu" << std::endl;
}

void task_2()
{
  // initialize data on cpu
  const int N = 12800000;
  //double a[N]; // int*  -> 1
  //double b[N]; // int*  -> 2
  //double c[N]; // int*  -> 0
  double *a_cpu = new double[N];
  double *b_cpu = new double[N];
  double *c_cpu = new double[N];

  double *a_res = new double[N];
  double *b_res = new double[N];
  double *c_res = new double[N];

  // Initialize cpu data 

  for (int i=0; i < N; i++){
    a_cpu[i] = 3;
    b_cpu[i] = 2;
    c_cpu[i] = 0;
  }

  // initialize gpu data
  double *a_gpu, *b_gpu, *c_gpu, *a_uni, *b_uni, *c_uni;
  checkCudaErrors(cudaMalloc((void**) &a_gpu, N*sizeof(double)));
  checkCudaErrors(cudaMalloc((void**) &b_gpu, N*sizeof(double)));
  checkCudaErrors(cudaMalloc((void**) &c_gpu, N*sizeof(double)));

  checkCudaErrors(cudaMallocManaged(&a_uni, N*sizeof(double)));
  checkCudaErrors(cudaMallocManaged(&b_uni, N*sizeof(double)));
  checkCudaErrors(cudaMallocManaged(&c_uni, N*sizeof(double)));

  for (int i=0; i < N; i++) {
    a_uni[i] = a_cpu[i];
    b_uni[i] = b_cpu[i];
    c_uni[i] = c_cpu[i];
  }

  // copy cpu data to gpu data
  checkCudaErrors(cudaMemcpy(a_gpu, a_cpu, N*sizeof(double), cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(b_gpu, b_cpu, N*sizeof(double), cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(c_gpu, c_cpu, N*sizeof(double), cudaMemcpyHostToDevice));

  std::chrono::system_clock::time_point begin = std::chrono::system_clock::now();

  function_1 (a_cpu, b_cpu, N);
  function_2 (a_cpu, b_cpu, c_cpu, N);
  function_3 (a_cpu, b_cpu, N);
  function_4 (c_cpu, b_cpu, N); 

  std::chrono::system_clock::time_point end = std::chrono::system_clock::now();

  std::cout << "Computational Time on CPU = " << std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count() << "[ms]" << std::endl;


  // compute dimensions
  const int block_size = 64;
  const int number_of_blocks = (N + block_size - 1)/block_size;

  dim3 block_dim(block_size);
  dim3 grid_dim(number_of_blocks);

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  cudaEventRecord(start);

  function_1_gpu<<<grid_dim, block_dim>>>(a_gpu, b_gpu, N);
  function_2_gpu<<<grid_dim, block_dim>>>(a_gpu, b_gpu, c_gpu, N);
  function_3_gpu<<<grid_dim, block_dim>>>(a_gpu, b_gpu, N);
  function_4_gpu<<<grid_dim, block_dim>>>(c_gpu, b_gpu, N);

  cudaEventRecord(stop);


  cudaEventSynchronize(stop);
  float milliseconds = 0;
  cudaEventElapsedTime(&milliseconds, start, stop);

  std::cout << milliseconds << " [ms]"<< std::endl;

  //function_1_gpu<<<grid_dim, block_dim>>>(a_uni, b_uni, N);

  //gpuErrchk(cudaPeekAtLastError());
  checkCudaErrors(cudaDeviceSynchronize());

  checkCudaErrors(cudaMemcpy(a_res, a_gpu, N*sizeof(double), cudaMemcpyDeviceToHost));
  checkCudaErrors(cudaMemcpy(b_res, b_gpu, N*sizeof(double), cudaMemcpyDeviceToHost));
  checkCudaErrors(cudaMemcpy(c_res, c_gpu, N*sizeof(double), cudaMemcpyDeviceToHost));

  checkCudaErrors(cudaFree(a_gpu));
  checkCudaErrors(cudaFree(b_gpu));
  checkCudaErrors(cudaFree(c_gpu));


  // compare results
  res_check (b_cpu, b_res, N);
  res_check (a_cpu, a_res, N);
  res_check (c_cpu, c_res, N);


  cudaEventRecord(start);

  function_1_gpu<<<grid_dim, block_dim>>>(a_uni, b_uni, N);
  function_2_gpu<<<grid_dim, block_dim>>>(a_uni, b_uni, c_uni, N);
  function_3_gpu<<<grid_dim, block_dim>>>(a_uni, b_uni, N);
  function_4_gpu<<<grid_dim, block_dim>>>(c_uni, b_uni, N);

  cudaEventRecord(stop);


  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&milliseconds, start, stop);

  std::cout << milliseconds << " [ms]"<< std::endl;

  checkCudaErrors(cudaDeviceSynchronize());


    // compare results
  res_check (b_cpu, b_uni, N);
  res_check (a_cpu, a_uni, N);
  res_check (c_cpu, c_uni, N);

  // Deleting allocated memory on gpu

  checkCudaErrors(cudaFree(a_uni));
  checkCudaErrors(cudaFree(b_uni));
  checkCudaErrors(cudaFree(c_uni));







  







  /*std::cout<<"a contains: \n";

  for (int i=0; i < N; i++){
    std::cout<<a[i]<<" ";
  }

  std::cout<<"\n";

    std::cout<<"b contains: \n";

  for (int i=0; i < N; i++){
    std::cout<<b[i]<<" ";
  }

  std::cout<<"\n";
  

 std::cout << " C contains: \n";

  for (int i = 0; i < N; i++){
    std::cout << c[i] << " ";

  }

 std::cout << "\n";
 */

// Deleting Allocated memory on cpu

  delete a_cpu;
  delete b_cpu;
  delete c_cpu;

  delete a_res;
  delete b_res;
  delete c_res;
}












struct thrust_function1
{
  // thrust::transform(a_gpu.begin(), a_gpu.end(), b_gpu.begin(), thrust::negate<int>());
  // b[i] = operator(a[i]);
  __host__ __device__ double operator()(double &a_i) const
  {
    // b = a
    // b = b + 1
    return a_i + 1.0;
  }
};


struct thrust_function2
{
  __host__ __device__ double operator()(double &a_i, double &b_i) const
  {
    return a_i + b_i ;
  }
};

struct thrust_function3
{
  __host__ __device__ double operator()(double &a_i) const
  {
    return log(a_i) ;
  }
};


struct thrust_function4
{
  __host__ __device__ double operator()(double &b_i) const
  {
    return exp(b_i) ;
  }
};





bool thrust_res_check (thrust::host_vector<double> &a_cpu, thrust::device_vector<double> &a_gpu, const int N)
{
  
for (int i = 0; i < N; i++)
  {
    assert(abs(a_gpu[i] - a_cpu[i]) < 1e-9);
  }

  std::cout <<"thrust check: result = reference" << std::endl;

  return true;
}

void task_3 ()
{
  int N = 100; 
  // double *a_cpu = new double[N]; // c++ array
  // std::vector<double> a_cpu(N);  // stl class
  // *a_cpu -> first element
  // a_cpu[0] -> first element
  // a_cpu[i] -> i-th element
  thrust::host_vector<double> a_cpu(N); // thrust class
  thrust::host_vector<double> b_cpu(N);
  thrust::host_vector<double> c_cpu(N);

  // allocate unified memory and convert to thrust
  double * a_uni;
  double * b_uni;
  double * c_uni; 

  checkCudaErrors(cudaMallocManaged(&a_uni, N*sizeof(double)));
  thrust::device_ptr<double> t_a_uni(a_uni);

  checkCudaErrors(cudaMallocManaged(&b_uni, N*sizeof(double)));
  thrust::device_ptr<double> t_b_uni(b_uni);

  checkCudaErrors(cudaMallocManaged(&c_uni, N* sizeof(double)));
  thrust::device_ptr<double> t_c_uni(c_uni);
  
  // initialize everthing
  for (int i=0; i < N; i++){
    a_cpu[i] = 3;
    b_cpu[i] = 2;
    c_cpu[i] = 0;

    t_a_uni[i] = 3;
    t_b_uni[i] = 2;
    t_c_uni[i] = 0;
  }

  // thrust::copy(a_cpu.begin(), a_cpu.end(), t_a_uni);

  thrust::device_vector<double> a_gpu = a_cpu;
  thrust::device_vector<double> b_gpu = b_cpu;
  thrust::device_vector<double> c_gpu = c_cpu;

  // computations cpu (ref)
  thrust::transform(a_cpu.begin(), a_cpu.end(), b_cpu.begin(), thrust_function1());
  thrust::transform(a_cpu.begin(), a_cpu.end(), b_cpu.begin(), c_cpu.begin(), thrust_function2());
  thrust::transform(a_cpu.begin(), a_cpu.end(), b_cpu.begin(), thrust_function3());
  thrust::transform(c_cpu.begin(), c_cpu.end(), b_cpu.begin(), thrust_function4());

  // computations gpu (res)
  thrust::transform(a_gpu.begin(), a_gpu.end(), b_gpu.begin(), thrust_function1());
  thrust::transform(a_gpu.begin(), a_gpu.end(), b_gpu.begin(), c_gpu.begin(), thrust_function2());
  thrust::transform(a_gpu.begin(), a_gpu.end(), b_gpu.begin(), thrust_function3());
  thrust::transform(c_gpu.begin(), c_gpu.end(), b_gpu.begin(), thrust_function4());

  // computations gpu (res) unified
  thrust::transform(t_a_uni, t_a_uni + N, t_b_uni, thrust_function1());
  thrust::transform(t_a_uni, t_a_uni + N, t_b_uni, t_c_uni, thrust_function2());
  thrust::transform(t_a_uni, t_a_uni + N, t_b_uni, thrust_function3());
  thrust::transform(t_c_uni, t_c_uni + N, t_b_uni, thrust_function4());


  std::cout << "Checks for gpu memory" << std::endl;
  thrust_res_check(a_cpu, a_gpu, N);
  thrust_res_check(b_cpu, b_gpu, N);
  thrust_res_check(c_cpu, c_gpu, N);

  std::cout << "Checks for unified memory" << std::endl;

  auto t_a_uni_res = thrust::device_vector<double>(t_a_uni, t_a_uni+N);
  auto t_b_uni_res = thrust::device_vector<double>(t_b_uni, t_b_uni+N);
  auto t_c_uni_res = thrust::device_vector<double>(t_c_uni, t_c_uni+N);



  thrust_res_check(a_cpu, t_a_uni_res, N);
  thrust_res_check(b_cpu, t_b_uni_res, N);
  thrust_res_check(c_cpu, t_c_uni_res, N);
  
  // std::cout << "b_cpu " << b_cpu[0] << std::endl;
  // std::cout << "b_gpu " << b_gpu[0] << std::endl;


}



void task_4()
{
  double lower_bound = 0;
  double upper_bound = 10000;
  std::uniform_real_distribution<double> unif(lower_bound,upper_bound);
  std::default_random_engine re;
  double a_random_double = unif(re);

  cublasHandle_t handle;
  checkCudaErrors(cublasCreate(&handle));

  // (i)

  const int N = 10; 

  double a = unif(re), b = unif(re), one = 1.0, gpu_norm = 0, gpu_dot = 0;

  

  double *x_cpu = new double [N];
  double *y_cpu = new double [N];
  double *z_cpu = new double [N];

  
  double *x_res = new double[N];
  double *y_res = new double[N];
  double *z_res = new double[N];



  for (int i =0; i < N; i++)
  {
    
    x_cpu[i] = unif(re);
    y_cpu[i] = unif(re);
    z_cpu[i] = 0;

    std::cout << x_cpu[i] << " ";

  }
  std::cout << std::endl;





  double *x_gpu;   double *y_gpu; double *z_gpu;
  checkCudaErrors(cudaMalloc((void**) &x_gpu, N*sizeof(double))); 
  checkCudaErrors(cudaMalloc((void**) &y_gpu, N*sizeof(double))); 
  checkCudaErrors(cudaMalloc((void**) &z_gpu, N*sizeof(double))); 
  checkCudaErrors(cudaMemcpy(x_gpu, x_cpu, N*sizeof(double), cudaMemcpyHostToDevice)); 
  checkCudaErrors(cudaMemcpy(y_gpu, y_cpu, N*sizeof(double), cudaMemcpyHostToDevice)); 
  checkCudaErrors(cudaMemcpy(z_gpu, z_cpu, N*sizeof(double), cudaMemcpyHostToDevice)); 





  // i [y:= ax + y]

  cblas_daxpy(N, a, x_cpu, 1, y_cpu, 1);
  checkCudaErrors(cublasDaxpy(handle, N, &a, x_gpu, 1, y_gpu, 1));
  checkCudaErrors(cudaMemcpy(y_res, y_gpu, N*sizeof(double), cudaMemcpyDeviceToHost));
  std::cout << "Check Point: Task 4_i_1 -> ";// << std::endl;
  res_check(y_cpu, y_res, N);


  // i [x:= ax + y]

  cblas_dscal(N, a, x_cpu, 1);
  cblas_daxpy(N, one, y_cpu, 1, x_cpu, 1);
  checkCudaErrors(cublasDscal(handle, N, &a, x_gpu, 1));
  checkCudaErrors(cublasDaxpy(handle, N, &one, y_gpu, 1, x_gpu, 1));
  checkCudaErrors(cudaMemcpy(x_res, x_gpu, N*sizeof(double), cudaMemcpyDeviceToHost));
  std::cout << "Check Point: Task 4_i_2 -> ";// << std::endl;
  res_check(x_cpu, x_res, N);


  // i [z:= ax + By]

  cblas_daxpy(N, a, x_cpu, 1, z_cpu, 1);
  cblas_daxpy(N, b, y_cpu, 1, z_cpu, 1);
  checkCudaErrors(cublasDaxpy(handle, N, &a, x_gpu, 1, z_gpu, 1));
  checkCudaErrors(cublasDaxpy(handle, N, &b, y_gpu, 1, z_gpu, 1));
  checkCudaErrors(cudaMemcpy(z_res, z_gpu, N * sizeof(double), cudaMemcpyDeviceToHost));
  std::cout << "Check Point: Task 4_i_3 -> "; 
  res_check(z_cpu, z_res, N);

  
  // i [<x,y>]
  double cpu_dot = cblas_ddot(N, x_cpu, 1, y_cpu, 1);
  checkCudaErrors(cublasDdot(handle, N, x_gpu, 1, y_gpu, 1, &gpu_dot));
  std::cout << "Check Point: Task 4_i_4 -> ";
  std::cout << cpu_dot << "-" << gpu_dot << "\n";
  // res_check(&cpu_dot, &gpu_dot, 1);


  // i [|x|]

  double cpu_norm = cblas_dnrm2(N, x_cpu, 1);
  checkCudaErrors(cublasDnrm2(handle, N, x_gpu, 1, &gpu_norm));
  //checkCudaErrors(cudaMemcpy(x_res, x_gpu, N * sizeof(double), cudaMemcpyDeviceToHost));
  std::cout << "Check Point: Task 4_i_5 -> ";
  res_check(&cpu_norm, &gpu_norm, 1);


  // ii [r = M * x ]




  
  checkCudaErrors(cudaFree(x_gpu));
  checkCudaErrors(cudaFree(y_gpu));
  checkCudaErrors(cudaFree(z_gpu));

  delete x_cpu; 
  delete y_cpu; 
  delete z_cpu; 

  delete x_res;
  delete y_res;
  delete z_res;



}






















//-----------------------------------------------------------------------------
// cuda kernel example
__global__ void gpu_add_scalar(const int N, double *data_ptr, const double value)
{
  const int idx = blockIdx.x*blockDim.x + threadIdx.x;

  if (idx < N)
    data_ptr[idx] += value;
}

//-----------------------------------------------------------------------------
// main function
int main()
{
  //task_2();
  // task_3();
  task_4();

  return 0;
}
//-----------------------------------------------------------------------------