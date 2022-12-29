#include <stdio.h>
#include <cooperative_groups.h>
namespace cg = cooperative_groups;

#include <helper_cuda.h>
#include <stdlib.h>
#include <iostream>
#include <vector>
#include <string>
#include <time.h>
#include <algorithm>

using namespace std;
# define REPEAT_QUERY 2
#define MAX_SEQS_SIZE 512
#define MAX_SEQS_NUM 8

__device__ bool smallerThan(int *pointA, int *pointB, int size) //<=
{
  bool flag = true;
  for (int i = 0; i < size; i++)
  {
    if (pointA[i] > pointB[i])
    {
      flag = false;
      break;
    }
  }
  return flag;
}

__device__ bool equal(int *pointA, int *pointB, int size) //=
{
  bool flag = true;
  for (int i = 0; i < size; i++)
  {
    if (pointA[i] != pointB[i])
    {
      flag = false;
      break;
    }
  }
  return flag;
}

#define MAX_DK_SIZE 40000
#define MAX_ALPHABET_SET_SIZE 256 // ascii

__device__ int AlphaNumber[MAX_ALPHABET_SET_SIZE]; // cast character in aplhabet set to index
__device__ unsigned int AlphaNumber_size = 0;
#define MAX_PAR_SET_SIZE 80000
__device__ int Par[MAX_PAR_SET_SIZE][MAX_SEQS_NUM];
__device__ unsigned int Par_size = 0;

__device__ int Par2[MAX_PAR_SET_SIZE][MAX_SEQS_NUM];
__device__ unsigned int Par2_size = 0;

// preProcess Kernel function
static __global__ void MLCS0(char **seqs_d, int seqs_num, char *alphabet_set,
                             int *Dk0, unsigned int *Dk_size) // process 0
{
  // printf("(%d,%d),%d,%d\n", threadIdx.x, blockIdx.x, blockDim.x, gridDim.x);
  if (blockIdx.x * blockDim.x + threadIdx.x == 0)
  {
    // printf("\n---%d",1);
    for (char *a = alphabet_set; *a != '\0'; a++)
    {
      AlphaNumber[*a] = AlphaNumber_size;
      AlphaNumber_size++;
    }
  }
}

__device__ volatile int blockCounter = 0;

// Kernel Function Part 1
static __global__ void MLCS1(char **seqs_d, int seqs_num, char *alphabet_set,
                             int *Dk0, unsigned int *Dk_size, int *error = 0)
{
#define MAX_PAR_SHARED_SIZE 1534
  __shared__ int Par_shared[MAX_PAR_SHARED_SIZE][MAX_SEQS_NUM];
  // NOTICE :for each point ,every alpha 's' has at most one Par_s
  __shared__ unsigned int Par_shared_size;
  __shared__ unsigned int Par_shared_Written;

  if (threadIdx.x == 0)
  {
    Par_shared_size = 0;
    Par_shared_Written=0;

  }
  __syncthreads();

  int pointQ[MAX_SEQS_NUM];
  // printf("A<%d>\n",threadIdx.x+blockDim.x*blockIdx.x);
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < *Dk_size; i += blockDim.x * gridDim.x)
  {
    // move Dk from __device__ to this thread
    memcpy(pointQ, Dk0 + i * MAX_SEQS_NUM, MAX_SEQS_NUM * sizeof(int));
    // printf("----%d", 2);
    int pointB[MAX_SEQS_NUM];
    // B'=(Par(q,xigema))
    // printf("\n%d+%d+%d+%d", pointQ[0], pointQ[1], pointQ[2], pointQ[3]);
    for (char *c = alphabet_set; *c != '\0'; c++)
    {
      // printf("\n<<<<<<<cccc:%c>>>>>>>",*c);
      bool flag = true;
      for (int j = 0; j < seqs_num; j++)
      {
        int index = pointQ[j] + 1;
        for (; seqs_d[j][index] != '\0'; index++)
        {
          if (seqs_d[j][index] == *c)
          {
            // printf("Test::%c-%d\n",*c,j);
            pointB[j] = index;
            break;
          }
        }
        if (seqs_d[j][index] == '\0')
        {
          flag = false;
          break;
        }
      }
      if (flag)
      {
        //  once get a pointB from set B' do:
        int repeatQuery=REPEAT_QUERY;
        unsigned int index0=0;
        while(repeatQuery&&flag){
          unsigned int index1 =atomicCAS(&Par_shared_Written,0,0);
          for(int i = index0; i <index1;i++){
            if(equal(Par_shared[i], pointB,seqs_num)){
              flag=false;
          // printf("release %d\n",i);
              break;
            }
          }
          index0=index1;
          repeatQuery--;
        }
      }
      if(flag){
        unsigned int index = atomicAdd(&Par_shared_size, 1);
          if (index >= MAX_PAR_SHARED_SIZE)
          {
            printf("\nERROR:>> Par_shared_size not enough:%d>%d\n", index, MAX_PAR_SHARED_SIZE);
            atomicAdd(error, 1);
            __threadfence();
          }
          memcpy(Par_shared[index], pointB, MAX_SEQS_NUM*sizeof(int));
          atomicAdd(&Par_shared_Written,1);
          // printf("\n[%d-%d-%d-%d] hasParent-> [%d-%d-%d-%d]", pointQ[0], pointQ[1], pointQ[2], pointQ[3],
          //        pointB[0], pointB[1], pointB[2], pointB[3]);
      }
    }
  }
    // printf("----%d",3);
  __syncthreads();
  // skip the repeat item in set Par_shared
  for (int i = threadIdx.x; i < Par_shared_size; i += blockDim.x)
  {
    // printf("\nPar_s%d/%d=[%d-%d-%d-%d]", i, Par_shared_size, Par_shared[i][0], Par_shared[i][1], Par_shared[i][2], Par_shared[i][3]);
    bool repeat = false;
    for (int j = 0; j < i; j++)
    {
      if (equal(Par_shared[j], Par_shared[i], seqs_num))
      {
        repeat = true;
        // printf("\n==[%d-%d-%d-%d]%d=%d", Par_shared[j][0], Par_shared[j][1], Par_shared[j][2], Par_shared[j][3], i, j);
        break;
      }
    }
    if (!repeat)
    {
      unsigned int index = atomicAdd(&Par_size, 1);
      if (index >= MAX_PAR_SET_SIZE)
      {
        printf("\nERROR:>> Par_size not enough:%d>%d\n", index, MAX_PAR_SET_SIZE);
        atomicAdd(error, 1);
        __threadfence();
      }
      // printf("\ncopyToPar[%d]=%d[%d-%d-%d-%d-%d]%d", index, i, Par_shared[i][0], Par_shared[i][1], Par_shared[i][2], Par_shared[i][3], Par_shared[i][4],seqs_num);
      memcpy(Par[index], Par_shared[i], MAX_SEQS_NUM*sizeof(int));
    }
  }
  __syncthreads();
  if (threadIdx.x == 0)
  {
    atomicAdd((int *)&blockCounter, 1);
    // printf("\n*#(%d,%d)", blockIdx.x, threadIdx.x);
    __threadfence();
  }
  if (blockIdx.x == 0)
  {
    if (threadIdx.x == 0)
    {
      while (blockCounter < gridDim.x)
      {
        // printf("#");
      };
    }
    __syncthreads();
    // skip the repeat item in set Par
    for (int i = threadIdx.x; i < Par_size; i += blockDim.x)
    {
      // printf("\nPar%d[%d-%d-%d-%d]",i,Par[i][0],Par[i][1],Par[i][2],Par[i][3]);
      bool repeat = false;
      for (int j = 0; j < i; j++)
      {
        if (equal(Par[j], Par[i], seqs_num))
        {
          repeat = true;
          // printf("\n=[%d-%d-%d-%d]",Par[j][0],Par[j][1],Par[j][2],Par[j][3]);
          break;
        }
      }
      if (!repeat)
      {
        unsigned int index = atomicAdd(&Par2_size, 1);
        if (index >= MAX_PAR_SET_SIZE)
        {
          printf("\nERROR:>> Par_size(2) not enough:%d>%d\n", index, MAX_PAR_SET_SIZE);
          atomicAdd(error, 1);
          __threadfence();
        }
        memcpy(Par2[index], Par[i], MAX_SEQS_NUM*sizeof(int));
        // printf("\n<%d>=[%d-%d-%d-%d]",index,Par2[index][0],Par2[index][1],Par2[index][2],Par2[index][3]);
      }
    }
    __syncthreads();
  }
  if (blockIdx.x * blockDim.x + threadIdx.x == 0)
  {
    *Dk_size = 0;
    blockCounter = 0;
    __threadfence();
    // printf("----%d", 4);
  }
}

// Kernel Function Part 2
static __global__ void MLCS2(char **seqs_d, int seqs_num, char *alphabet_set,
                             int *Dk0, unsigned int *Dk_size, int *error)
{
  //...
  // printf("\n<<<<<%d--->>>>>",Par2_size);
  // printf("B<%d,%d>\n", threadIdx.x + blockDim.x * blockIdx.x, Par2_size);
  for (int i = threadIdx.x + blockDim.x * blockIdx.x; i < Par2_size; i += blockDim.x * gridDim.x)
  {
    // printf("<%d---%d>\n",threadIdx.x+blockDim.x*blockIdx.x,i);
    bool isMinima = true;
    for (int j = 0; j < Par2_size; j++)
    {
      if (i != j && smallerThan(Par2[j], Par2[i], seqs_num))
      {
        isMinima = false;
        // printf("\n<<%d<%d:[%d-%d-%d-%d]<=[%d-%d-%d-%d]",j,i,Par2[j][0],Par2[j][1],Par2[j][2],Par2[j][3],
        // Par2[i][0],Par2[i][1],Par2[i][2],Par2[i][3]);
        break;
      }
    }

    if (isMinima)
    {
      // printf("\nsend %d to Dk+1 [%d-%d-%d-%d-%d]",i,Par2[i][0],Par2[i][1],Par2[i][2],Par2[i][3],Par2[i][4]);
      int index = atomicAdd(Dk_size, 1);
      if (index >= MAX_DK_SIZE)
      {
        printf("\nERROR:>> Dk_size not enough:%d>%d\n", index, MAX_DK_SIZE);
        atomicAdd(error, 1);
        __threadfence();
      }
      memcpy(Dk0 + index * MAX_SEQS_NUM, Par2[i], seqs_num * sizeof(int));
    }
    // printf("\n<<<<<<<%d>>>>>>>%d",555,i);
  }
  // printf("C<%d>\n", threadIdx.x + blockDim.x * blockIdx.x);
  __syncthreads();
  if (threadIdx.x == 0)
  {
    // printf("\n*$(%d,%d)", blockIdx.x, threadIdx.x);
    atomicAdd((int *)&blockCounter, 1);
    __threadfence();
  }
  if (0 == blockIdx.x * blockDim.x + threadIdx.x)
  {
    // printf("sdfsdfAAA");
    while (blockCounter < gridDim.x)
    {
      // printf("$");
    };
    Par_size = 0;
    Par2_size = 0;
    blockCounter = 0;
    __threadfence();
  }
}

void MLCS(char **seqs_d, int num_seqs, char *alphabet_set,
          int *Dk0, unsigned int *Dk_size, int blockDimx, int threadDimx, int *error)
{

  MLCS1<<<blockDimx, threadDimx>>>(seqs_d, num_seqs, alphabet_set, Dk0, Dk_size, error);
  checkCudaErrors(cudaDeviceSynchronize());
  MLCS2<<<blockDimx, threadDimx>>>(seqs_d, num_seqs, alphabet_set, Dk0, Dk_size, error);
  checkCudaErrors(cudaDeviceSynchronize());
}

struct Point
{
  int *p;
  int len;
  Point(int size, int *q)
  {
    len = size;
    p = new int[size];
    for (int i = 0; i < size; i++)
    {
      p[i] = q[i];
    }
  }

  bool isSmallerThan(Point q)
  {
    bool res = true;
    for (int i = 0; i < len; i++)
    {
      if (p[i] >= q.p[i])
      {
        res = false;
        break;
      }
    }
    return res;
  }

  string toString(){
    string res = "[";
    for(int i = 0; i < len-1; i++){
      res=res+to_string(p[i])+",";
    }
    res=res+to_string(p[len-1])+"]";
    return res;
  }
};

int main(int argc, char **argv)
{
  printf("MCLS Starting...\n\n");
  if (argc < 2)
  {
    cout << "Usage: \"./xxx input.txt\" or \"./xxx input.txt gridDimx blockDimx \" or \"./xxx input.txt gridDimx blockDimx  anything(means show Dk)\"" << endl;
    exit(EXIT_SUCCESS);
  }
  // read input from 2.txt

  std::ifstream file;
  file.open(argv[1]);
  if (!file.is_open())
  {
    printf("open file error\n");
  }
  // set the alphabet_set : also can got form the input strings
  std::string alphabet_set = "ACTG";
  char *alphabet_set_d;
  checkCudaErrors(cudaMalloc(&alphabet_set_d, sizeof(char) * (alphabet_set.length() + 1)));
  checkCudaErrors(cudaMemcpy(alphabet_set_d, alphabet_set.c_str(), sizeof(char) * (alphabet_set.length() + 1), cudaMemcpyHostToDevice));

  int num_seqs = 0;
  char *seqs[MAX_SEQS_NUM];
  vector<string> sequences;
  std::string s;
  while (getline(file, s))
  {
    sequences.push_back(s);
    printf("%d:%s\n", num_seqs, s.c_str());
    if (s.length() >= MAX_SEQS_SIZE)
    {
      printf("sequences too long : %d >= %d\n", s.length(), MAX_SEQS_SIZE);
      exit(EXIT_SUCCESS);
    }
    
    if (num_seqs+1 > MAX_SEQS_NUM)
    {
      printf("sequences too much : %d >= %d\n", num_seqs, MAX_SEQS_NUM);
      exit(EXIT_SUCCESS);
    }
    checkCudaErrors(cudaMalloc((void **)&seqs[num_seqs], (MAX_SEQS_SIZE) * sizeof(char)));
    checkCudaErrors(cudaMemcpy(seqs[num_seqs], s.c_str(), (MAX_SEQS_SIZE) * sizeof(char), cudaMemcpyHostToDevice));
    num_seqs++;
    
  }


  char **seqs_d;
  checkCudaErrors(cudaMalloc(&seqs_d, sizeof(char *) * MAX_SEQS_NUM));
  checkCudaErrors(cudaMemcpy(seqs_d, seqs, sizeof(char *) * MAX_SEQS_NUM, cudaMemcpyHostToDevice));

  int Dk[MAX_DK_SIZE][MAX_SEQS_NUM];
  vector<vector<Point>> D;
  unsigned int Dk_size = 0;

  for (int i = 0; i < num_seqs; i++)
  { // Dk[0]=[-1,-1,-1,...]
    Dk[0][i] = -1;
  }
  Dk_size++;
  vector<Point> temp;
  temp.push_back(Point(num_seqs, Dk[0]));
  D.push_back(temp);

  int k = 0;
  int *Dk_d;
  unsigned int *Dk_size_d;
  checkCudaErrors(cudaMalloc(&Dk_d, sizeof(int) * MAX_SEQS_NUM * MAX_DK_SIZE));
  checkCudaErrors(cudaMalloc(&Dk_size_d, sizeof(int)));

  checkCudaErrors(cudaMemcpy(Dk_d, *Dk, sizeof(int) * MAX_SEQS_NUM * MAX_DK_SIZE, cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(Dk_size_d, &Dk_size, sizeof(int), cudaMemcpyHostToDevice));
  int blockNum = 512;
  int threadNum = 1024;

  if(argc>=3){
    cout<<"enter grimDim and blockDim.\n";
    scanf("%d", &blockNum);
    scanf("%d",&threadNum);
  }

  MLCS0<<<blockNum, threadNum>>>(seqs_d, num_seqs, alphabet_set_d, Dk_d, Dk_size_d);
  checkCudaErrors(cudaDeviceSynchronize());
  // do util Dk is empty

  int *error;
  cudaMalloc(&error, sizeof(int));
  int error_host = 0;
  checkCudaErrors(cudaMemcpy(error, &error_host, sizeof(int), cudaMemcpyHostToDevice));
  clock_t start, end;
  double time = 0.0;

  while (Dk_size > 0)
  {
    k++;
    start = clock();
    MLCS(seqs_d, num_seqs, alphabet_set_d, Dk_d, Dk_size_d, blockNum, threadNum, error);
    end = clock();
    time += double(end - start) * 1000 / CLOCKS_PER_SEC;
    checkCudaErrors(cudaMemcpy(*Dk, Dk_d, sizeof(int) * MAX_SEQS_NUM * MAX_DK_SIZE, cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaMemcpy(&Dk_size, Dk_size_d, sizeof(int), cudaMemcpyDeviceToHost));

    checkCudaErrors(cudaMemcpy(&error_host, error, sizeof(int), cudaMemcpyDeviceToHost));
    if (error_host > 0)
    {
      cout << "Error: at least " << error_host << " of them.\n";
      exit(EXIT_FAILURE);
    }
  if (argc >= 4)
      {
        int charNumber[256] = {0};
      printf("\n{D%d}: |%d| \n", k, Dk_size);
      for (int i = 0; i < Dk_size; i++)
      {
        charNumber[s[Dk[i][num_seqs - 1]]]++;
        printf("%c:[", s[Dk[i][num_seqs - 1]]);
        for (int j = 0; j < num_seqs; j++)
        {
          printf("%d,", Dk[i][j]);
        }
        printf("]\n");
      }
        getchar();
    }
    

    vector<Point> DkNow;
    for (int i = 0; i < Dk_size; i++)
    {
      DkNow.push_back(Point(num_seqs, Dk[i]));
      // cout<<"[";
      // for(int j = 0; j < num_seqs; j++){
      //   cout<<Dk[i][j]<<",";
      // }
      // cout<<"]"<<endl;
    }
    D.push_back(DkNow);
    // printf("\n{D%d}: |%d|= A%d-G%d-C%d-T%d\n", k, Dk_size, charNumber['A'], charNumber['G'], charNumber['C'], charNumber['T']);
    
  }
  checkCudaErrors(cudaFree(Dk_d));
  checkCudaErrors(cudaFree(Dk_size_d));
  for (int i = 0; i < num_seqs; i++)
  {
    checkCudaErrors(cudaFree(seqs[i]));
  }
  checkCudaErrors(cudaFree(seqs_d));
  checkCudaErrors(cudaFree(alphabet_set_d));
  // print the result
  printf("Result: LCS length:%d\n", k - 1);
  cout << "GPU USE TIME:" << time << " ms.\n";
  // print a lcs sample:
  k--;
  if (k > 0)
  {
    string lcs = "";
    Point p = D[k][0];
    lcs = sequences[0][p.p[0]] + lcs;
    k--;
    while (k > 0)
    {
      // cout<<k<<" "<<D[k].size()<<endl;
      // for(int i = 0; i < D[k].size(); i++){
      //   cout<<D[k][i].toString()<<endl;
      // }
      // getchar();
      for (Point q : D[k])
      {
        if (q.isSmallerThan(p))
        {
          p = q;

          lcs = sequences[0][p.p[0]] + lcs;
          k--;
          break;
        }
      }
    }
    cout << "a lcs sample :\n"
         << lcs << endl;
  }

  exit(EXIT_SUCCESS);
}
