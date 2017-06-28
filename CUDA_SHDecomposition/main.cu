#include <stdio.h>
#include "SyncedMemory.h"
#include "imageio.h"
#include "mipmap.h"
#include "spectrum.h"
#include "sh.h"
#include "timer.h"

// define CPU to run the spherical harmonic evaluation on CPU
// define ZERO to run the spherical harmonic evaluation on CUDA, without any optimization
// define TWO to run the spherical harmonic evaluation on CUDA, without some optimization
// define THREE to run the spherical harmonic evaluation on CUDA, without full optimization
//#define CPU
#define ZERO
//#define ONE
//#define TWO
//#define THREE

#define CHECK {\
	auto e = cudaDeviceSynchronize();\
	if (e != cudaSuccess) {\
		printf("At " __FILE__ ":%d, %s\n", __LINE__, cudaGetErrorString(e));\
		abort();\
	}\
}

__device__ int Cuda_SHIndex(int l, int m) {
	return l * l + l + m;
}
__device__ float divfact(int a, int b) {
	if(b == 0) return 1.f;
	float fa = a, fb = b > 0 ? b : -b;
	float v = 1.f;
	for(float x = fa - fb + 1.f; x <= fa + fb; x += 1.f) {
		v *= x;
	}
	return 1.f / v;
}
__device__ float Cuda_K(int l, int m) {
	//Klm = ((2 * l + 1) / 4pi) * ((l - |m|)! / (l + |m|)!))^0.5
	return sqrt((2.f * l + 1.f) * 0.07957f * divfact(l, m));
}
__device__ void getVector(float *out, const int phi, const int nPhi, const int theta, const int nTheta) {
	out[0] = sin((theta + 0.5f) / nTheta * 3.1415f) * cos((phi + 0.5f) / nPhi * 2.0f * 3.1415f);
	out[1] = sin((theta + 0.5f) / nTheta * 3.1415f) * sin((phi + 0.5f) / nPhi * 2.0f * 3.1415f);
	out[2] = cos((theta + 0.5f) / nTheta * 3.1415f);
}
__device__ void normalize(float *out) {
	float length = sqrt(out[0] * out[0] + out[1] * out[1] + out[2] * out[2]);
	out[0] /= length;
	out[1] /= length;
	out[2] /= length;
}
__device__ float initP(float x, int m) {
	if(m <= 0)
		return 0;

	float neg = -1.f;
	float dfact = 1.f;
	float xroot = sqrtf(max(0.f, 1.f - x*x));
	float xpow = xroot;

	for(int i = 1; i < m; i++) {
		neg *= -1 * i / i;      // neg = (-1)^l
		dfact *= 2 * i + 1; // dfact = (2*l-1)!!
		xpow *= xroot;    // xpow = powf(1.f - x*x, float(l) * 0.5f);		
	}

	return neg * dfact * xpow;
	//return neg * m;
}

__global__ void Cuda_SHEvaluate(const float *envMap, const int nPhi, const int nTheta, const int lMax, float *coeffs, float *tmpValue, int targetM) {
	const int a = nPhi;
	extern __shared__ float sdata[];

	const int phi = blockIdx.x * blockDim.x + threadIdx.x;
	const int theta = blockIdx.y * blockDim.y + threadIdx.y;
	const int slice = gridDim.z;
	const unsigned int tid = threadIdx.y * blockDim.x + threadIdx.x;
	
	if(phi < nPhi && theta < nTheta) {
		const float *Le = envMap + ((theta * nPhi + phi) * 3);

		float w[3];
		getVector(w, phi, nPhi, theta, nTheta);
		normalize(w);
		float z = w[2];

		const float sqrt2 = sqrt(2.f);
		float xyLen = sqrt(max(0.f, 1.f - w[2] * w[2]));
		float s = (xyLen == 0.f) ? 0 : w[1] / xyLen;
		float c = (xyLen == 0.f) ? 1 : w[0] / xyLen;
		float si = 0, ci = 1;

		float ppYlm = 0;
		float pYlm = 0;
		float currentYlm;

		for(int i = 1; i < blockIdx.z; i++) {
			float oldsi = si;
			si = si * c + ci * s;
			ci = ci * c - oldsi * s;
		}
#ifdef ZERO
		for (int i = blockIdx.z; i <= lMax; i += slice) {
			//for(int i = 0; i <= lMax; i++) {
			if (i == 0) {
				for (int L = 0; L <= lMax; L++) {
					if (L == 0) {
						currentYlm = 1.f;
					}
					else if (L == 1) {
						currentYlm = z;
					}
					else {
						currentYlm = ((2 * L - 1) * z * pYlm - (L - 1) * ppYlm) / L;
					}
					ppYlm = pYlm;
					pYlm = currentYlm;

					currentYlm *= Cuda_K(L, 0);

					for (int j = 0; j < 3; j++) {
						sdata[tid * 3 + j] = Le[j] * currentYlm * sin((theta + 0.5f) / nTheta * 3.1415f) * (3.1415f / nTheta) * (2.f * 3.1415f / nPhi);
						//sdata[tid * 3 + j] = blockIdx.z;
					}
					__syncthreads();

					for (unsigned int s = 1; s < blockDim.x * blockDim.y; s *= 2) {
						if (tid % (2 * s) == 0 && tid + s < blockDim.x * blockDim.y) {
							for (int j = 0; j < 3; j++) {
								sdata[tid * 3 + j] += sdata[(tid + s) * 3 + j];
							}
						}
						__syncthreads();
					}

					if (tid == 0) {
						for (int j = 0; j < 3; j++) {
							tmpValue[(Cuda_SHIndex(L, 0) * gridDim.x * gridDim.y + blockIdx.y * gridDim.x + blockIdx.x) * 3 + j] = sdata[j];
						}
					}
				}
			}
			else {
				float oldsi = si;
				si = si * c + ci * s;
				ci = ci * c - oldsi * s;
				for (int L = i; L <= lMax; L++) {
					if (L == i) {
						currentYlm = initP(z, i);
					}
					else if (L == i + 1) {
						currentYlm = z *(2 * (i + 1) - 1) * pYlm;;
					}
					else {
						currentYlm = ((2 * (L - 1) + 1) * z * pYlm - (L - 1 + i) * ppYlm) / (L - i);
					}
					ppYlm = pYlm;
					pYlm = currentYlm;

					float tmpYlm = sqrt2 * Cuda_K(L, i) * currentYlm * si;

					for (int j = 0; j < 3; j++) {
						sdata[tid * 3 + j] = Le[j] * tmpYlm * sin((theta + 0.5f) / nTheta * 3.1415f) * (3.1415f / nTheta) * (2.f * 3.1415f / nPhi);
						//sdata[tid * 3 + j] = blockIdx.z;
					}
					__syncthreads();

					for (unsigned int s = 1; s < blockDim.x * blockDim.y; s *= 2) {
						if (tid % (2 * s) == 0 && tid + s < blockDim.x * blockDim.y) {
							for (int j = 0; j < 3; j++) {
								sdata[tid * 3 + j] += sdata[(tid + s) * 3 + j];
							}
						}
						__syncthreads();
					}

					if (tid == 0) {
						for (int j = 0; j < 3; j++) {
							tmpValue[(Cuda_SHIndex(L, -i) * gridDim.x * gridDim.y + blockIdx.y * gridDim.x + blockIdx.x) * 3 + j] = sdata[j];
						}
					}

					currentYlm *= sqrt2 * Cuda_K(L, i) *ci;

					for (int j = 0; j < 3; j++) {
						sdata[tid * 3 + j] = Le[j] * currentYlm * sin((theta + 0.5f) / nTheta * 3.1415f) * (3.1415f / nTheta) * (2.f * 3.1415f / nPhi);
						//sdata[tid * 3 + j] = blockIdx.z;
					}
					__syncthreads();

					for (unsigned int s = 1; s < blockDim.x * blockDim.y; s *= 2) {
						if (tid % (2 * s) == 0 && tid + s < blockDim.x * blockDim.y) {
							for (int j = 0; j < 3; j++) {
								sdata[tid * 3 + j] += sdata[(tid + s) * 3 + j];
							}
						}
						__syncthreads();
					}

					if (tid == 0) {
						for (int j = 0; j < 3; j++) {
							tmpValue[(Cuda_SHIndex(L, i) * gridDim.x * gridDim.y + blockIdx.y * gridDim.x + blockIdx.x) * 3 + j] = sdata[j];
						}
					}

				}
			}
			for (int j = 0; j < slice - 1; j++) {
				float oldsi = si;
				si = si * c + ci * s;
				ci = ci * c - oldsi * s;
			}
		}
#endif // ZERO
#ifdef  ONE
		for (int i = blockIdx.z; i <= lMax; i += slice) {
			//for(int i = 0; i <= lMax; i++) {
			if (i == 0) {
				for (int L = 0; L <= lMax; L++) {
					if (L == 0) { currentYlm = 1.f; }
					else if (L == 1) { currentYlm = z; }
					else { currentYlm = ((2 * L - 1) * z * pYlm - (L - 1) * ppYlm) / L; }

					ppYlm = pYlm;
					pYlm = currentYlm;
					currentYlm *= Cuda_K(L, 0);

					float temp = currentYlm * sin((theta + 0.5f) / nTheta * 3.1415f) * (3.1415f / nTheta) * (2.f * 3.1415f / nPhi);
					sdata[tid * 3 + 0] = Le[0] * temp;
					sdata[tid * 3 + 1] = Le[1] * temp;
					sdata[tid * 3 + 2] = Le[2] * temp;

					__syncthreads();

					for (unsigned int s = 1; s < blockDim.x * blockDim.y; s *= 2) {
						if (tid % (2 * s) == 0 && tid + s < blockDim.x * blockDim.y) {
							sdata[tid * 3 + 0] += sdata[(tid + s) * 3 + 0];
							sdata[tid * 3 + 1] += sdata[(tid + s) * 3 + 1];
							sdata[tid * 3 + 2] += sdata[(tid + s) * 3 + 2];
						}
						__syncthreads();
					}

					if (tid == 0) {
						int temp = (Cuda_SHIndex(L, 0) * gridDim.x * gridDim.y + blockIdx.y * gridDim.x + blockIdx.x) * 3;
						tmpValue[temp + 0] = sdata[0];
						tmpValue[temp + 1] = sdata[1];
						tmpValue[temp + 2] = sdata[2];
					}
				}
			}
			else {
				float oldsi = si;
				si = si * c + ci * s;
				ci = ci * c - oldsi * s;
				for (int L = i; L <= lMax; L++) {
					if (L == i) { currentYlm = initP(z, i); }
					else if (L == i + 1) { currentYlm = z *(2 * (i + 1) - 1) * pYlm;; }
					else { currentYlm = ((2 * (L - 1) + 1) * z * pYlm - (L - 1 + i) * ppYlm) / (L - i); }

					ppYlm = pYlm;
					pYlm = currentYlm;
					float tmpYlm = sqrt2 * Cuda_K(L, i) * currentYlm * si;

					float temp = tmpYlm * sin((theta + 0.5f) / nTheta * 3.1415f) * (3.1415f / nTheta) * (2.f * 3.1415f / nPhi);
					sdata[tid * 3 + 0] = Le[0] * temp;
					sdata[tid * 3 + 1] = Le[1] * temp;
					sdata[tid * 3 + 2] = Le[2] * temp;
					__syncthreads();

					for (unsigned int s = 1; s < blockDim.x * blockDim.y; s *= 2) {
						if (tid % (2 * s) == 0 && tid + s < blockDim.x * blockDim.y) {
							sdata[tid * 3 + 0] += sdata[(tid + s) * 3 + 0];
							sdata[tid * 3 + 1] += sdata[(tid + s) * 3 + 1];
							sdata[tid * 3 + 2] += sdata[(tid + s) * 3 + 2];
						}
						__syncthreads();
					}

					if (tid == 0) {
						int temp = (Cuda_SHIndex(L, -i) * gridDim.x * gridDim.y + blockIdx.y * gridDim.x + blockIdx.x) * 3;
						tmpValue[temp + 0] = sdata[0];
						tmpValue[temp + 1] = sdata[1];
						tmpValue[temp + 2] = sdata[2];
					}

					currentYlm *= sqrt2 * Cuda_K(L, i) *ci;
					temp = currentYlm * sin((theta + 0.5f) / nTheta * 3.1415f) * (3.1415f / nTheta) * (2.f * 3.1415f / nPhi);
					sdata[tid * 3 + 0] = Le[0] * temp;
					sdata[tid * 3 + 1] = Le[1] * temp;
					sdata[tid * 3 + 2] = Le[2] * temp;
					__syncthreads();

					for (unsigned int s = 1; s < blockDim.x * blockDim.y; s *= 2) {
						if (tid % (2 * s) == 0 && tid + s < blockDim.x * blockDim.y) {
							sdata[tid * 3 + 0] += sdata[(tid + s) * 3 + 0];
							sdata[tid * 3 + 1] += sdata[(tid + s) * 3 + 1];
							sdata[tid * 3 + 2] += sdata[(tid + s) * 3 + 2];
						}
						__syncthreads();
					}

					if (tid == 0) {
						int temp = (Cuda_SHIndex(L, i) * gridDim.x * gridDim.y + blockIdx.y * gridDim.x + blockIdx.x) * 3;
						tmpValue[temp + 0] = sdata[0];
						tmpValue[temp + 1] = sdata[1];
						tmpValue[temp + 2] = sdata[2];
					}
				}
			}
			for (int j = 0; j < slice - 1; j++) {
				float oldsi = si;
				si = si * c + ci * s;
				ci = ci * c - oldsi * s;
			}
		}
#endif //  ONE
#ifdef TWO
		for (int i = blockIdx.z; i <= lMax; i += slice) {
			//for(int i = 0; i <= lMax; i++) {
			if (i == 0) {
				for (int L = 0; L <= lMax; L++) {
					if (L == 0) { currentYlm = 1.f; }
					else if (L == 1) { currentYlm = z; }
					else { currentYlm = ((2 * L - 1) * z * pYlm - (L - 1) * ppYlm) / L; }

					ppYlm = pYlm;
					pYlm = currentYlm;
					currentYlm *= Cuda_K(L, 0);

					float temp = currentYlm * sin((theta + 0.5f) / nTheta * 3.1415f) * (3.1415f / nTheta) * (2.f * 3.1415f / nPhi);
					sdata[tid * 3 + 0] = Le[0] * temp;
					sdata[tid * 3 + 1] = Le[1] * temp;
					sdata[tid * 3 + 2] = Le[2] * temp;

					__syncthreads();

					for (unsigned int s = blockDim.x * blockDim.y/2; s > 0; s >>= 1) {
						if (tid < s) {
							sdata[tid * 3 + 0] += sdata[(tid + s) * 3 + 0];
							sdata[tid * 3 + 1] += sdata[(tid + s) * 3 + 1];
							sdata[tid * 3 + 2] += sdata[(tid + s) * 3 + 2];
						}
						__syncthreads();
					}

					if (tid == 0) {
						int temp = (Cuda_SHIndex(L, 0) * gridDim.x * gridDim.y + blockIdx.y * gridDim.x + blockIdx.x) * 3;
						tmpValue[temp + 0] = sdata[0];
						tmpValue[temp + 1] = sdata[1];
						tmpValue[temp + 2] = sdata[2];
					}
				}
			}
			else {
				float oldsi = si;
				si = si * c + ci * s;
				ci = ci * c - oldsi * s;
				for (int L = i; L <= lMax; L++) {
					if (L == i) { currentYlm = initP(z, i); }
					else if (L == i + 1) { currentYlm = z *(2 * (i + 1) - 1) * pYlm;; }
					else { currentYlm = ((2 * (L - 1) + 1) * z * pYlm - (L - 1 + i) * ppYlm) / (L - i); }

					ppYlm = pYlm;
					pYlm = currentYlm;
					float tmpYlm = sqrt2 * Cuda_K(L, i) * currentYlm * si;

					float temp = tmpYlm * sin((theta + 0.5f) / nTheta * 3.1415f) * (3.1415f / nTheta) * (2.f * 3.1415f / nPhi);
					sdata[tid * 3 + 0] = Le[0] * temp;
					sdata[tid * 3 + 1] = Le[1] * temp;
					sdata[tid * 3 + 2] = Le[2] * temp;
					__syncthreads();

					for (unsigned int s = blockDim.x * blockDim.y / 2; s > 0; s >>= 1) {
						if (tid < s) {
							sdata[tid * 3 + 0] += sdata[(tid + s) * 3 + 0];
							sdata[tid * 3 + 1] += sdata[(tid + s) * 3 + 1];
							sdata[tid * 3 + 2] += sdata[(tid + s) * 3 + 2];
						}
						__syncthreads();
					}

					if (tid == 0) {
						int temp = (Cuda_SHIndex(L, -i) * gridDim.x * gridDim.y + blockIdx.y * gridDim.x + blockIdx.x) * 3;
						tmpValue[temp + 0] = sdata[0];
						tmpValue[temp + 1] = sdata[1];
						tmpValue[temp + 2] = sdata[2];
					}

					currentYlm *= sqrt2 * Cuda_K(L, i) *ci;
					temp = currentYlm * sin((theta + 0.5f) / nTheta * 3.1415f) * (3.1415f / nTheta) * (2.f * 3.1415f / nPhi);
					sdata[tid * 3 + 0] = Le[0] * temp;
					sdata[tid * 3 + 1] = Le[1] * temp;
					sdata[tid * 3 + 2] = Le[2] * temp;
					__syncthreads();

					for (unsigned int s = blockDim.x * blockDim.y / 2; s > 0; s >>= 1) {
						if (tid < s) {
							sdata[tid * 3 + 0] += sdata[(tid + s) * 3 + 0];
							sdata[tid * 3 + 1] += sdata[(tid + s) * 3 + 1];
							sdata[tid * 3 + 2] += sdata[(tid + s) * 3 + 2];
						}
						__syncthreads();
					}

					if (tid == 0) {
						int temp = (Cuda_SHIndex(L, i) * gridDim.x * gridDim.y + blockIdx.y * gridDim.x + blockIdx.x) * 3;
						tmpValue[temp + 0] = sdata[0];
						tmpValue[temp + 1] = sdata[1];
						tmpValue[temp + 2] = sdata[2];
					}
				}
			}
			for (int j = 0; j < slice - 1; j++) {
				float oldsi = si;
				si = si * c + ci * s;
				ci = ci * c - oldsi * s;
			}
		}
#endif // TWO
#ifdef THREE
		for (int i = blockIdx.z; i <= lMax; i += slice) {
			//for(int i = 0; i <= lMax; i++) {
			if (i == 0) {
				for (int L = 0; L <= lMax; L++) {
					if (L == 0) { currentYlm = 1.f; }
					else if (L == 1) { currentYlm = z; }
					else { currentYlm = ((2 * L - 1) * z * pYlm - (L - 1) * ppYlm) / L; }

					ppYlm = pYlm;
					pYlm = currentYlm;
					currentYlm *= Cuda_K(L, 0);

					float temp = currentYlm * sin((theta + 0.5f) / nTheta * 3.1415f) * (3.1415f / nTheta) * (2.f * 3.1415f / nPhi);
					sdata[tid * 3 + 0] = Le[0] * temp;
					sdata[tid * 3 + 1] = Le[1] * temp;
					sdata[tid * 3 + 2] = Le[2] * temp;
					__syncthreads();

					if (tid < 128) {
						sdata[tid * 3 + 0] += sdata[(tid + 128) * 3 + 0];
						sdata[tid * 3 + 1] += sdata[(tid + 128) * 3 + 1];
						sdata[tid * 3 + 2] += sdata[(tid + 128) * 3 + 2];
						__syncthreads();
						sdata[tid * 3 + 0] += sdata[(tid + 64) * 3 + 0];
						sdata[tid * 3 + 1] += sdata[(tid + 64) * 3 + 1];
						sdata[tid * 3 + 2] += sdata[(tid + 64) * 3 + 2];
						__syncthreads();
						sdata[tid * 3 + 0] += sdata[(tid + 32) * 3 + 0];
						sdata[tid * 3 + 1] += sdata[(tid + 32) * 3 + 1];
						sdata[tid * 3 + 2] += sdata[(tid + 32) * 3 + 2];
						__syncthreads();
						sdata[tid * 3 + 0] += sdata[(tid + 16) * 3 + 0];
						sdata[tid * 3 + 1] += sdata[(tid + 16) * 3 + 1];
						sdata[tid * 3 + 2] += sdata[(tid + 16) * 3 + 2];
						__syncthreads();
						sdata[tid * 3 + 0] += sdata[(tid + 8) * 3 + 0];
						sdata[tid * 3 + 1] += sdata[(tid + 8) * 3 + 1];
						sdata[tid * 3 + 2] += sdata[(tid + 8) * 3 + 2];
						__syncthreads();
						sdata[tid * 3 + 0] += sdata[(tid + 4) * 3 + 0];
						sdata[tid * 3 + 1] += sdata[(tid + 4) * 3 + 1];
						sdata[tid * 3 + 2] += sdata[(tid + 4) * 3 + 2];
						__syncthreads();
						sdata[tid * 3 + 0] += sdata[(tid + 2) * 3 + 0];
						sdata[tid * 3 + 1] += sdata[(tid + 2) * 3 + 1];
						sdata[tid * 3 + 2] += sdata[(tid + 2) * 3 + 2];
						__syncthreads();
						sdata[tid * 3 + 0] += sdata[(tid + 1) * 3 + 0];
						sdata[tid * 3 + 1] += sdata[(tid + 1) * 3 + 1];
						sdata[tid * 3 + 2] += sdata[(tid + 1) * 3 + 2];
						__syncthreads();
					}

					if (tid == 0) {
						int temp = (Cuda_SHIndex(L, 0) * gridDim.x * gridDim.y + blockIdx.y * gridDim.x + blockIdx.x) * 3;
						tmpValue[temp + 0] = sdata[0];
						tmpValue[temp + 1] = sdata[1];
						tmpValue[temp + 2] = sdata[2];
					}
				}
			}
			else {
				float oldsi = si;
				si = si * c + ci * s;
				ci = ci * c - oldsi * s;
				for (int L = i; L <= lMax; L++) {
					if (L == i) { currentYlm = initP(z, i); }
					else if (L == i + 1) { currentYlm = z *(2 * (i + 1) - 1) * pYlm;; }
					else { currentYlm = ((2 * (L - 1) + 1) * z * pYlm - (L - 1 + i) * ppYlm) / (L - i); }

					ppYlm = pYlm;
					pYlm = currentYlm;
					float tmpYlm = sqrt2 * Cuda_K(L, i) * currentYlm * si;

					float temp = tmpYlm * sin((theta + 0.5f) / nTheta * 3.1415f) * (3.1415f / nTheta) * (2.f * 3.1415f / nPhi);
					sdata[tid * 3 + 0] = Le[0] * temp;
					sdata[tid * 3 + 1] = Le[1] * temp;
					sdata[tid * 3 + 2] = Le[2] * temp;
					__syncthreads();

					if (tid < 128) {
						sdata[tid * 3 + 0] += sdata[(tid + 128) * 3 + 0];
						sdata[tid * 3 + 1] += sdata[(tid + 128) * 3 + 1];
						sdata[tid * 3 + 2] += sdata[(tid + 128) * 3 + 2];
						__syncthreads();
						sdata[tid * 3 + 0] += sdata[(tid + 64) * 3 + 0];
						sdata[tid * 3 + 1] += sdata[(tid + 64) * 3 + 1];
						sdata[tid * 3 + 2] += sdata[(tid + 64) * 3 + 2];
						__syncthreads();
						sdata[tid * 3 + 0] += sdata[(tid + 32) * 3 + 0];
						sdata[tid * 3 + 1] += sdata[(tid + 32) * 3 + 1];
						sdata[tid * 3 + 2] += sdata[(tid + 32) * 3 + 2];
						__syncthreads();
						sdata[tid * 3 + 0] += sdata[(tid + 16) * 3 + 0];
						sdata[tid * 3 + 1] += sdata[(tid + 16) * 3 + 1];
						sdata[tid * 3 + 2] += sdata[(tid + 16) * 3 + 2];
						__syncthreads();
						sdata[tid * 3 + 0] += sdata[(tid + 8) * 3 + 0];
						sdata[tid * 3 + 1] += sdata[(tid + 8) * 3 + 1];
						sdata[tid * 3 + 2] += sdata[(tid + 8) * 3 + 2];
						__syncthreads();
						sdata[tid * 3 + 0] += sdata[(tid + 4) * 3 + 0];
						sdata[tid * 3 + 1] += sdata[(tid + 4) * 3 + 1];
						sdata[tid * 3 + 2] += sdata[(tid + 4) * 3 + 2];
						__syncthreads();
						sdata[tid * 3 + 0] += sdata[(tid + 2) * 3 + 0];
						sdata[tid * 3 + 1] += sdata[(tid + 2) * 3 + 1];
						sdata[tid * 3 + 2] += sdata[(tid + 2) * 3 + 2];
						__syncthreads();
						sdata[tid * 3 + 0] += sdata[(tid + 1) * 3 + 0];
						sdata[tid * 3 + 1] += sdata[(tid + 1) * 3 + 1];
						sdata[tid * 3 + 2] += sdata[(tid + 1) * 3 + 2];
						__syncthreads();
					}

					if (tid == 0) {
						int temp = (Cuda_SHIndex(L, -i) * gridDim.x * gridDim.y + blockIdx.y * gridDim.x + blockIdx.x) * 3;
						tmpValue[temp + 0] = sdata[0];
						tmpValue[temp + 1] = sdata[1];
						tmpValue[temp + 2] = sdata[2];
					}

					currentYlm *= sqrt2 * Cuda_K(L, i) *ci;
					temp = currentYlm * sin((theta + 0.5f) / nTheta * 3.1415f) * (3.1415f / nTheta) * (2.f * 3.1415f / nPhi);
					sdata[tid * 3 + 0] = Le[0] * temp;
					sdata[tid * 3 + 1] = Le[1] * temp;
					sdata[tid * 3 + 2] = Le[2] * temp;
					__syncthreads();

					if (tid < 128) {
						sdata[tid * 3 + 0] += sdata[(tid + 128) * 3 + 0];
						sdata[tid * 3 + 1] += sdata[(tid + 128) * 3 + 1];
						sdata[tid * 3 + 2] += sdata[(tid + 128) * 3 + 2];
						__syncthreads();
						sdata[tid * 3 + 0] += sdata[(tid + 64) * 3 + 0];
						sdata[tid * 3 + 1] += sdata[(tid + 64) * 3 + 1];
						sdata[tid * 3 + 2] += sdata[(tid + 64) * 3 + 2];
						__syncthreads();
						sdata[tid * 3 + 0] += sdata[(tid + 32) * 3 + 0];
						sdata[tid * 3 + 1] += sdata[(tid + 32) * 3 + 1];
						sdata[tid * 3 + 2] += sdata[(tid + 32) * 3 + 2];
						__syncthreads();
						sdata[tid * 3 + 0] += sdata[(tid + 16) * 3 + 0];
						sdata[tid * 3 + 1] += sdata[(tid + 16) * 3 + 1];
						sdata[tid * 3 + 2] += sdata[(tid + 16) * 3 + 2];
						__syncthreads();
						sdata[tid * 3 + 0] += sdata[(tid + 8) * 3 + 0];
						sdata[tid * 3 + 1] += sdata[(tid + 8) * 3 + 1];
						sdata[tid * 3 + 2] += sdata[(tid + 8) * 3 + 2];
						__syncthreads();
						sdata[tid * 3 + 0] += sdata[(tid + 4) * 3 + 0];
						sdata[tid * 3 + 1] += sdata[(tid + 4) * 3 + 1];
						sdata[tid * 3 + 2] += sdata[(tid + 4) * 3 + 2];
						__syncthreads();
						sdata[tid * 3 + 0] += sdata[(tid + 2) * 3 + 0];
						sdata[tid * 3 + 1] += sdata[(tid + 2) * 3 + 1];
						sdata[tid * 3 + 2] += sdata[(tid + 2) * 3 + 2];
						__syncthreads();
						sdata[tid * 3 + 0] += sdata[(tid + 1) * 3 + 0];
						sdata[tid * 3 + 1] += sdata[(tid + 1) * 3 + 1];
						sdata[tid * 3 + 2] += sdata[(tid + 1) * 3 + 2];
						__syncthreads();
					}

					if (tid == 0) {
						int temp = (Cuda_SHIndex(L, i) * gridDim.x * gridDim.y + blockIdx.y * gridDim.x + blockIdx.x) * 3;
						tmpValue[temp + 0] = sdata[0];
						tmpValue[temp + 1] = sdata[1];
						tmpValue[temp + 2] = sdata[2];
					}
				}
			}
			for (int j = 0; j < slice - 1; j++) {
				float oldsi = si;
				si = si * c + ci * s;
				ci = ci * c - oldsi * s;
			}
		}
#endif // THREE	
	}
}

void MySHDEvaluation(Transform LightToWorld, MIPMap<RGBSpectrum> *radianceMap, int lmax, Spectrum *coeffs) {
	/*
		The following is totally from PBRT.
		SHEvaluate(const Vector, int, float*) is the PBRT implementation
	*/
	for(int i = 0; i < SHTerms(lmax); ++i)
		coeffs[i] = 0.f;
	int ntheta = radianceMap->Height(), nphi = radianceMap->Width();
	if(min(ntheta, nphi) > 50) {
		// Project _InfiniteAreaLight_ to SH from lat-long representation
		// Precompute $\theta$ and $\phi$ values for lat-long map projection
		float *buf = new float[2 * ntheta + 2 * nphi];
		float *bufp = buf;
		float *sintheta = bufp;  bufp += ntheta;
		float *costheta = bufp;  bufp += ntheta;
		float *sinphi = bufp;    bufp += nphi;
		float *cosphi = bufp;
		for(int theta = 0; theta < ntheta; ++theta) {
			sintheta[theta] = sinf((theta + .5f) / ntheta * M_PI);
			costheta[theta] = cosf((theta + .5f) / ntheta * M_PI);
		}
		for(int phi = 0; phi < nphi; ++phi) {
			sinphi[phi] = sinf((phi + .5f) / nphi * 2.f * M_PI);
			cosphi[phi] = cosf((phi + .5f) / nphi * 2.f * M_PI);
		}
		float *Ylm = ALLOCA(float, SHTerms(lmax));
		for(int i = 0; i < SHTerms(lmax); i++) {
			Ylm[i] = 0;
		}
		for(int theta = 0; theta < ntheta; ++theta) {
			for(int phi = 0; phi < nphi; ++phi) {
				// Add _InfiniteAreaLight_ texel's contribution to SH coefficients
				Vector w = Vector(sintheta[theta] * cosphi[phi],
								  sintheta[theta] * sinphi[phi],
								  costheta[theta]);
				//We don't need the transformation so we modified this line
				//w = Normalize(LightToWorld(w));
				w = Normalize(w);
				
				Spectrum Le = Spectrum(radianceMap->Texel(0, phi, theta),
					 				   SPECTRUM_ILLUMINANT);
				SHEvaluate(w, lmax, Ylm);

				for(int i = 0; i < SHTerms(lmax); ++i)
					coeffs[i] += Le * Ylm[i] * sintheta[theta] *
					(M_PI / ntheta) * (2.f * M_PI / nphi);
			}
		}
		// Free memory used for lat-long theta and phi values
		delete[] buf;
	}
	else {
		printf("use project cube instead\n");
	}
}

int main(int argc, char **argv)
{
	Timer *timer = new Timer();

	int width;
	int height;
	
	// Read the environment map
	RGBSpectrum *texels = ReadImage("grace-new_latlong.exr", &width, &height);
	printf("width = %d, height = %d\n", width, height);
	//int lmax = 25;

#ifdef  CPU
	printf("cpu\n");
	MIPMap<RGBSpectrum> *radianceMap = new MIPMap<RGBSpectrum>(width, height, texels);
	for (int lmax = 0; lmax < 25; lmax++) {
		timer->Reset();
		timer->Start();
		
		Spectrum *coeffs_Spectrum = (Spectrum *)malloc(sizeof(Spectrum) * SHTerms(lmax));

		// Spherical harmonic evaluation on CPU
		MySHDEvaluation(Transform(), radianceMap, lmax, coeffs_Spectrum);
		timer->Stop();
		printf("%2d   time: %lf\n",lmax, timer->Time());
	}
#else
	printf("gpu\n");
	for (int lmax = 0; lmax < 25; lmax++) {
		// Memory size for environment map
		const int sizeE = width * height * 3;
		// Memory size for spherical harmonic coefs
		const int sizeC = SHTerms(lmax) * 3;
		// Allocate memory
		MemoryBuffer<float> envMap(sizeE), coeffs(sizeC);
		auto envMap_s = envMap.CreateSync(sizeE);
		CHECK;
		auto coeffs_s = coeffs.CreateSync(sizeC);
		CHECK;

		float *envMap_cpu = envMap_s.get_cpu_wo();
		float *coeffs_cpu = coeffs_s.get_cpu_wo();

		// Pass the environment to gpu
		for (int i = 0; i < width * height; i++) {
			texels[i].ToRGB(envMap_cpu + i * 3);
		}
		for (int i = 0; i < SHTerms(lmax); i++) {
			for (int j = 0; j < 3; j++) {
				coeffs_cpu[i * 3 + j] = 0;
			}
		}

		dim3 blocks = dim3(width / 16, height / 16, 1);
		dim3 threads = dim3(16, 16);

		// Memory size for the addition result for every block
		const int sizeT = blocks.x * blocks.y * 3 * SHTerms(lmax);
		MemoryBuffer<float> tmpValue(sizeT);
		auto tmpValue_s = tmpValue.CreateSync(sizeT);
		CHECK;
		float *tmpValue_cpu = tmpValue_s.get_cpu_rw();
		for (int i = 0; i < sizeT; i++) {
			tmpValue_cpu[i] = 0;
		}

		timer->Reset();
		timer->Start();

		float *tmpValue_gpu = tmpValue_s.get_gpu_rw();
		// Spherical harmonic evaluation on GPU
		Cuda_SHEvaluate << <blocks, threads, sizeof(float) * threads.x * threads.y * 3 >> > (envMap_s.get_gpu_ro(), width, height, lmax, coeffs_s.get_gpu_rw(), tmpValue_gpu, 0);
		CHECK;

		// We directely add the result in every block together here, in CPU
		const float *tmpValue_cpu_ro = tmpValue_s.get_cpu_ro();
		float *value = new float[SHTerms(lmax) * 3];
		for (int i = 0; i < SHTerms(lmax) * 3; i++) {
			value[i] = 0;
		}
		for (int i = 0; i < blocks.x * blocks.y; i++) {
			for (int L = 0; L < SHTerms(lmax); L++) {
				for (int j = 0; j < 3; j++) {
					value[L * 3 + j] += tmpValue_cpu_ro[(L * blocks.x * blocks.y + i) * 3 + j];
				}
			}
		}

		timer->Stop();
		printf("%2d   time: %lf\n", lmax, timer->Time());

		envMap.Free();
		coeffs.Free();
		tmpValue.Free();
	}
#endif //  CPU

	system("pause");
    return 0;
}
