# ifndef __CODEGEN_HPP__
# define __CODEGEN_HPP__

# include <sstream>
# include <vector>
# include <tuple>
# include "autoFS.hpp"


class codeGen 
{
	private:
		std::stringstream header;
		std::stringstream gpu_code;
		std::stringstream host_code;
		std::stringstream gold_code;
		semiStencil* fs_stencil;
		int bx, by, sn;
		int L, M, N, iterations;
		int stream_unroll;
		std::string stencil_name;
		bool check_correctness;
	public:
		codeGen (semiStencil* fs, int bx_, int by_, int sn_, int s_unroll, std::string stc_name, bool check) 
			: fs_stencil (fs), bx (bx_), by (by_), sn (sn_), stream_unroll (s_unroll), stencil_name (stc_name), check_correctness (check)
			{}
		codeGen (semiStencil* fs, std::string stc_name) 
			: fs_stencil (fs), stencil_name (stc_name) 
			{}
		void header_gen ();
		void gpu_code_gen ();
		void host_code_gen ();
		std::string gold_gpu_code_gen ();
		void gold_code_gen ();
		void output (const std::string &);
};

void codeGen::output (const std::string & out_name)
{
	// check whether the configuration is valid
	int halo2 = fs_stencil->get_order () * 2;
	if ((halo2 >= bx && fs_stencil->forward_i_avilable ()) ||
		(halo2 >= by && fs_stencil->forward_j_avilable ())) {
		std::cout << "Invalid configuration!" << std::endl;
		exit (-1);
	}

	header_gen ();
	gpu_code_gen ();
	host_code_gen ();

	std::ofstream outfile;
	outfile.open(out_name, std::ios::out | std::ios::trunc );
	outfile << header.str();
	outfile << gpu_code.str();
	if (check_correctness) {
		outfile << gold_gpu_code_gen ();
	}
	outfile << host_code.str();

	outfile.close();
}

void codeGen::header_gen ()
{
	header << "# include <stdio.h>\n";
	header << "# include <cuda.h>\n";
	header << "# include \"common.hpp\"\n";
	header << "\n#define max(x,y) ((x) > (y) ? (x) : (y)) \n";
	header << "#define min(x,y) ((x) < (y) ? (x) : (y)) \n";
	header << "#define ceil(a,b) ((a) % (b) == 0 ? (a) / (b) : ((a) / (b) + 1)) \n";
	header << std::endl;
	fs_stencil->get_problem_size (L, M, N, iterations);
	header << "#define L " << L << std::endl;
	header << "#define M " << M << std::endl;
	header << "#define N " << N << std::endl;
	header << "#define Iterations " << iterations << std::endl;
	header << std::endl;
	header << "#define Range " << fs_stencil->get_high_k () - fs_stencil->get_low_k () + 1 << std::endl;
	header << "#define Halo " << fs_stencil->get_order () << std::endl;
	header << "#define Dist " << fs_stencil->get_distance () << std::endl;
	header << "#define Bx " << bx << std::endl;
	header << "#define By " << by << std::endl;
	header << "#define Sn " << sn << std::endl;
	header << std::endl;

	std::string indent = "\t";

	header << "void check_error (const char* message) {\n";
	header << indent << "cudaError_t error = cudaGetLastError ();\n";
	header << indent << "if (error != cudaSuccess) {\n";
	header << indent << indent << "printf (\"CUDA error : %s, %s\\n\", message, cudaGetErrorString (error));\n";
	header << indent << indent << "exit(-1);\n";
	header << indent << "}\n";
	header << "}\n" << "\n";

}

void codeGen::gpu_code_gen ()
{
	std::string indent = "\t";

	gpu_code << "__global__ void " << stencil_name << " (double *d_in, double *d_out)\n";
	gpu_code << "{\n";
	gpu_code << indent << "int i0 = (int)(blockIdx.x) * ((int)blockDim.x-Halo*2) + (int)(threadIdx.x);\n";
	gpu_code << indent << "int i = (int)(threadIdx.x);\n";
	gpu_code << indent << "int j0 = (int)(blockIdx.y) * ((int)blockDim.y-Halo*2) + (int)(threadIdx.y);\n";
	gpu_code << indent << "int j = (int)(threadIdx.y);\n";
	gpu_code << indent << "int k = (int)(blockIdx.z) * Sn;\n";
	gpu_code << indent << "int k_ed = min(L - Halo, k + Sn + Halo);\n";

	gpu_code << "\n";
	gpu_code << indent << "double (*in)[M][N] = (double (*)[M][N]) d_in;\n";
	gpu_code << indent << "double (*out)[M][N] = (double (*)[M][N]) d_out;\n";

	gpu_code << "\n";
	gpu_code << indent << "double __shared__ in_shm[Range][By][Bx];\n";

	int low_k = fs_stencil->get_low_k ();
	int high_k = fs_stencil->get_high_k ();
	int range = high_k - low_k + 1;
	gpu_code << indent << "int k0";
	for (int i = 1; i < range; i ++) {
		gpu_code << ", k" << i;
	}
	gpu_code << ";" << std::endl;
	gpu_code << indent << "bool i_ok = (i >= Halo && i < Bx - Halo && i0 < N - Halo);" << std::endl;
	gpu_code << indent << "bool j_ok = (j >= Halo && j < By - Halo && j0 < M - Halo);" << std::endl;
	gpu_code << std::endl;
	gpu_code << indent << "int k_st = k + Halo;" << std::endl;
	if (fs_stencil->get_distance () < high_k) {
		gpu_code << indent << "k = k + Halo - Dist;" << std::endl;
	}
	gpu_code << std::endl;

	gpu_code << indent << "// Initial loads" << std::endl;
	gpu_code << indent << "if (k + " << high_k - 1 << " < k_ed && j0 < M && i0 < N) {" << std::endl;
	for (int i = low_k; i < 0; i ++) {
		gpu_code << indent << indent << "if (k" << i << " < 0) in_shm[(k" << i<< "+Range)%Range][j][i] = 0;" << std::endl;
		gpu_code << indent << indent << "else ";
		gpu_code << "in_shm[(k" << i << ")%Range][j][i] = in[k" << i << "][j0][i0];" << std::endl;
	}
	for (int i = (low_k > 0 ? low_k : 0); i < high_k; i ++) {
		gpu_code << indent << indent << "in_shm[(k+" << i << ")%Range][j][i] = in[k+" << i << "][j0][i0];" << std::endl;
	}
	gpu_code << indent << "}" << std::endl;
	gpu_code << indent << "__syncthreads ();" << std::endl;
	gpu_code << indent << "#pragma unroll " << stream_unroll << std::endl;
	gpu_code << indent << "for (; k < k_ed; k ++) {" << std::endl;
	for (int i = 0, j = low_k; j <= high_k; i ++, j ++) {
		gpu_code << indent << indent << "k" << i << " = (k+" << (j+range)%range << ") % Range;" << std::endl;
	}
	gpu_code << indent << indent << "in_shm[k" << range - 1 << "][j][i] = in[k+" << high_k << "][j0][i0];" << std::endl << std::endl;
	gpu_code << indent << indent << "//__syncthreads ();" << std::endl;
	// forward k
	gpu_code << indent << indent << "if (k < k_ed - Dist && j_ok && i_ok) {" << std::endl;
	gpu_code << indent << indent << indent << "// forward k" << std::endl;
	gpu_code << indent << indent << indent << "out[k+Dist][j0][i0] = " << fs_stencil->gen_forward_k () << ";" << std::endl;
	gpu_code << indent << indent << "}" << std::endl;
	gpu_code << indent << indent << "__syncthreads ();" << std::endl;

	// forward j
	if (fs_stencil->forward_j_avilable ()) {
		gpu_code << indent << indent << "if (k >= k_st && " 
				<< (fs_stencil->get_order() == fs_stencil->get_distance() ? "" : "j >= Halo - Dist &&") 
				<< " j0 < M - Halo - Dist && j < By - Halo - Dist && i_ok) {" << std::endl;
		gpu_code << indent << indent << indent << "// forward j" << std::endl;
		gpu_code << indent << indent << indent << "atomicAdd(&out[k][j0+Dist][i0], " << fs_stencil->gen_forward_j () << ");" << std::endl;
		gpu_code << indent << indent << "}" << std::endl;
		gpu_code << indent << indent << "//__syncthreads ();" << std::endl;
	}

	// forward i
	if (fs_stencil->forward_i_avilable ()) {
		gpu_code << indent << indent << "if (k >= k_st && j_ok && " 
				<< (fs_stencil->get_order() == fs_stencil->get_distance() ? "" : "i >= Halo - Dist &&") 
				<< " i0 < N - Halo - Dist && i < Bx - Halo - Dist) {" << std::endl;
		gpu_code << indent << indent << indent << "// forward i" << std::endl;
		gpu_code << indent << indent << indent << "atomicAdd(&out[k][j0][i0+Dist], " << fs_stencil->gen_forward_i () << ");" << std::endl;
		gpu_code << indent << indent << "}" << std::endl;
		gpu_code << indent << indent << "//__syncthreads ();" << std::endl;
	}

	// backward
	gpu_code << indent << indent << "if (k >= k_st && j_ok && i_ok) {" << std::endl;
	gpu_code << indent << indent << indent << "// backward" << std::endl;
	gpu_code << indent << indent << indent << "atomicAdd(&out[k][j0][i0], " << fs_stencil->gen_backward () << ");" << std::endl;
	gpu_code << indent << indent << "}" << std::endl;
	gpu_code << indent << indent << "__syncthreads ();" << std::endl;

	gpu_code << indent << "}" << std::endl;
	gpu_code << "}" << std::endl;
}


void codeGen::host_code_gen ()
{
	std::string indent = "\t";
	
	host_code << R"(
int main(int argc, char **argv)
{
	puts("Initiating...");
	double (*h_in)[M][N] = (double (*)[M][N]) getRandom3DArray (L, M, N);
	double (*h_out)[M][N] = (double (*)[M][N]) getZero3DArray (L, M, N);

	unsigned int nbytes = sizeof(double) * L * M * N;
	double *in;
	cudaMalloc (&in, nbytes);
	check_error ("Failed to allocate device memory for in.\n");
	cudaMemcpy (in, h_in, nbytes, cudaMemcpyHostToDevice);
	double *out;
	cudaMalloc (&out, nbytes);
	check_error ("Failed to allocate device memory for out.\n");
	cudaMemcpy (out, h_out, nbytes, cudaMemcpyHostToDevice);
	
	dim3 block_config(Bx, By, 1);
	dim3 grid_config(ceil(N, Bx-Halo*2), ceil(M, By-Halo*2), ceil(L, Sn));

	puts("GPU computing...");)" << std::endl;
	host_code << indent << "for (int t = 0; t < Iterations; t += " << 2 * fs_stencil->get_step_num () <<") {" << std::endl;
	host_code << indent << indent << stencil_name << "<<<grid_config, block_config>>> (in, out);" << std::endl;
	host_code << indent << indent << stencil_name << "<<<grid_config, block_config>>> (out, in);" << std::endl;
	host_code << indent << R"(}
	cudaDeviceSynchronize();
	check_error ("Kernel error");
	puts("GPU finished computing.");
	)";
	if (check_correctness) {
		host_code << R"(
	// run the gold kernel and check error

	puts ("Checking error ...");

	double *g_in;
	cudaMalloc (&g_in, nbytes);
	check_error ("Failed to allocate device memory for g_in.\n");
	cudaMemcpy (g_in, h_in, nbytes, cudaMemcpyHostToDevice);
	double *g_out;
	cudaMalloc (&g_out, nbytes);
	check_error ("Failed to allocate device memory for g_out.\n");
	cudaMemcpy (g_out, h_out, nbytes, cudaMemcpyHostToDevice);
	
	dim3 block_config_1(8, 8, 8);
	dim3 grid_config_1(ceil(N, 8), ceil(M, 8), ceil(L, 8));
	)" << std::endl;

		host_code << indent << "for (int t = 0; t < Iterations; t += " << 2 * fs_stencil->get_step_num () <<") {" << std::endl;
		host_code << indent << indent << "gold_" << stencil_name << "<<<grid_config_1, block_config_1>>> (g_in, g_out);" << std::endl;
		host_code << indent << indent << "gold_" << stencil_name << "<<<grid_config_1, block_config_1>>> (g_out,g_in);" << std::endl;
		host_code << indent << R"(}
	cudaDeviceSynchronize();
	check_error ("Kernel(gold) error");
	
	double* h_g_out = (double*)h_in; 	// reuse the memory of the input array
	cudaMemcpy(h_out, in, nbytes, cudaMemcpyDeviceToHost);		
	cudaMemcpy(h_g_out, g_in, nbytes, cudaMemcpyDeviceToHost);		
	double error = checkError3D (M, N, (double*)h_out, (double*)h_g_out, Halo, L-Halo, Halo, M-Halo, Halo, N-Halo);
	printf("[Test] RMS Error: %e\n", error);
	)";

	}
	host_code << R"(
	free(h_in);
	free(h_out);
	cudaFree(in);
	cudaFree(out);
	return 0;
})";
}

std::string codeGen::gold_gpu_code_gen ()
{
	std::stringstream out_code;
	std::string indent ("\t");

	out_code << "__global__ void gold_" << stencil_name << " (double *d_in, double *d_out)\n";
	out_code << "{\n";
	out_code << indent << "int i = (int)(blockIdx.x) * (int)(blockDim.x) + (int)(threadIdx.x);\n";
	out_code << indent << "int j = (int)(blockIdx.y) * (int)(blockDim.y) + (int)(threadIdx.y);\n";
	out_code << indent << "int k = (int)(blockIdx.z) * (int)(blockDim.z) + (int)(threadIdx.z);\n";

	out_code << "\n";
	out_code << indent << "double (*in)[M][N] = (double (*)[M][N]) d_in;\n";
	out_code << indent << "double (*out)[M][N] = (double (*)[M][N]) d_out;\n";

	out_code << "\n";
	out_code << indent << "if (k >= Halo && k < L - Halo && j >= Halo && j < M - Halo && i >= Halo && i < N - Halo) {" << std:: endl;
	out_code << indent << indent << fs_stencil->gen_gold (); 
	out_code << indent << "}" << std::endl;
	out_code << "}" << std::endl << std::endl;
	
	return out_code.str();
}

void codeGen::gold_code_gen ()
{
	std::string indent = "\t";

	// header
	header_gen ();
	gold_code << header.str();

	// gpu
	gold_code << gold_gpu_code_gen ();

	// host
	gold_code << R"(
int main(int argc, char **argv)
{
	puts("Initiating...");
	double (*h_in)[M][N] = (double (*)[M][N]) getRandom3DArray (L, M, N);
	double (*h_out)[M][N] = (double (*)[M][N]) getZero3DArray (L, M, N);

	unsigned int nbytes = sizeof(double) * L * M * N;
	double *in;
	cudaMalloc (&in, nbytes);
	check_error ("Failed to allocate device memory for in.\n");
	cudaMemcpy (in, h_in, nbytes, cudaMemcpyHostToDevice);
	double *out;
	cudaMalloc (&out, nbytes);
	check_error ("Failed to allocate device memory for out.\n");
	cudaMemcpy (out, h_out, nbytes, cudaMemcpyHostToDevice);
	
	dim3 block_config(Bx, By, Bz);
	int gx = ceil(N, block_config.x);
	int gy = ceil(M, block_config.y);
	int gz = ceil(L, block_config.z);
	dim3 grid_config(gx, gy, gz);

	puts("GPU computing...");)" << std::endl;
	gold_code << indent << "for (int t = 0; t < Iterations; t += " << 2 * fs_stencil->get_step_num () <<") {" << std::endl;
	gold_code << indent << indent << "gold_" << stencil_name << "<<<grid_config, block_config>>> (in, out);" << std::endl;
	gold_code << indent << indent << "gold_" << stencil_name << "<<<grid_config, block_config>>> (out, in);" << std::endl;
	gold_code << indent << R"(}
	cudaDeviceSynchronize();
	check_error ("Kernel error");
	puts("GPU finished computing.");

	cudaMemcpy(h_out, in, nbytes, cudaMemcpyDeviceToHost);		
	free(h_in);
	free(h_out);
	cudaFree(in);
	cudaFree(out);
	return 0;
})";
	std::ofstream outfile;
	std::string out_name (stencil_name);
	out_name += "_gold.cu";
	outfile.open(out_name, std::ios::out | std::ios::trunc );
	outfile << gold_code.str();

	outfile.close();
}

#endif
