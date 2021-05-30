# ifndef __CODEGEN_HPP__
# define __CODEGEN_HPP__

# include <sstream>
# include <vector>
# include <tuple>
# include "drstencil.hpp"


class codeGen 
{
	private:
		std::stringstream header;
		std::stringstream gpu_code;
		std::stringstream host_code;
		std::stringstream gold_code;
		DRStencil* dr_stencil;
		int bx, by, sn;
		int L, M, N, iterations;
		int stream_unroll;
		bool bmerge_x, bmerge_y; // true for block merging, false for cyclic merging
		int mx, my;
        bool prefetch;
		std::string stencil_name;
		bool check_correctness;
    private:
		std::string for_declare (int&);
		void header_gen ();
		void gpu_code_gen ();
		void host_code_gen ();
		std::string gold_gpu_code_gen ();
	public:
		codeGen (DRStencil* fs, int bx_, int by_, int sn_, int s_unroll, 
				bool bmerge_x_, bool bmerge_y_, int mx_, int my_, bool pref,
				std::string stc_name, bool check) 
			: dr_stencil (fs), bx (bx_), by (by_), sn (sn_), stream_unroll (s_unroll), 
				bmerge_x (bmerge_x_), bmerge_y (bmerge_y_), mx (mx_), my (my_), prefetch (pref),
				stencil_name (stc_name), check_correctness (check)
			{}
		codeGen (DRStencil* fs, std::string stc_name) 
			: dr_stencil (fs), stencil_name (stc_name) 
			{}
		void output (const std::string &);
		void gold_code_gen ();
};

void codeGen::output (const std::string & out_name)
{
	// check whether the configuration is valid
	int halo2 = dr_stencil->get_order () * 2;
	if ((halo2 >= bx * mx && dr_stencil->forward_i_avilable ()) ||
		(halo2 >= by * my && dr_stencil->forward_j_avilable ())) {
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
	header << "# include <sys/time.h>" << std::endl;
	header << "\n#define max(x,y) ((x) > (y) ? (x) : (y)) \n";
	header << "#define min(x,y) ((x) < (y) ? (x) : (y)) \n";
	header << "#define ceil(a,b) ((a) % (b) == 0 ? (a) / (b) : ((a) / (b) + 1)) \n";
	header << std::endl;
	dr_stencil->get_problem_size (L, M, N, iterations);
	header << "#define L " << L << std::endl;
	header << "#define M " << M << std::endl;
	header << "#define N " << N << std::endl;
	header << "#define Iterations " << iterations << std::endl;
	header << std::endl;
	header << "#define Range " << dr_stencil->get_high_k () - dr_stencil->get_low_k () + 1 << std::endl;
	header << "#define Halo " << dr_stencil->get_order () << std::endl;
	header << "#define Dist " << dr_stencil->get_distance () << std::endl;
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

	header << "double get_time() {\n";
	header << indent << "struct timeval tv;\n";
	header << indent << "double t;\n";
	header << indent << "gettimeofday(&tv, (struct timezone *)0);\n";
	header << indent << "t = tv.tv_sec + (double)tv.tv_usec * 1e-6;\n";
	header << indent << "return t;\n";
	header << "}\n";
}

std::string codeGen::for_declare (int &indent_cnt)
{
    std::stringstream code;
	if (my > 1) {
		code << std::string(indent_cnt, '\t') << "#pragma unroll " << my << std::endl;
		if (bmerge_y) 
			code << std::string(indent_cnt, '\t') << "for (int mj = 0; mj < "
						<< my << "; mj ++) {" << std::endl;
		else // cyclic merging
			code << std::string(indent_cnt, '\t') << "for (int mj = 0, mj_1 = 0; mj_1 < "
						<< my << "; mj_1 ++, mj += blockDim.y) {" << std::endl;
		indent_cnt ++;
	}
	if (mx > 1) {
		code << std::string(indent_cnt, '\t') << "#pragma unroll " << mx << std::endl;
		if (bmerge_x) 
			code << std::string(indent_cnt, '\t') << "for (int mi = 0; mi < "
						<< mx << "; mi ++) {" << std::endl;
		else // cyclic merging
			code << std::string(indent_cnt, '\t') << "for (int mi = 0, mi_1 = 0; mi_1 < "
						<< mx << "; mi_1 ++, mi += blockDim.x) {" << std::endl;
		indent_cnt ++;
	}
    code << std::endl;
    return code.str();
}

void codeGen::gpu_code_gen ()
{
	std::string indent = "\t";
    int indent_cnt = 1;

	gpu_code << "__global__ void dr_" << stencil_name << " (double *d_in, double *d_out)\n";
	gpu_code << "{" << std::endl;
	// index i
	gpu_code << std::string(indent_cnt, '\t') << "int i = " << (bmerge_x && mx > 1 ? std::to_string(mx) + "*" : "" ) 
			<< "(int)(threadIdx.x);" << std::endl;
	gpu_code << std::string(indent_cnt, '\t') << "int i0 = (int)(blockIdx.x) * (" 
			<< (mx > 1 ? std::to_string(mx) + "*" : "") 
			<< "(int)blockDim.x-Halo*2) + i;" << std::endl;
	// index j
	gpu_code << std::string(indent_cnt, '\t') << "int j = " << (bmerge_y && my > 1 ? std::to_string(my) + "*" : "" ) 
			<< "(int)(threadIdx.y);" << std::endl;
	gpu_code << std::string(indent_cnt, '\t') << "int j0 = (int)(blockIdx.y) * (" 
			<< (my > 1 ? std::to_string(my) + "*" : "") 
			<< "(int)blockDim.y-Halo*2) + j;" << std::endl;
	gpu_code << R"(
	int k = (int)(blockIdx.z) * Sn;
	int k_ed = min(L - Halo, k + Sn + Halo);

	double (*in)[M][N] = (double (*)[M][N]) d_in;
	double (*out)[M][N] = (double (*)[M][N]) d_out;
	)" << std::endl;

	// variables declations
	gpu_code << std::string(indent_cnt, '\t') << "double __shared__ in_shm[Range][" 
			<< (my > 1 ? std::to_string(my) + "*" : "") << "By]["
			<< (mx > 1 ? std::to_string(mx) + "*" : "") << "Bx];" << std::endl;
	if (prefetch)
        gpu_code << std::string(indent_cnt, '\t') << "double pre" << (my > 1 ? "[" + std::to_string(my) + "]" : "")
	    		<< (mx > 1 ? "[" + std::to_string(mx) + "]" : "") << ";" << std::endl;
	//gpu_code << std::string(indent_cnt, '\t') << "double __shared__ out_shm["
	//		<< (my > 1 ? std::to_string(my) + "*" : "") << "By-Halo*2]["
	//		<< (mx > 1 ? std::to_string(mx) + "*" : "") << "Bx-Halo*2];" << std::endl;
	//gpu_code << std::string(indent_cnt, '\t') << "double forward_k[Dist]" 
    //      << (my > 1 ? "[" + std::to_string(my) + "]" : "")
	//		<< (mx > 1 ? "[" + std::to_string(mx) + "]" : "") << ";" << std::endl;

	int low_k = dr_stencil->get_low_k ();
	int high_k = dr_stencil->get_high_k ();
	int range = high_k - low_k + 1;
	//gpu_code << std::string(indent_cnt, '\t') << "int k0";
	for (int i = 0; i < range; i ++) {
	    gpu_code << std::string(indent_cnt, '\t') << "int k" << i << " = " << i << ";" << std::endl;
	}
	gpu_code << std::endl;;
   
    // j_ok, i_ok
    gpu_code << std::string(indent_cnt, '\t') << "bool j_ok"
            << (my > 1 ? "[" + std::to_string(my) + "]" : "") << ", i_ok" 
            << (mx > 1 ? "[" + std::to_string(mx) + "]" : "") << ";" << std::endl;

    gpu_code << std::string(indent_cnt, '\t') << "j_ok" << (my > 1 ? "[0]" : "")
            << " = (j >= Halo && j < By" << (my > 1 ? " * " + std::to_string(my) : "")
            << "- Halo && j0 < M - Halo);" << std::endl;
    for (int i = 1; i < my; i ++) {
        gpu_code << std::string(indent_cnt, '\t') << "j_ok[" << i << "] = (j + " 
                << i << (bmerge_y ? "" : " * By") << " >= Halo && j + "
                << i << (bmerge_y ? "" : " * By") << " < By * " << my << " - Halo && j0 + " 
                << i << (bmerge_y ? "" : " * By") << " < M - Halo);" << std::endl;
    }
	//gpu_code << std::endl;

    gpu_code << std::string(indent_cnt, '\t') << "i_ok" << (mx > 1 ? "[0]" : "")
            << " = (i >= Halo && i < Bx" << (mx > 1 ? " * " + std::to_string(mx) : "")
            << "- Halo && i0 < N - Halo);" << std::endl;
    for (int i = 1; i < mx; i ++) {
        gpu_code << std::string(indent_cnt, '\t') << "i_ok[" << i << "] = (i + " 
                << i << (bmerge_x ? "" : " * Bx") << " >= Halo && i + "
                << i << (bmerge_x ? "" : " * Bx") << " < Bx * " << mx << " - Halo && i0 + " 
                << i << (bmerge_x ? "" : " * Bx") << " < N - Halo);" << std::endl;
    }
	gpu_code << std::endl;

	gpu_code << std::string(indent_cnt, '\t') << "int k_st = k + Halo;" << std::endl;
	if (dr_stencil->get_distance () < high_k) {
		gpu_code << std::string(indent_cnt, '\t') << "k = k + Halo - Dist;" << std::endl;
	}
	gpu_code << std::endl;

	gpu_code << std::string(indent_cnt, '\t') << "if (k_st >= k_ed || j0 >= M || i0 >= N) return;" << std::endl;

	gpu_code << std::string(indent_cnt, '\t') << "// Initial loads" << std::endl;

    std::string index_j = "j" + std::string(my > 1 ? "+mj" : "");
    std::string index_j0 = "j0" + std::string(my > 1 ? "+mj" : "");
    std::string index_i = "i" + std::string(mx > 1 ? "+mi" : "");
    std::string index_i0 = "i0" + std::string(mx > 1 ? "+mi" : "");


	// load data
    if (mx > 1 || my > 1) gpu_code << for_declare (indent_cnt);

    if (mx > 1 && my > 1)
        gpu_code << std::string(indent_cnt ++, '\t') << "if (j0 + mj < M && i0 + mi < N) {" << std::endl;
    else if (mx > 1)
        gpu_code << std::string(indent_cnt ++, '\t') << "if (i0 + mi < N) {" << std::endl;
    else if (my > 1)
        gpu_code << std::string(indent_cnt ++, '\t') << "if (j0 + mj < M) {" << std::endl;

	for (int i = low_k; i < high_k; i ++) {
        if (i < 0)
		    gpu_code << std::string(indent_cnt, '\t') << "if (k" << i << " >= 0) in_shm[k" 
                << i - low_k << "][" << index_j << "][" << index_i << "] = in[k" 
                << i << "][" << index_j0 << "][" << index_i0 << "];" << std::endl;
		else 
            gpu_code << std::string(indent_cnt, '\t') << "in_shm[k" << i - low_k << "][" << index_j << "]["
				<< index_i << "] = in[k+" << i << "][" << index_j0 << "][" << index_i0 << "];" << std::endl;
	}
    if (prefetch)
            gpu_code << std::string(indent_cnt, '\t') << "pre" 
                    << (my > 1 ? (bmerge_y ? "[mj]" : "[mj_1]") : "")
                    << (mx > 1 ? (bmerge_x ? "[mi]" : "[mi_1]") : "") 
				    << " = in[k+" << high_k << "][" << index_j0 << "][" << index_i0 << "];" << std::endl;

        

    if (mx > 1 || my > 1) gpu_code << std::string(-- indent_cnt, '\t') << "}" << std::endl; 
	if (mx > 1) gpu_code << std::string(-- indent_cnt, '\t') << "}" << std::endl; 
	if (my > 1) gpu_code << std::string(-- indent_cnt, '\t') << "}" << std::endl;
	gpu_code << std::endl;

	// forward k
	gpu_code << std::string(indent_cnt, '\t') << "#pragma unroll " << dr_stencil->get_distance () << std::endl;
	gpu_code << std::string(indent_cnt ++, '\t') << "for (int ki = 0; ki < Dist; ki ++, k ++) {" << std::endl;

    if (mx > 1 || my > 1) gpu_code << for_declare (indent_cnt);
    if (prefetch)
        gpu_code << std::string(indent_cnt, '\t') << "in_shm[k" << range - 1 
                << "][" << index_j << "][" << index_i <<"] = pre" 
                << (my > 1 ? (bmerge_y ? "[mj]" : "[mj_1]") : "")
                << (mx > 1 ? (bmerge_x ? "[mi]" : "[mi_1]") : "") << ";" << std::endl;
    else {
        if (mx > 1 && my > 1)
            gpu_code << std::string(indent_cnt ++, '\t') << "if (j0 + mj < M && i0 + mi < N) {" << std::endl;
        else if (mx > 1)
            gpu_code << std::string(indent_cnt ++, '\t') << "if (i0 + mi < N) {" << std::endl;
        else if (my > 1)
            gpu_code << std::string(indent_cnt ++, '\t') << "if (j0 + mj < M) {" << std::endl;

        gpu_code << std::string(indent_cnt, '\t') << "in_shm[k" << range - 1
                << "][" << index_j << "][" << index_i <<"] = in[k+" 
				<< high_k << "][" << index_j0 << "][" << index_i0 << "];" << std::endl;
	    if (mx > 1 || my > 1) gpu_code << std::string(-- indent_cnt, '\t') << "}" << std::endl; 
    }
	if (mx > 1) gpu_code << std::string(-- indent_cnt, '\t') << "}" << std::endl; 
	if (my > 1) gpu_code << std::string(-- indent_cnt, '\t') << "}" << std::endl;

	gpu_code << std::string(indent_cnt, '\t') << "__syncthreads ();" << std::endl;
    if (prefetch) {
        gpu_code << std::string(indent_cnt, '\t') << "pre"
                << (my > 1 ? "[0]" : "") << (mx > 1 ? "[0]" : "") 
                << " = in[k+" << high_k + 1 << "][j0][i0];" << std::endl;
        for (int j = 0; j < my; j ++)
        for (int i = 0; i < mx; i ++) {
            if (i > 0 || j > 0)
            gpu_code << std::string(indent_cnt, '\t') << "if ("
                    << (j ? "j0 + " + std::to_string(j) + (bmerge_y && j ? "" : " * By") + " < M" : "") 
                    << (i > 0 && j > 0 ? " && " : "")
                    << (i ? "i0 + " + std::to_string(i) + (bmerge_x && i ? "" : " * Bx") + " < N" : "") << ") pre"
                    << (my > 1 ? "[" + std::to_string(j) + "]" : "") 
                    << (mx > 1 ? "[" + std::to_string(i) + "]" : "")
	                << " = in[k+" << high_k + 1 << "][j0+" 
                    << j << (bmerge_y && j ? "" : " * By") << "][i0+" 
                    << i << (bmerge_x && i ? "" : " * Bx") << "];" << std::endl;
        }
    }
    
    if (mx > 1 || my > 1) gpu_code << for_declare (indent_cnt);
	
	gpu_code << std::string(indent_cnt, '\t') << "// forward k" << std::endl;
	gpu_code << std::string(indent_cnt ++, '\t') << "if (k < k_ed - Dist && j_ok"
            << (my > 1 ? (bmerge_y ? "[mj]" : "[mj_1]") : "") << " && i_ok"
            << (mx > 1 ? (bmerge_x ? "[mi]" : "[mi_1]") : "") << ") {" << std::endl;

	gpu_code << std::string(indent_cnt, '\t') << "out[k+Dist][" << index_j0 << "][" << index_i0
			<< "] = " << dr_stencil->gen_forward_k (indent_cnt, my>1, mx>1) << ";" << std::endl;

	gpu_code << std::string(-- indent_cnt, '\t') << "}" << std::endl;
	if (mx > 1) gpu_code << std::string(-- indent_cnt, '\t') << "}" << std::endl; 
	if (my > 1) gpu_code << std::string(-- indent_cnt, '\t') << "}" << std::endl;
    gpu_code << std::string(indent_cnt, '\t') << "__syncthreads ();" << std::endl;
	
    gpu_code << std::string(indent_cnt, '\t') << "int temp = k0;" << std::endl;
    for (int i = 0; i < range - 1; i ++)
        gpu_code << std::string(indent_cnt, '\t') << "k" << i << " = k" << i + 1 << ";" << std::endl; 
    gpu_code << std::string(indent_cnt, '\t') << "k" << range - 1 << " = temp" << ";" << std::endl; 
    gpu_code << std::string(-- indent_cnt, '\t') << "}" << std::endl;
    gpu_code << std::endl;


	gpu_code << std::string(indent_cnt, '\t') << "#pragma unroll " << stream_unroll << std::endl;
	gpu_code << std::string(indent_cnt ++, '\t') << "for (; k < k_ed - Dist; k ++) {" << std::endl;

    if (mx > 1 || my > 1) gpu_code << for_declare (indent_cnt);

    if (prefetch)
        gpu_code << std::string(indent_cnt, '\t') << "in_shm[k" << range - 1 
                << "][" << index_j << "][" << index_i <<"] = pre" 
                << (my > 1 ? (bmerge_y ? "[mj]" : "[mj_1]") : "")
                << (mx > 1 ? (bmerge_x ? "[mi]" : "[mi_1]") : "") << ";" << std::endl;
    else {
        if (mx > 1 && my > 1)
            gpu_code << std::string(indent_cnt ++, '\t') << "if (j0 + mj < M && i0 + mi < N) {" << std::endl;
        else if (mx > 1)
            gpu_code << std::string(indent_cnt ++, '\t') << "if (i0 + mi < N) {" << std::endl;
        else if (my > 1)
            gpu_code << std::string(indent_cnt ++, '\t') << "if (j0 + mj < M) {" << std::endl;

        gpu_code << std::string(indent_cnt, '\t') << "in_shm[k" << range - 1
                << "][" << index_j << "][" << index_i <<"] = in[k+" 
				<< high_k << "][" << index_j0 << "][" << index_i0 << "];" << std::endl;
	    if (mx > 1 || my > 1) gpu_code << std::string(-- indent_cnt, '\t') << "}" << std::endl; 
    }
	if (mx > 1) gpu_code << std::string(-- indent_cnt, '\t') << "}" << std::endl; 
	if (my > 1) gpu_code << std::string(-- indent_cnt, '\t') << "}" << std::endl;
	gpu_code << std::string(indent_cnt, '\t') << "__syncthreads ();" << std::endl;

    if (prefetch) {
        gpu_code << std::string(indent_cnt, '\t') << "// prefetching input data for next iteration" << std::endl;
        gpu_code << std::string(indent_cnt, '\t') << "pre"
                << (my > 1 ? "[0]" : "") << (mx > 1 ? "[0]" : "") 
                << " = in[k+" << high_k + 1 << "][j0][i0];" << std::endl;
        for (int j = 0; j < my; j ++)
        for (int i = 0; i < mx; i ++) {
            if (i > 0 || j > 0)
            gpu_code << std::string(indent_cnt, '\t') << "if ("
                    << (j ? "j0 + " + std::to_string(j) + (bmerge_y && j ? "" : " * By") + " < M" : "") 
                    << (i > 0 && j > 0 ? " && " : "")
                    << (i ? "i0 + " + std::to_string(i) + (bmerge_x && i ? "" : " * Bx") + " < N" : "") << ") pre"
                    << (my > 1 ? "[" + std::to_string(j) + "]" : "") 
                    << (mx > 1 ? "[" + std::to_string(i) + "]" : "")
	                << " = in[k+" << high_k + 1 << "][j0+" 
                    << j << (bmerge_y && j ? "" : "*By") << "][i0+" 
                    << i << (bmerge_x && i ? "" : "*Bx") << "];" << std::endl;
        }
    }
    if (mx > 1 || my > 1) gpu_code << for_declare (indent_cnt);
	
	gpu_code << std::string(indent_cnt ++, '\t') << "if (j_ok"
            << (my > 1 ? (bmerge_y ? "[mj]" : "[mj_1]") : "") << " && i_ok"
            << (mx > 1 ? (bmerge_x ? "[mi]" : "[mi_1]") : "") << ") {" << std::endl;

    // forward k
	gpu_code << std::string(indent_cnt, '\t') << "// forward k" << std::endl;
	gpu_code << std::string(indent_cnt, '\t') << "out[k+Dist][" << index_j0 << "][" << index_i0
			<< "] = " << dr_stencil->gen_forward_k (indent_cnt, my>1, mx>1) << ";" << std::endl
            << std::endl;

    // backward
	gpu_code << std::string(indent_cnt, '\t') << "// backward" << std::endl;
	gpu_code << std::string(indent_cnt, '\t') << "atomicAdd(&out[k]["
			<< index_j0 << "][" << index_i0 << "], "
			<< dr_stencil->gen_backward (indent_cnt, my > 1, mx > 1) << ");" << std::endl;
	gpu_code << std::string(-- indent_cnt, '\t') << "}" << std::endl << std::endl;

	// forward j
	if (dr_stencil->forward_j_avilable ()) {
		gpu_code << std::string(indent_cnt, '\t') << "// forward j" << std::endl;
		gpu_code << std::string(indent_cnt ++, '\t') << "if (" << index_j << " < By*" << my << "-Halo-Dist && "
                << index_j0 << " < M-Halo-Dist && " 
                << (dr_stencil->get_order() == dr_stencil->get_distance() ? "" 
                    : index_j + " >= Halo-Dist && ")
                << "i_ok" << (mx > 1 ? (bmerge_x ? "[mi]" : "[mi_1]") : "") << ") {" << std::endl;
		gpu_code << std::string(indent_cnt, '\t') << "atomicAdd(&out[k]["
				<< index_j0 << "+Dist][" << index_i0 << "], "
				<< dr_stencil->gen_forward_j (indent_cnt, my > 1, mx > 1) << ");" << std::endl;
	    gpu_code << std::string(-- indent_cnt, '\t') << "}" << std::endl << std::endl;
	}

	// forward i
	if (dr_stencil->forward_i_avilable ()) {
		gpu_code << std::string(indent_cnt, '\t') << "// forward i" << std::endl;
		gpu_code << std::string(indent_cnt ++, '\t') << "if (" << index_i << " < Bx*" << mx << "-Halo-Dist && "
                << index_i0 << " < N-Halo-Dist && " 
                << (dr_stencil->get_order() == dr_stencil->get_distance() ? "" 
                    : index_i + " >= Halo-Dist && ")
                << "j_ok" << (my > 1 ? (bmerge_y ? "[mj]" : "[mj_1]") : "") << ") {" << std::endl;
		gpu_code << std::string(indent_cnt, '\t') << "atomicAdd(&out[k]["
				<< index_j0 << "][" << index_i0 << "+Dist], " 
				<< dr_stencil->gen_forward_i (indent_cnt, my > 1, mx > 1) << ");" << std::endl;
		gpu_code << std::string(indent_cnt --, '\t') << "}" << std::endl << std::endl;
	}

	if (mx > 1) gpu_code << std::string(-- indent_cnt, '\t') << "}" << std::endl; 
	if (my > 1) gpu_code << std::string(-- indent_cnt, '\t') << "}" << std::endl;

	gpu_code << std::string(indent_cnt, '\t') << "__syncthreads ();" << std::endl;

    gpu_code << std::string(indent_cnt, '\t') << "int temp = k0;" << std::endl;
    for (int i = 0; i < range - 1; i ++)
        gpu_code << std::string(indent_cnt, '\t') << "k" << i << " = k" << i + 1 << ";" << std::endl; 
    gpu_code << std::string(indent_cnt, '\t') << "k" << range - 1 << " = temp" << ";" << std::endl; 
	gpu_code << std::string(-- indent_cnt, '\t') << "}" << std::endl;
    gpu_code << std::endl;


	gpu_code << std::string(indent_cnt, '\t') << "#pragma unroll " << dr_stencil->get_distance() << std::endl;
	gpu_code << std::string(indent_cnt ++, '\t') << "for (; k < k_ed; k ++) {" << std::endl;

    if (mx > 1 || my > 1) gpu_code << for_declare (indent_cnt);

    if (prefetch)
        gpu_code << std::string(indent_cnt, '\t') << "in_shm[k" << range - 1 
                << "][" << index_j << "][" << index_i <<"] = pre" 
                << (my > 1 ? (bmerge_y ? "[mj]" : "[mj_1]") : "")
                << (mx > 1 ? (bmerge_x ? "[mi]" : "[mi_1]") : "") << ";" << std::endl;
    else {
        if (mx > 1 && my > 1)
            gpu_code << std::string(indent_cnt ++, '\t') << "if (j0 + mj < M && i0 + mi < N) {" << std::endl;
        else if (mx > 1)
            gpu_code << std::string(indent_cnt ++, '\t') << "if (i0 + mi < N) {" << std::endl;
        else if (my > 1)
            gpu_code << std::string(indent_cnt ++, '\t') << "if (j0 + mj < M) {" << std::endl;

        gpu_code << std::string(indent_cnt, '\t') << "in_shm[k" << range - 1
                << "][" << index_j << "][" << index_i <<"] = in[k+" 
				<< high_k << "][" << index_j0 << "][" << index_i0 << "];" << std::endl;
	    if (mx > 1 || my > 1) gpu_code << std::string(-- indent_cnt, '\t') << "}" << std::endl; 
    }
                
	if (mx > 1) gpu_code << std::string(-- indent_cnt, '\t') << "}" << std::endl; 
	if (my > 1) gpu_code << std::string(-- indent_cnt, '\t') << "}" << std::endl;
	gpu_code << std::string(indent_cnt, '\t') << "__syncthreads ();" << std::endl;

    if (prefetch) {
	    gpu_code << std::string(indent_cnt ++, '\t') << "if (k+1 < k_ed) {" << std::endl;
        gpu_code << std::string(indent_cnt, '\t') << "pre"
                << (my > 1 ? "[0]" : "") << (mx > 1 ? "[0]" : "") 
                << " = in[k+" << high_k + 1 << "][j0][i0];" << std::endl;
        for (int j = 0; j < my; j ++)
        for (int i = 0; i < mx; i ++) {
            if (i > 0 || j > 0)
            gpu_code << std::string(indent_cnt, '\t') << "if ("
                    << (j ? "j0 + " + std::to_string(j) + (bmerge_y && j ? "" : " * By") + " < M" : "") 
                    << (i > 0 && j > 0 ? " && " : "")
                    << (i ? "i0 + " + std::to_string(i) + (bmerge_x && i ? "" : " * Bx") + " < N" : "") << ") pre"
                    << (my > 1 ? "[" + std::to_string(j) + "]" : "") 
                    << (mx > 1 ? "[" + std::to_string(i) + "]" : "")
	                << " = in[k+" << high_k + 1 << "][j0+" 
                    << j << (bmerge_y && j ? "" : " * By") << "][i0+" 
                    << i << (bmerge_x && i ? "" : " * Bx") << "];" << std::endl;
        }
	    gpu_code << std::string(-- indent_cnt, '\t') << "}" << std::endl;
    }
    if (mx > 1 || my > 1) gpu_code << for_declare (indent_cnt);
	
    // backward
	gpu_code << std::string(indent_cnt ++, '\t') << "if (j_ok"
            << (my > 1 ? (bmerge_y ? "[mj]" : "[mj_1]") : "") << " && i_ok"
            << (mx > 1 ? (bmerge_x ? "[mi]" : "[mi_1]") : "") << ") {" << std::endl;
	gpu_code << std::string(indent_cnt, '\t') << "// backward" << std::endl;
	gpu_code << std::string(indent_cnt, '\t') << "atomicAdd(&out[k]["
			<< index_j0 << "][" << index_i0 << "], "
			<< dr_stencil->gen_backward (indent_cnt, my > 1, mx > 1) << ");" << std::endl;
	gpu_code << std::string(-- indent_cnt, '\t') << "}" << std::endl << std::endl;

	// forward j
	if (dr_stencil->forward_j_avilable ()) {
		gpu_code << std::string(indent_cnt, '\t') << "// forward j" << std::endl;
		gpu_code << std::string(indent_cnt ++, '\t') << "if (" << index_j << " < By*" << my << "-Halo-Dist && "
                << index_j0 << " < M-Halo-Dist && " 
                << (dr_stencil->get_order() == dr_stencil->get_distance() ? "" 
                    : index_j + " >= Halo-Dist && ")
                << "i_ok" << (mx > 1 ? (bmerge_x ? "[mi]" : "[mi_1]") : "") << ") {" << std::endl;
		gpu_code << std::string(indent_cnt, '\t') << "atomicAdd(&out[k]["
				<< index_j0 << "+Dist][" << index_i0 << "], "
				<< dr_stencil->gen_forward_j (indent_cnt, my > 1, mx > 1) << ");" << std::endl;
	    gpu_code << std::string(-- indent_cnt, '\t') << "}" << std::endl << std::endl;
	}

	// forward i
	if (dr_stencil->forward_i_avilable ()) {
		gpu_code << std::string(indent_cnt, '\t') << "// forward i" << std::endl;
		gpu_code << std::string(indent_cnt ++, '\t') << "if (" << index_i << " < Bx*" << mx << "-Halo-Dist && "
                << index_i0 << " < N - Halo - Dist && " 
                << (dr_stencil->get_order() == dr_stencil->get_distance() ? "" 
                    : index_i + " >= Halo-Dist && ")
                << "j_ok" << (my > 1 ? (bmerge_y ? "[mj]" : "[mj_1]") : "") << ") {" << std::endl;
		gpu_code << std::string(indent_cnt, '\t') << "atomicAdd(&out[k]["
				<< index_j0 << "][" << index_i0 << "+Dist], " 
				<< dr_stencil->gen_forward_i (indent_cnt, my > 1, mx > 1) << ");" << std::endl;
		gpu_code << std::string(indent_cnt --, '\t') << "}" << std::endl;
	}

	if (mx > 1) gpu_code << std::string(-- indent_cnt, '\t') << "}" << std::endl; 
	if (my > 1) gpu_code << std::string(-- indent_cnt, '\t') << "}" << std::endl;

	gpu_code << std::string(indent_cnt, '\t') << "__syncthreads ();" << std::endl;

    gpu_code << std::string(indent_cnt, '\t') << "int temp = k0;" << std::endl;
    for (int i = 0; i < range - 1; i ++)
        gpu_code << std::string(indent_cnt, '\t') << "k" << i << " = k" << i + 1 << ";" << std::endl; 
    gpu_code << std::string(indent_cnt, '\t') << "k" << range - 1 << " = temp" << ";" << std::endl; 
	gpu_code << std::string(-- indent_cnt, '\t') << "}" << std::endl;


	gpu_code << "}" << std::endl;
}


void codeGen::host_code_gen ()
{
	std::string indent = "\t";
	
	host_code << R"(
int main(int argc, char **argv)
{
	puts("Initiating ...");
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
	dim3 grid_config(ceil(N, )";
	host_code << (mx > 1 ? std::to_string(mx) + "*" : "") << "Bx-Halo*2), ceil(M, ";
	host_code << (my > 1 ? std::to_string(my) + "*" : "") << R"(By-Halo*2), ceil(L, Sn));

	puts("GPU computing ...");

	// warm up
	for (int i = 0; i < 10; i ++) {)" << std::endl;	
	host_code << indent << indent << "dr_" << stencil_name << "<<<grid_config, block_config>>> (in, out);" << std::endl;
	host_code << indent << "}" << std::endl << std::endl;
	//host_code << indent << "cudaEventRecord (startTime, 0);" << std::endl;
	host_code << indent << "double startTime = get_time();" << std::endl;
	host_code << indent << "for (int t = 0; t < Iterations; t += " << 2 * dr_stencil->get_step_num () <<") {" << std::endl;
	host_code << indent << indent << "dr_" << stencil_name << "<<<grid_config, block_config>>> (in, out);" << std::endl;
	host_code << indent << indent << "dr_" << stencil_name << "<<<grid_config, block_config>>> (out, in);" << std::endl;
	host_code << indent << R"(}
	cudaDeviceSynchronize();
	double endTime = get_time();
	check_error ("Kernel error");
	puts("GPU finished computing.");
	printf("GPU computation time: %f ms\n", 1000*(endTime - startTime));
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

	for (int t = 0; t < Iterations; t += )" << 2 * dr_stencil->get_step_num () <<") {" << std::endl;

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

	cudaFree(g_in);
	cudaFree(g_out);
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

	out_code << std::endl;
	out_code << "__global__ void gold_" << stencil_name << " (double *d_in, double *d_out)\n";
	out_code << "{\n";
	out_code << indent << "int i = (int)(blockIdx.x) * (int)(blockDim.x) + (int)(threadIdx.x);\n";
	out_code << indent << "int j = (int)(blockIdx.y) * (int)(blockDim.y) + (int)(threadIdx.y);\n";
	out_code << indent << "int k = (int)(blockIdx.z) * (int)(blockDim.z) + (int)(threadIdx.z);\n";

	out_code << "\n";
	out_code << indent << "double (*in)[M][N] = (double (*)[M][N]) d_in;\n";
	out_code << indent << "double (*out)[M][N] = (double (*)[M][N]) d_out;\n";

	out_code << "\n";
	out_code << indent << "if (k >= Halo && k < L - Halo && j >= Halo && j < M - Halo && i >= Halo && i < N - Halo) {" << std::endl;
	out_code << indent << indent << dr_stencil->gen_gold (); 
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
	gold_code << indent << "for (int t = 0; t < Iterations; t += " << 2 * dr_stencil->get_step_num () <<") {" << std::endl;
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
