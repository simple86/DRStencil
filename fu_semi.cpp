# include <iostream>
# include <string>
# include "autoFS.hpp"
# include "autoFS_2d.hpp"
# include "codegen.hpp"
# include "codegen_2d.hpp"
//# include "codegen_2d_no_stream.hpp"
using namespace std;


int main(int argc, char** argv)
{
    string outfile ("-o");
    string out_name ("out.cu");

    string _3d ("--3d");
    bool is3d = false;

    string step ("--step");
    int step_num = 1;

    string dist ("--dist");
    int distance = 0;

    string streaming_ ("--streaming");
    bool streaming = false;

    string block_x ("--bx");
    int bx = 16;
    string block_y ("--by");
    int by = 16;
    string stream_length ("--sn");
    int sn = 16;
    
    string stream_unroll ("--stream-unroll");
    int s_unroll = 4;
    
    string block_merge_x ("--block-merge-x");
    int bmx = 1;
    string block_merge_y ("--block-merge-y");
    int bmy = 1;
    string cyclic_merge_x ("--cyclic-merge-x");
    int cmx = 1;
    string cyclic_merge_y ("--cyclic-merge-y");
    int cmy = 1;

    string prefetch ("--prefetch");
    bool pref = false;

    string merge_forward ("--merge-forward");
    int merge_f = 5;

    string gold ("--gold");
    bool isGold = false;

    string check ("--check");
    bool check_correctness = false;

    string help ("--help");
    string _h ("-h");

    if (argc < 2) {
        cout << "Please specify the .stc file." << endl;
        return 0;
    }
    if (help.compare (argv[1]) == 0 || _h.compare (argv[1]) == 0) {
        cout << R"(
    Generate data-reusing stencil.

    Usage: fsstencil [options] <input_stcfile>
Options:

-o <file>               Specify the name of output cuda file. 
                        (out.cu by default)

--is3d                  Choose 3D mode.

--step <num>            Specify the number of time steps to fuse the stencil.
                        (step_num = 1 by default)

--dist <num>            Specify the number of the distance between points for semi-stencil.

--streaming             Not using streaming optimization.

--bx <num>              Specify the block size bx.
                        (bx = 16 by default)

--by <num>              Specify the block size by.
                        (by = 16 by default)

--sn <num>              Specify the length of stream block sn.
                        (sn = 16 by default)

--stream-unroll <num>   Specify the unroll factor of the streaming for loop.
                        (stream_unroll = 4 by default)

--block-merge-x <num>   Specify the number of points for block merging along dimension x.

--block-merge-y <num>   Specify the number of points for block merging along dimension y.

--cyclic-merge-x <num>  Specify the number of points for cyclic merging along dimension x.

--cyclic-merge-y <num>  Specify the number of points for cyclic merging along dimension y.

--prefetch              Prefetch input data to hide the tranfer latency.

--merge-forward <num>   Specify the threshold for whether to merge the forward_j or forward_i into backward,
                        since the overhead may offset the gain from partitation.
                        (merge_forward = 5 by default)

--check                 Check the correctness of the generated code.

--gold                  Generate the naive code.

--help  (-h)            Print this help information on this tool.
        )" << endl;
        return 0;
    }
    for (int i = 1; i < argc - 1; i ++) {
        if (outfile.compare (argv[i]) == 0) {
            if (i != argc - 2)
                out_name = argv[++i];
        }
        else if (_3d.compare (argv[i]) == 0) {
            is3d = true;
        }
        else if (step.compare (argv[i]) == 0) {
            if (i != argc - 2)
                step_num = atoi (argv[++i]);
            else {
                cout << "Illegal input." << endl;
                exit(-1);
            }
        }
        else if (dist.compare (argv[i]) == 0) {
            if (i != argc - 2)
                distance = atoi (argv[++i]);
            else {
                cout << "Illegal input." << endl;
                exit(-1);
            }
        }
        else if (streaming_.compare (argv[i]) == 0) {
            streaming = true;
        }
        else if (block_x.compare (argv[i]) == 0) {
            if (i != argc - 2)
                bx = atoi (argv[++i]);
            else {
                cout << "Illegal input." << endl;
                exit(-1);
            }
        }
        else if (block_y.compare (argv[i]) == 0) {
            if (i != argc - 2)
                by = atoi (argv[++i]);
            else {
                cout << "Illegal input." << endl;
                exit(-1);
            }
        }
        else if (stream_length.compare (argv[i]) == 0) {
            if (i != argc - 2)
                sn = atoi (argv[++i]);
            else {
                cout << "Illegal input." << endl;
                exit(-1);
            }
        }
        else if (block_merge_x.compare (argv[i]) == 0) {
            if (i != argc - 2)
                bmx = atoi (argv[++i]);
            else {
                cout << "Illegal input." << endl;
                exit(-1);
            }
        }
        else if (block_merge_y.compare (argv[i]) == 0) {
            if (i != argc - 2)
                bmy = atoi (argv[++i]);
            else {
                cout << "Illegal input." << endl;
                exit(-1);
            }
        }
        else if (cyclic_merge_x.compare (argv[i]) == 0) {
            if (i != argc - 2)
                cmx = atoi (argv[++i]);
            else {
                cout << "Illegal input." << endl;
                exit(-1);
            }
        }
        else if (cyclic_merge_y.compare (argv[i]) == 0) {
            if (i != argc - 2)
                cmy = atoi (argv[++i]);
            else {
                cout << "Illegal input." << endl;
                exit(-1);
            }
        }
        else if (stream_unroll.compare (argv[i]) == 0) {
            if (i != argc - 2)
                s_unroll = atoi (argv[++i]);
            else {
                cout << "Illegal input." << endl;
                exit(-1);
            }
        }
        else if (prefetch.compare (argv[i]) == 0) {
            pref = true;
        }
        else if (merge_forward.compare (argv[i]) == 0) {
            if (i != argc - 2)
                merge_f = atoi (argv[++i]);
            else {
                cout << "Illegal input." << endl;
                exit(-1);
            }
        }
        else if (check.compare (argv[i]) == 0) {
            check_correctness = true;
        }
        else if (gold.compare (argv[i]) == 0) {
            isGold = true;
        }
        else {
            cout << "Illegal input." << endl;
            return 0;
        }
    }

    bool bmerge_x = bmx > cmx;
    bool bmerge_y = bmy > cmy;
    int mx = bmerge_x ? bmx : cmx;
    int my = bmerge_y ? bmy : cmy;

    //cout << "Initiation ..." << endl;
    if (is3d) {
        DRStencil* dr_stencil = new DRStencil (distance, step_num, merge_f);
        char* stcfile = argv[argc - 1];
        if (dr_stencil->get_stencil (stcfile) != 0)
            exit (-1);
        //dr_stencil->gen_original ();
        string stencil_name = stcfile;
        stencil_name.erase (stencil_name.size() - 4);
    
        // Fusing stencil
        dr_stencil->fusing();

        //cout << "Generating semi Stencil ... " << endl;
        dr_stencil->semiGen();

        codeGen* code = new codeGen (dr_stencil, bx, by, sn, s_unroll, 
                                bmerge_x, bmerge_y, mx, my, pref,
                                stencil_name, check_correctness);
        code->output (out_name);
        delete code;
        delete dr_stencil;
    }
    else {
        DRStencil_2d* dr_stencil = new DRStencil_2d (distance, step_num, merge_f);
        char* stcfile = argv[argc - 1];
        if (dr_stencil->get_stencil (stcfile) != 0)
            exit (-1);
        //dr_stencil->gen_original ();
        string stencil_name = stcfile;
        stencil_name.erase (stencil_name.size() - 4);
    
        // Fusing stencil
        dr_stencil->fusing();
        
        //cout << "Generating semi Stencil ... " << endl;
        dr_stencil->semiGen();

        codeGen_2d* code = new codeGen_2d (dr_stencil, bx, by, streaming, sn, s_unroll, 
                               bmerge_x, bmerge_y, mx, my, pref,
                               stencil_name, check_correctness);
        code->output (out_name);
        delete code;
        delete dr_stencil;
    }
    return 0;
}
