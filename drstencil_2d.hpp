#ifndef __DRSTENCIL_2D_HPP__
#define __DRSTENCIL_2D_HPP__

# include <iostream>
# include <vector>
# include <tuple>
# include <sstream>
# include <fstream>
# include <map>
# include <set>
# include <algorithm>


class DRStencil_2d
{
	private:
		int order, distance, step_num, low_j, high_j;
		int merge_forward;
		int M, N, iterations;
		std::set<std::tuple<int, int>> forward_j, forward_i, backward;
		std::map<std::tuple<int, int>, double> stencil, fused;
		void do_fusing (int, int, double, int);
	public:
		DRStencil_2d (int distance_, int step_num_, int merge_f) 
			: distance (distance_), step_num (step_num_), merge_forward (merge_f)
			{}
		int get_stencil (const std::string&);
		void get_problem_size (int &, int &, int &);
		void set_order_distance ();
		std::string print_in (const std::tuple<int, int>&, bool, bool, bool, bool);
		void dataReuse ();
		void partition ();
		void fusing ();
		void cal_range ();
		int get_order () { return order; }
		int get_distance () { return distance; }
		int get_low_j () { return low_j; }
		int get_high_j () { return high_j; }
		int get_step_num () { return step_num; }
		std::string gen_forward_j (int, bool, bool, bool);
		std::string gen_forward_i (int, bool, bool, bool);
		std::string gen_backward (int, bool, bool, bool);
		bool forward_i_avilable () { return forward_i.size() > 0; }
		std::string gen_gold ();
};

// Get stencil from .stc file.
int DRStencil_2d::get_stencil (const std::string &filename)
{
	std::ifstream stcFile(filename, std::ios::in);
	if (!stcFile) {
		std::cout << "Error opening stencil file." << std::endl;
		return 1;
	}

	int j, i;
	double coe;
	std::string str;
	while (!stcFile.eof()) {
		stcFile >> str;
		if (str == "M") stcFile >> M;
		else if (str == "N") stcFile >> N;
		else if (str == "iterations") stcFile >> iterations;
		else if (str == "stencil") {
			while (stcFile >> j >> i >> coe) {
				stencil[std::make_tuple(j, i)] = coe;
			}
		}
	}
	//std::cout << "There are " << stencil.size() << " points." << std::endl;
	stcFile.close();
	return 0;
}

void DRStencil_2d::get_problem_size (int &m, int &n, int &iter)
{
	m = M;
	n = N;
	iter = iterations;
}

void DRStencil_2d::set_order_distance ()
{
	// If the distance is not given, set distance.
	int high = 0, low = 0;
	for (const auto &[point, _] : stencil) {
		const auto &[j, i] = point;
		high = high > j ? high : j;
		low = low < j ? low : j;
	}
	//std::cout << "high is: " << high << " and low is: " << low << std::endl;
	order = high;

	// The user has set the distance.
	if (distance != 0) return;
	distance = ((high - low) >> 1);
}

std::string DRStencil_2d::print_in(const std::tuple<int, int> &point, bool streaming, bool merge_j, bool merge_i, bool isGlobal)
{
	const auto &[j, i] = point;
	std::stringstream out;
	if (isGlobal) out << "in[j" << (j < 0 ? "" : "+" ) << j;
	else if (streaming) out << "in_shm[j" << j - low_j;
    else {
        out << "in_shm[j" << (merge_j ? "+mj" : "");
	    if (j > 0) out << "+" << j;
	    if (j < 0) out << j;
    }
	//out << j - low_j;
	//if (j < 0) out << j;
	out << "][i";
	if (merge_i) out << "+mi";
	if (i > 0) out << "+" << i;
	if (i < 0) out << i;
	out << "]";
	return out.str();
}

std::string DRStencil_2d::gen_forward_j (int cnt, bool streaming, bool merge_j, bool merge_i)
{
	std::stringstream out;
	std::string indent = "\t";
	bool flag = false;
	for (const auto &[j, i] : forward_j) {
		if (flag) out << std::endl << std::string (cnt + 1, '\t') << "+ ";
		else flag = true;
		out << "(" << stencil[std::make_tuple(j-distance, i)] << ") * "  << print_in(std::make_tuple(j, i), streaming, merge_j, merge_i, false);
	}
	//out << ";" << std::endl;
	return out.str();
}


std::string DRStencil_2d::gen_forward_i (int cnt, bool streaming, bool merge_j, bool merge_i)
{
	std::stringstream out;
	std::string indent = "\t";
	bool flag = false;
	for (const auto &[j, i] : forward_i) {
		if (flag) out << std::endl << std::string (cnt + 1, '\t') << "+ ";
		else flag = true;
		out << "(" << stencil[std::make_tuple(j, i-distance)] << ") * " << print_in(std::make_tuple(j, i), streaming, merge_j, merge_i, false);
	}
	//out << ";" << std::endl;
	return out.str();
}

std::string DRStencil_2d::gen_backward (int cnt, bool streaming, bool merge_j, bool merge_i)
{
	std::stringstream out;
	std::string indent = "\t";
	bool flag = false;
	for (const auto &point : backward) {
		if (flag) out << std::endl << std::string (cnt + 1, '\t') << "+ ";
		else flag = true;
		out << "(" << stencil[point] << ") * " << print_in(point, streaming, merge_j, merge_i, false);
	}
	//out << ";" << std::endl;
	return out.str();
}

// Generate naive code for error checj.
std::string DRStencil_2d::gen_gold ()
{
	std::stringstream out;
	out << "out[j][i] = ";

	std::string indent ("\t\t\t");
	bool flag = false;
	for (const auto &[point, coe] : stencil) {
		if (flag) out << std::endl << indent << "+ ";
		else flag = true;
		out << "(" << coe << ") * " << print_in(point, false, false, false, true);
	}
	out << ";" << std::endl;
	return out.str();
}

void DRStencil_2d::partition () {

	std::set<std::tuple<int, int> > contri_j, contri_i;

	// (j - distance, j, i) contributing to (0, 0, 0) means 
	// (j, i) contributing to (distance, 0, 0)
	// same for j and i

	for (const auto &[point, _] : stencil) {
		const auto &[j, i] = point;
		if (stencil.find(std::make_tuple(j - distance, i)) != stencil.end()) {
			contri_j.insert(point);
		}
		if (stencil.find(std::make_tuple(j, i - distance)) != stencil.end()) {
			contri_i.insert(point);
		} 
	}

	// done means contribution from the point to (0, 0, 0) is calulated before
	std::set<std::tuple<int, int> > done;

	for (const auto &[j, i] : contri_j) {
		forward_j.insert(std::make_tuple(j, i));
		done.insert(std::make_tuple(j - distance, i));
	}
	for (const auto &[j, i] : contri_i) {
		if (done.find(std::make_tuple(j, i - distance)) != done.end()) continue;
		forward_i.insert(std::make_tuple(j, i));
		done.insert(std::make_tuple(j, i - distance));
	}
	for (const auto &[point, _] : stencil ) {
		if (done.find(point) == done.end()) {
			backward.insert(point);
			done.insert(point);
		}
	}

    if (forward_j.size() == 0) {
        std::cout << "No data to reuse. You can try another dist.\n";
        exit (1);
    }

	// Merge if the forward brings no benefit.
	if (forward_i.size() < merge_forward) {
		for (const auto &[j, i] : forward_i)
			backward.insert(std::make_tuple(j, i - distance));
		forward_i.clear();
	}
}

// fusing the stencil recursively
void DRStencil_2d::do_fusing(int j, int i, double coe, int step)
{
	if(step == 0) {
		if (fused.find(std::make_tuple(j, i)) != fused.end())
			fused[std::make_tuple(j, i)] += coe;
		else fused[std::make_tuple(j, i)] = coe;
		return;
	}

	for (const auto &[point, coe0] : stencil) {
		const auto &[j0, i0] = point;
		//do_fusing(j + j0, i + i0, coe + (step == step_num ? "" : "*") + coe0, step - 1);
		do_fusing(j + j0, i + i0, coe * coe0, step - 1);
	}
}

void DRStencil_2d::fusing ()
{
	do_fusing (0, 0, 1.0, step_num);
	stencil = fused;
}

// calculate the range of points to fetch
void DRStencil_2d::cal_range ()
{
	low_j = 1, high_j = -1;
	for (const auto &[j, i] : forward_j) {
		low_j = low_j < j ? low_j : j;
		high_j = high_j > j ? high_j : j;
	}
	for (const auto &[j, i] : forward_i) {
		low_j = low_j < j ? low_j : j;
		high_j = high_j > j ? high_j : j;
	}
	for (const auto &[j, i] : backward) {
		low_j = low_j < j ? low_j : j;
		high_j = high_j > j ? high_j : j;
	}
}

void DRStencil_2d::dataReuse ()
{
	set_order_distance ();
	partition ();
	cal_range ();
}

#endif
