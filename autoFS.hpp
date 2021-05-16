#ifndef __AUTOFS_HPP__
#define __AUTOFS_HPP__

# include <iostream>
# include <vector>
# include <tuple>
# include <sstream>
# include <fstream>
# include <map>
# include <set>
# include <algorithm>


class DRStencil
{
	private:
		int order, distance, step_num, low_k, high_k;
		int merge_forward;
		int L, M, N, iterations;
		std::set<std::tuple<int, int, int>> forward_k, forward_j, forward_i, backward;
		std::map<std::tuple<int, int, int>, double> stencil, fused;
		void do_fusing (int, int, int, double, int);
	public:
		DRStencil (int distance_, int step_num_, int merge_f) 
			: distance (distance_), step_num (step_num_), merge_forward (merge_f)
			{}
		int get_stencil (const std::string&);
		void get_problem_size (int &, int &, int &, int &);
		void set_order_distance ();
		std::string print_in (const std::tuple<int, int, int>&, bool, bool, bool);
		void semiGen ();
		void partition ();
		void fusing ();
		void cal_range ();
		int get_order () { return order; }
		int get_distance () { return distance; }
		int get_low_k () { return low_k; }
		int get_high_k () { return high_k; }
		int get_step_num () { return step_num; }
		std::string gen_forward_k (int, bool, bool);
		std::string gen_forward_j (int, bool, bool);
		std::string gen_forward_i (int, bool, bool);
		std::string gen_backward (int, bool, bool);
		bool forward_j_avilable () { return forward_j.size() > 0; }
		bool forward_i_avilable () { return forward_i.size() > 0; }
		std::string gen_gold ();
};

// Get stencil from .stc file.
int DRStencil::get_stencil (const std::string &filename)
{
	std::ifstream stcFile(filename, std::ios::in);
	if (!stcFile) {
		std::cout << "Error opening stencil file." << std::endl;
		return 1;
	}

	int k, j, i;
	double coe;
	std::string str;
	while (!stcFile.eof()) {
		stcFile >> str;
		if (str == "L") stcFile >> L;
		else if (str == "M") stcFile >> M;
		else if (str == "N") stcFile >> N;
		else if (str == "iterations") stcFile >> iterations;
		else if (str == "stencil") {
			while (stcFile >> k >> j >> i >> coe) {
				stencil[std::make_tuple(k, j, i)] = coe;
			}
		}
	}
	//std::cout << "There are " << stencil.size() << " points." << std::endl;
	stcFile.close();
	return 0;
}

void DRStencil::get_problem_size (int &l, int &m, int &n, int &iter)
{
	l = L;
	m = M;
	n = N;
	iter = iterations;
}

void DRStencil::set_order_distance ()
{
	// If the distance is not given, set distance.
	int high = 0, low = 0;
	for (const auto &[point, _] : stencil) {
		const auto &[k, j, i] = point;
		high = high > k ? high : k;
		low = low < k ? low : k;
	}
	//std::cout << "high is: " << high << " and low is: " << low << std::endl;
	order = high;

	// The user has set the distance.
	if (distance != 0) return;
	distance = ((high - low) >> 1);
}

std::string DRStencil::print_in(const std::tuple<int, int, int> &point, bool merge_j, bool merge_i, bool isGlobal)
{
	const auto &[k, j, i] = point;
	std::stringstream out;
	if (isGlobal) out << "in[k" << (k < 0 ? "" : "+" ) << k;
	else out << "in_shm[k" << k - low_k;
	//out << k - low_k;
	//if (k < 0) out << k;
	out << "][j";
	if (merge_j) out << "+mj";
	if (j > 0) out << "+" << j;
	if (j < 0) out << j;
	out << "][i";
	if (merge_i) out << "+mi";
	if (i > 0) out << "+" << i;
	if (i < 0) out << i;
	out << "]";
	return out.str();
}

std::string DRStencil::gen_forward_k (int cnt, bool merge_j, bool merge_i)
{
	std::stringstream out;
	std::string indent = "\t";
	bool flag = false;
	for (const auto &[k, j, i] : forward_k) {
		if (flag) out << std::endl << std::string (cnt + 1, '\t') << "+ ";
		else flag = true;
		out << "(" << stencil[std::make_tuple(k-distance, j, i)] << ") * "  << print_in(std::make_tuple(k, j, i), merge_j, merge_i, false);
	}
	//out << ";" << std::endl;
	return out.str();
}

std::string DRStencil::gen_forward_j (int cnt, bool merge_j, bool merge_i)
{
	std::stringstream out;
	std::string indent = "\t";
	bool flag = false;
	for (const auto &[k, j, i] : forward_j) {
		if (flag) out << std::endl << std::string (cnt + 1, '\t') << "+ ";
		else flag = true;
		out << "(" << stencil[std::make_tuple(k, j-distance, i)] << ") * " << print_in(std::make_tuple(k, j, i), merge_j, merge_i, false);
	}
	//out << ";" << std::endl;
	return out.str();
}

std::string DRStencil::gen_forward_i (int cnt, bool merge_j, bool merge_i)
{
	std::stringstream out;
	std::string indent = "\t";
	bool flag = false;
	for (const auto &[k, j, i] : forward_i) {
		if (flag) out << std::endl << std::string (cnt + 1, '\t') << "+ ";
		else flag = true;
		out << "(" << stencil[std::make_tuple(k, j, i-distance)] << ") * " << print_in(std::make_tuple(k, j, i), merge_j, merge_i, false);
	}
	//out << ";" << std::endl;
	return out.str();
}

std::string DRStencil::gen_backward (int cnt, bool merge_j, bool merge_i)
{
	std::stringstream out;
	std::string indent = "\t";
	bool flag = false;
	for (const auto &point : backward) {
		if (flag) out << std::endl << std::string (cnt + 1, '\t') << "+ ";
		else flag = true;
		out << "(" << stencil[point] << ") * " << print_in(point, merge_j, merge_i, false);
	}
	//out << ";" << std::endl;
	return out.str();
}

// Generate naive code for error check.
std::string DRStencil::gen_gold ()
{
	std::stringstream out;
	out << "out[k][j][i] = ";

	std::string indent ("\t\t\t");
	bool flag = false;
	for (const auto &[point, coe] : stencil) {
		if (flag) out << std::endl << indent << "+ ";
		else flag = true;
		out << "(" << coe << ") * " << print_in(point, false, false, true);
	}
	out << ";" << std::endl;
	return out.str();
}

void DRStencil::partition () {

	std::set<std::tuple<int, int, int> > contri_k, contri_j, contri_i;

	// (k - distance, j, i) contributing to (0, 0, 0) means 
	// (k, j, i) contributing to (distance, 0, 0)
	// same for j and i

	for (const auto &[point, _] : stencil) {
		const auto &[k, j, i] = point;
		if (stencil.find(std::make_tuple(k - distance, j, i)) != stencil.end()) {
			contri_k.insert(point);
		}
		if (stencil.find(std::make_tuple(k, j - distance, i)) != stencil.end()) {
			contri_j.insert(point);
		} 
		if (stencil.find(std::make_tuple(k, j, i - distance)) != stencil.end()) {
			contri_i.insert(point);
		} 
	}

	// done means contribution from the point to (0, 0, 0) is calulated before
	std::set<std::tuple<int, int, int> > done;

	for (const auto &[k, j, i] : contri_k) {
		forward_k.insert(std::make_tuple(k, j, i));
		done.insert(std::make_tuple(k - distance, j, i));
	}
	for (const auto &[k, j, i] : contri_j) {
		if (done.find(std::make_tuple(k, j - distance, i)) != done.end()) continue;
		forward_j.insert(std::make_tuple(k, j, i));
		done.insert(std::make_tuple(k, j - distance, i));
	}
	for (const auto &[k, j, i] : contri_i) {
		if (done.find(std::make_tuple(k, j, i - distance)) != done.end()) continue;
		forward_i.insert(std::make_tuple(k, j, i));
		done.insert(std::make_tuple(k, j, i - distance));
	}
	for (const auto &[point, _] : stencil ) {
		if (done.find(point) == done.end()) {
			backward.insert(point);
			done.insert(point);
		}
	}

	// Merge if the forward brings no benefit.
	if (forward_j.size() < merge_forward) {
		for (const auto &[k, j, i] : forward_j)
			backward.insert(std::make_tuple(k, j - distance, i));
		forward_j.clear();
	}
	if (forward_i.size() < merge_forward) {
		for (const auto &[k, j, i] : forward_i)
			backward.insert(std::make_tuple(k, j, i - distance));
		forward_i.clear();
	}
}

// fusing the stencil recursively
void DRStencil::do_fusing(int k, int j, int i, double coe, int step)
{
	if(step == 0) {
		if (fused.find(std::make_tuple(k, j, i)) != fused.end())
			fused[std::make_tuple(k, j, i)] += coe;
		else fused[std::make_tuple(k, j, i)] = coe;
		return;
	}

	for (const auto &[point, coe0] : stencil) {
		const auto &[k0, j0, i0] = point;
		//do_fusing(k + k0, j + j0, i + i0, coe + (step == step_num ? "" : "*") + coe0, step - 1);
		do_fusing(k + k0, j + j0, i + i0, coe * coe0, step - 1);
	}
}

void DRStencil::fusing ()
{
	do_fusing (0, 0, 0, 1.0, step_num);
	stencil = fused;
}

// calculate the range of points to fetch
void DRStencil::cal_range ()
{
	low_k = 1, high_k = -1;
	for (const auto &[k, j, i] : forward_k) {
		low_k = low_k < k ? low_k : k;
		high_k = high_k > k ? high_k : k;
	}
	for (const auto &[k, j, i] : forward_j) {
		low_k = low_k < k ? low_k : k;
		high_k = high_k > k ? high_k : k;
	}
	for (const auto &[k, j, i] : forward_i) {
		low_k = low_k < k ? low_k : k;
		high_k = high_k > k ? high_k : k;
	}
	for (const auto &[k, j, i] : backward) {
		low_k = low_k < k ? low_k : k;
		high_k = high_k > k ? high_k : k;
	}
}

void DRStencil::semiGen()
{
	set_order_distance ();
	//std::cout << "Dividing the stencil into several part ..." << std::endl;
	partition ();
	//std::cout << "The four part of the stencil is as following: " << std::endl;
	cal_range ();
}

#endif
