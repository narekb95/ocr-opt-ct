#include<bits/stdc++.h>
#include<fstream>
using namespace std;

using I = unsigned long long int;
using P = pair<I, I>;
using PP = pair<P, P>;

using V = vector<I>;
using VV = vector<V>;
using VP = vector<P>;
using VS = vector<string>;

using VV = vector<V>;
using VVP = vector<VP>;

constexpr I INF = I(-1);

// Prints a vector
template<class T>
ostream& operator <<(ostream& out, const vector<T>& a){
	for(const auto& x : a){
		out << x << " " ;
	}
	return out;
}

template<class S, class T>
ostream& operator<<(ostream& out, const pair<S,T>& p)
{
	out <<"(" << p.first << ", " << p.second << "),  ";
	return out;
}

void debug(const string& s)
{
#ifdef __DEBUG
	cout << s << endl;
#endif
}
void d_assert(bool b)
{
#ifdef __DEBUG
	assert(b);
#endif
}

void split_line(const string& str, VS& line)
{
	I s = str.size();
	line.clear();
	I i = 0, j =0;
	do{
		j++;
		if(j == s || str[j] == ' ')
		{
			line.push_back(str.substr(i, j-i));
			j++;
			i = j;
		}
	}while(j <= s);
}

// Arrangement: is the linear arrangement as given in input
// Index: is the index of each vertex in the linear arrangement
// Graph: adj lists
PP read_input(const string& file, V& arrangement, VV& graph)
{
	ifstream fin(file);
	I n1, n2, m, ctw;
	string s;

	I first_line = 0;
	I arrg_itr = 0;
	while(getline(fin, s))
	{
		if(s[0] == 'c')
			continue;
		VS line;
		split_line(s, line);
		if(s[0] == 'p')
		{
			assert(!first_line);
			first_line = 1;
			assert(line[1] == "ocr");
			n1 = stoi(line[2]);
			n2 = stoi(line[3]);
			m = stoi(line[4]);
			ctw = stoi(line[5]);
			graph.assign(n1+n2, V());
			arrangement.resize(n1 + n2);
			// index.resize(n1+n2);
			continue;
		}
		assert(first_line);
		if(arrg_itr < n1+ n2)
		{
			assert(line.size() == 1);
			arrangement[arrg_itr] = stoi(line[0]) - 1;
			// index[arrangement[arrg_itr]] = arrg_itr;
			arrg_itr++;
			continue;
		}
		assert(line.size() == 2);
		I x, y;
		x = stoi(line[0]) - 1;
		y = stoi(line[1]) - 1;
		graph[x].push_back(y);
		graph[y].push_back(x);		
	}
	return PP(P(n1, n2), P(m, ctw));
}



 void err(const string& err)
 {
// #ifdef __DEBUG
 	// cout << "Error " << err << endl;
	exit(-1);
// #else
	// while(true);
// #endif
 }
 I count_masks(I size)
 {
	if(size > 63)
	{
		err("Too many vertices in cut");
	}
	return 1LL<<size;
 }

bool is_fixed_partition(const I& v, const PP& parameters)
{
	return v < parameters.first.first;
}

V get_set_from_mask(const V& cut, const I& mask, I end, I start = 0)
{
	V out;
	for(I i = start; i < end; i++)
	{
		if(mask & (1LL<<i))
		{
			out.push_back(cut[i]);
		}
	}
	return out;
}

void print_current_dp(const V& cut, const I& cut_size, const V& sol){
#ifdef __DEBUG
		cout << "Solutions:" << endl;
		for(I mask = 0; mask < count_masks(cut_size); mask++)
		{
			cout << "[" << get_set_from_mask(cut, mask, cut_size) <<"]: " << sol[mask] << endl;
		}
		cout << endl << endl;
#endif
}

// Returns cuv and cvu.
// cvu is number of pairs i, j in permanent such that i < j, i neighbor of u and j neighbor of v.
// i.e. c(u,v) is the number of crossings of (v, u)
I count_crossings(I u, I v, const VV& graph, const V& index)
{
	static map<P,I> crossings;
	if(crossings.find({u,v}) == crossings.end())
	{
		const V& a1 = graph[u];
		const V& a2 = graph[v];
		I n1 = a1.size();
		I n2 = a2.size();
		I common = 0;
		I cuv = 0;
		I cvu = 0;
		for(I i = 0, j = 0; i < n1; i++)
		{
			while(j < n2 && index[a2[j]] < index[a1[i]])
			{
				j++;
			}		
			cvu += j;
			cuv += (n2 - j); // assume first not equal
			if(j < n2 && index[a2[j]] == index[a1[i]]) //subtract one if equal
			{
				common++;
				cuv--;
			}
		}
		assert(cuv + cvu + common == n1 * n2);
		crossings[{u, v}] = cvu;
		crossings[{v,u}] = cuv;
// #ifdef __DEBUG
// 		cout << "\n<>~~<>~~<> Count crossings: " << u << ", " << v << ": (" << cuv << ", " << cvu << "| " << crossings[P(u,v)] << ", " << crossings[P(v,u)] << ", " << common <<")"<< endl;
// #endif
	}
	return crossings[{u,v}];
}

// Counts crossings between S[i-1](MASK) and W, and between  W(MASK) and W\W(MASK)
// Added vertices in Mask come after rest of mask 
I crossings_with_mask(const I& mask, const V& cut, const I& cut_size, I sep_index, const VV& graph, const V& index)
{
	// Between S(mask) and W
	I Sm_W = 0;
	for(I i = 0 ; i < sep_index; i++)
	{
		if(!((1LL<<i)&mask))
		{
			continue;
		}
		I u = cut[i];
		for(I j = sep_index; j < cut_size; j++)
		{
			I v = cut[j];
			I c = count_crossings(u, v, graph, index);
			Sm_W += c;
		}
	}

	// Between W(mask) and all non-mask
	I Wm_R = 0;
	for(I i = sep_index; i < cut_size; i++)
	{
		if(!((1LL<<i)&mask))
		{
			continue;
		}
		for(I j = 0; j < cut_size; j++)
		{
			if((1LL<<j)&mask)
			{
				continue;
			}			
			I u = cut[i];
			I v = cut[j];
			Wm_R += count_crossings(u,v, graph, index);
		}
	}
	// cout << "S[mask]-W[all] crosses:  " << Sm_W << endl;
	// cout << "W[mask]-rest crosses: " << Wm_R << endl;
	return Sm_W + Wm_R;
}

V perm_DP;
void compute_best_permutation(const V& permutation_vertices, const VV& graph, const V& index, I start = 0, I end = INF)
{
	if(end == INF)
	{
		end = permutation_vertices.size();
	}
	I n = end - start;
	I m = count_masks(n);
	if(perm_DP.size()<m)
	{
		perm_DP.resize(m);
	}
	fill(perm_DP.begin(), perm_DP.begin()+m, INF);
	for(I mask = 0; mask < m; mask++)
	{
		if(__builtin_popcountll(mask) <= 1)
		{
			perm_DP[mask] = 0;
			continue;
		}
		for(I i = 0; i < n; i++)
		{
			I bit = (1LL<<i);
			if(!(mask & bit))
			{
				continue;
			}
			I u = permutation_vertices[start + i];
			I prev_mask = mask ^ bit;
			I cost = perm_DP[prev_mask];
			if(cost == INF)
			{
				continue;
			}
			for(I j = 0; j < n; j++)
			{
				I bit2 = (1LL<<j);
				if(!(prev_mask & bit2))
				{
					continue;
				}
				I w = permutation_vertices[start + j];
				cost += count_crossings(u, w, graph, index);
			}
			perm_DP[mask] = min(perm_DP[mask], cost);	
		}
	}
}

//assumes vertices have the same relative order in cut
void remove_vertices_from_cut(const V& vertices, V& cut, I& cut_size, V& cut_mask)
{
	I new_cut_ind = 0;
	I forget_ind = 0;
	for(I old_ind = 0; old_ind < cut_size; old_ind++)
	{
		I w = cut[old_ind];
		if(w == vertices[forget_ind])
		{
			forget_ind++;
			continue;
		}
		cut[new_cut_ind++] = w;
	}
	cut_size =  new_cut_ind;
	for(const auto& v : vertices)
	{
		assert(cut_mask[v] == 1);
		cut_mask[v] = 0;
	}
}

I two_masks_crossings(I prev_mask, I suffix_mask, const V& cut, const I& cut_size, const VV& graph, const V& index)
{
	V prev_vertices = get_set_from_mask(cut, prev_mask, cut_size);
	V suffix_vertices = get_set_from_mask(cut, suffix_mask, cut_size);
	I crossings = 0;
	for(const auto& u : prev_vertices)
	{
		for(const auto& v : suffix_vertices)
		{
			crossings += count_crossings(u, v, graph, index);
		}
	}
	return crossings;
}

I remove_bits_from_mask(I mask, I remove, I n_bits)
{
	I out_mask = 0;
	for(I i = 0, out_ind = 0; i < n_bits; i++)
	{
		if((1LL<<i) & remove)
		{
		    assert((1LL<<i) & mask);
			continue;
		}
		if((1LL<<i) & mask)
		{
			out_mask |= (1LL<<out_ind);
		}
		out_ind++;
	}
	return out_mask;
}

// Forget vertex and update cut and cut_mask.
// For each subset, try subset in mask and find best permutation over rest
// Since dp[subset] is local over vertices in the subset, can reuse values.
// i.e. first compute dp for the whole set and then try all subsets in mask 
void forget_vertices(VP& forget_data, V& cut, V& cut_mask, I& cut_size, const VV& graph, const V& index,
		V& curr_sol, I& curr_size, const V& last_sol, const I& last_size)
{
	I forget_size = forget_data.size();
	I new_size = cut_size - forget_size;

	V forget_vert(forget_size);
	V forget_cut_indices(forget_size);
	transform(forget_data.begin(), forget_data.end(), forget_vert.begin(), [](const P& p){return p.first;});
	transform(forget_data.begin(), forget_data.end(), forget_cut_indices.begin(), [](const P& p){return p.second;});

	I cut_forget_mask = 0;
	for(const auto& ind : forget_cut_indices)
	{
		cut_forget_mask |= 1LL<<ind;
	}

#ifdef __DEBUG
	cout << "Forget vertices: " << forget_vert << endl;
	cout << "Forget indices:  " << forget_cut_indices << endl;
	cout << endl;
#endif
	
	I total_masks_n = count_masks(cut_size);
	curr_size = count_masks(new_size);
	
	if(curr_sol.size() < curr_size)
	{
		curr_sol.resize(curr_size);
	}
	fill(curr_sol.begin(), curr_sol.begin()+curr_size, INF); // new mask size

	compute_best_permutation(cut, graph, index);

	I non_forget_bits = (total_masks_n - 1) ^ cut_forget_mask;
	for(I non_forget_mask = non_forget_bits;; non_forget_mask = (non_forget_mask - 1) & non_forget_bits)
	{
		I mask = non_forget_mask | cut_forget_mask;
		I anti_mask = (total_masks_n - 1) ^ mask;
		for(I prev_mask = mask;; prev_mask = (prev_mask-1)&mask) // mask that was already in solution
		{
			I suffix_mask = mask ^ prev_mask;
			if(prev_mask >= last_size || last_sol[prev_mask] == INF)
			{
				continue;
			}
			I prev_cost = last_sol[prev_mask];
			I suffix_cost = perm_DP[suffix_mask];
			I prev_to_suffix_cost =  two_masks_crossings(prev_mask, suffix_mask, cut, cut_size, graph, index);
			I forget_to_non_mask = two_masks_crossings(cut_forget_mask, anti_mask, cut, cut_size, graph, index);
			// left is non_forget to non_mask but will be added when either forgotten 
#ifdef __DEBUG
			cout << "-----" << endl;
			cout << "Mask set: " << get_set_from_mask(cut, mask, cut_size) << endl;
			cout << "Recursion set: " << get_set_from_mask(cut, prev_mask, cut_size) << endl;
			cout << "DP cost: " << prev_cost << endl;
			cout << "Permutation cost: " << suffix_cost << endl;
			cout <<  "Previous to suffix: " << prev_to_suffix_cost << endl;
			cout << "Forget to non-mask: " << forget_to_non_mask << endl;
#endif
			I cost = prev_cost + suffix_cost + prev_to_suffix_cost + forget_to_non_mask;
			// [TODO] not mask but updated mask
			I output_mask = remove_bits_from_mask(mask, cut_forget_mask, cut_size);
			assert(output_mask < curr_size);
			curr_sol[output_mask] = min(curr_sol[output_mask], cost);
			if(prev_mask == 0)
			{
				break;
			}
		}
		if(non_forget_mask == 0)
		{
			break;
		}
	}
	// remove vertices from cut
	remove_vertices_from_cut(forget_vert, cut, cut_size, cut_mask);
}

// Cut are non-fixed endpoints of cut edges 
// Vertices are only forgotten when no right neighbors
// Implication: No vertex is forgotten and reintroduced.
void run_solver(const VV& graph, const V& arrangement, const V& index, const VP& neighbor_range, const PP& parameters)
{
	I n = graph.size();
	V sol[2] = {V(), V(1, 0)};
	I sol_size[2] = {0, 1};
	V cut(graph.size());
	I cut_size = 0;
	V cut_mask(n); // this mask is over all vertices but later masks are over cut vertices only

	//swap after you get a new solution
	I curr_par = 0;
	I other_par = 1;

	
	for(I ind = 0; ind < n; ind++) // arrangement index
	{
		I v = arrangement[ind];
		d_assert(ind == index[v]);
		
#ifdef __DEBUG
		cout << "_______________________________________________________" << endl;
		cout << "Cut index: " << ind << " Vertex: " << v << endl;
		V cut_print;
		cout << "Cut before processing: ";
		copy(cut.begin(), cut.begin()+cut_size, ostream_iterator<I>(cout, " "));
		cout << endl;
#endif

		// Forget vertices with no right neighbors
		VP forget_vert;
		for(I i = 0; i < cut_size;i++) // index of vertices in cut
		{
			I w = cut[i];
			if(index[w] <= ind && neighbor_range[w].second <= ind)
			{
				forget_vert.push_back(P(w, i));
			}
		}
		if(!forget_vert.empty())
		{
			forget_vertices(forget_vert, cut, cut_mask, cut_size, graph, index, sol[curr_par], sol_size[curr_par], sol[other_par], sol_size[other_par]);
			print_current_dp(cut, cut_size, sol[curr_par]);
			swap(curr_par, other_par);
		}
		else
		{
			debug("No forget vertices.");
		}

		// Introduce v or its right neighbors if it has right neighbors
		// Either v or all its neighbors are fixed partition
		// Anything introduced is suited anyway so just permute added vertices and append them.
		debug("Introducing vertices");
		I prev_size = cut_size;
		if(neighbor_range[v].second > ind)
		{
			if(is_fixed_partition(v, parameters))
			{
				for(const auto& w : graph[v])
				{
					if(index[w] > ind && !cut_mask[w])
					{
						cut[cut_size++] = w;
						cut_mask[w] = 1;
					}
				}
			} 
			else
			{
				if(!cut_mask[v])
				{
					cut[cut_size++] = v;
					cut_mask[v] = 1;
				}
			}
		}
#ifdef __DEBUG
		if(prev_size < cut_size)
		{
			cout << "Added vertices: ";
			for(I h = prev_size; h < cut_size; h++)
			{
				cout << cut[h] << " ";
			}
			cout << endl;
		}
		else
		{
			cout << "No vertices added." << endl;
		
		}
#endif
	}
	assert(cut_size == 0);
	I ans = sol[other_par][0]; // empty mask
	cout << ans << endl;
}

void remove_isolated_vertices(VV& graph, V& arrangement, V& old_ids, V& solution, PP& parameters)
{
	I& n1 = parameters.first.first;
	I& n2 = parameters.first.second;
	I n = n1 + n2;
	V removed(n, 0);
	V new_ids(n);
	old_ids.clear();

	VV newgraph;
	for(uint i = 0; i < n; i++)
	{
		if(graph[i].empty())
		{
			solution.push_back(i);
			removed[i] = 1;
		}
		else
		{
			new_ids[i] = newgraph.size();
			old_ids.push_back(i);
			newgraph.push_back(move(graph[i]));
		}
	}
	graph = move(newgraph);

	// update data
	for(auto &v : graph)
	{
		for(auto &x : v)
		{
			x = new_ids[x];
		}
	}

	V new_arrangement;
	for(auto x : arrangement)
	{
		if(!removed[x])
		{
			new_arrangement.push_back(new_ids[x]);
		}
	}
	arrangement = move(new_arrangement);
	if(n1 > 0)
	{
		I last_fixed = n1 - 1;
		while(removed[last_fixed])
		{
			last_fixed--;
		}
		n1 = last_fixed + 1;
	}
	n = graph.size();
	n2 = n - n1;
}

void compute_index_and_sort(const V& arrangement, V& index, VV& graph, VP& neighbor_range)
{
	I n = arrangement.size();
	index.resize(n);
	for(I i = 0; i < n; i++)
	{
		index[arrangement[i]] = i;
	}
	for(auto& v : graph)
	{
		sort(v.begin(), v.end(), [&](const I& x, const I& y){
			return index[x] < index[y];
		});
	}
	neighbor_range.resize(n);
	transform(graph.begin(), graph.end(), neighbor_range.begin(), [&](const V& v){
		return P(index[v.front()], index[v.back()]);
	});
}

int main(int argc, char* argv[])
{
	ios_base::sync_with_stdio(0);
// #ifdef __DEBUG
	// // Test remove from cut
	// V testcut = {1, 5, 3, 7, 2,  6, 0, 0, 0};
	// I testsize = 6;
	// V remove{1,5, 2, 6};
	// 3 7
	// remove_vertices_from_cut(remove, testcut, testsize);
	// for(I i = 0; i < testsize; i++)
	// {
	// 	cout << testcut[i] << " ";
	// }
	// cout << endl;


	// I test_mask = 0b1101101;
	// I remv_mask =    0b1100;
	// I sz = 10;
	// I ans = remove_bits_from_mask(test_mask, remv_mask, sz);
	// bitset<10> expect(0b11001);
	// bitset<10> b1(ans);
	// cout << expect << " " << b1 << endl;
// #endif
	string file = argv[1];
	V arrangement;
	VV graph;
	PP parameters = read_input(file, arrangement, graph);
	
	// Remove isolated vertices
	V solution;
	V old_ids;
	remove_isolated_vertices(graph, arrangement, old_ids, solution, parameters);

	V index;
	VP neighbor_range;
	compute_index_and_sort(arrangement, index, graph, neighbor_range);

#ifdef __DEBUG
	cout << "Solution: " << solution << endl;
	cout << "Old ids: " << old_ids << endl;
	cout << "n1: " << parameters.first.first << " n2: " << parameters.first.second << endl;
	cout << "Graph: " << endl;
	for(const auto& v : graph)
	{
		cout << v << endl;
	}
	cout << endl;
	cout << "Arrangement: " << arrangement << endl;
	cout << "Index: " << index << endl;
	cout << "Neighbor range: " << neighbor_range << endl << endl << endl;
#endif

	run_solver(graph, arrangement, index, neighbor_range, parameters);
	return 0;
}
