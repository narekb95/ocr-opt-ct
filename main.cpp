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

// Updates mask after removing element from a set
I remove_index(const I& mask, const I& ind)
{
	I all_ones = I(-1);
	I bit = 1LL<<ind;
	I all_before = bit-1;
	I all_after = all_ones ^ all_before;
	I before = all_before & mask;
	I after = all_after & mask; 
	after >>= 1;
	return (before | after);
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
			cuv += j;
			cvu += (n2 - j); // assume first that current is larger
			if(j < n2 && index[a2[j]] == index[a1[i]]) //subtract one if equal
			{
				common++;
				cvu--;
			}
		}
		assert(cuv + cvu + common == n1 * n2);
		crossings[{u, v}] = cvu;
		crossings[{v,u}] = cuv;
// #ifdef __DEBUG
		// cout << "Count crossings: " << u << ", " << v << ": (" << crossings[P(u,v)] << ", " << crossings[P(v,u)] << ", " << common <<")"<< endl;
// #endif
	}
	return crossings[{u,v}];
}

// Counts crossings between S[i-1](MASK) and W, and between  W(MASK) and W\W(MASK)
// Added vertices in Mask come after rest of mask 
I crossings_with_mask(const I& mask, const V& cut, const I& cut_size, I sep_index, const VV& graph, const V& index)
{
	I ans = 0;
	
	// Between S(mask) and W
	for(I i = 0 ; i < sep_index; i++)
	{
		I u = cut[i];
		if(!((1LL<<i)&mask))
		{
			continue;
		}
		for(I j = sep_index; j < cut_size; j++)
		{
			I v = cut[j];
			ans += count_crossings(u,v, graph, index);
		}
	}

	// Between W(mask) and W(rest)
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

			ans += count_crossings(u,v, graph, index);
		}
	}
	return ans;
}

V perm_DP;
I best_permutation(const V& permutation_vertices, const VV& graph, const V& index)
{
	I n = permutation_vertices.size();
	I m = count_masks((permutation_vertices.size()));
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
			I u = permutation_vertices[i];
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
				I w = permutation_vertices[j];
				cost += count_crossings(u, w, graph, index);
			}
			perm_DP[mask] = min(perm_DP[mask], cost);	
		}
	}
	return perm_DP[m-1];
}

// [TODO] This should become obsolete by precomputing DP for the full set of added vertices
// Use dynamic programming over subsets of new vertices (Hamiltonian cycle)
I best_permutation_fixed_mask(const I& mask, const V& cut, const I& cut_size, const I& sep_index, const VV& graph, const V& index)
{
	V permutation_vertices = get_set_from_mask(cut, mask, cut_size, sep_index);
	if(permutation_vertices.size() <= 1)
	{
		return 0;
	}
// #ifdef __DEBUG
// 	cout << "permutation crossings:\nPermutation vertices: " << get_set_from_mask(cut, mask, cut_size, sep_index) << endl; 
// 	cout << "first added index: " << sep_index << " cut size: " << cut_size << endl;
// 	cout << "Permutation vertices: " <<  permutation_vertices << endl;
// #endif

	best_permutation(permutation_vertices, graph, index);
	return perm_DP[count_masks(permutation_vertices.size())-1];
}

I add_indices_back(I mask, const V& removed_indices, I old_size)
{
	if(removed_indices[0] == 0)
	{
		mask <<= 1;
		mask |= 1;
	}
	I all_bits = count_masks(old_size) - 1;
	I c = removed_indices.size();
	for(I bit = 1, ind = 1, prev_bits = 1; bit < old_size; bit++)
	{
		if(ind < c && removed_indices[ind] == bit)
		{
			I otherbits = all_bits ^ prev_bits;
			I small_mask = mask & prev_bits;
			I big_mask = mask & otherbits;
			big_mask = (big_mask << 1);
			mask = small_mask | (1LL<<bit) | big_mask;
			ind++;
		}
		prev_bits = ((prev_bits<<1) | 1);
	}
	return mask;
}


//assumes vertices have the same relative order in cut
void remove_vertices_from_cut(const V& vertices, V& cut, I& cut_size)
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
}


// [TODO]
// Forget vertex and update cut and cut_mask.
// For each subset, try subset in mask and find best permutation over rest
// Since dp[subset] is local over vertices in the subset, can reuse values.
// i.e. first compute dp for the whole set and then try all subsets in mask 
void forget_vertices(VP& forget_data, V& cut, V& cut_mask, I& cut_size, const VV& graph, V& curr_sol, const V& last_sol)
{
	I forget_size = forget_data.size();
	V forget_vert(forget_size);
	V forget_cut_indices(forget_size);
	transform(forget_data.begin(), forget_data.end(), forget_vert.begin(), [](const P& p){return p.first;});
	transform(forget_data.begin(), forget_data.end(), forget_cut_indices.begin(), [](const P& p){return p.second;});
#ifdef __DEBUG
	cout << "Forget vertices: " << forget_vert << endl;
	cout << "Forget indices: " << forget_cut_indices << endl;
#endif
	I new_size = cut_size - forget_size;
	I total_masks_n = count_masks(cut_size);
	I new_masks_n = count_masks(new_size);
	I forget_masks_n = count_masks(forget_size);
	curr_sol.assign(new_masks_n, INF); // new mask size
	for(I mask = 0; mask < total_masks_n; mask++)
	{
		I forget_mask = 0;
		I left_mask = 0;
		I forget_ind = 0;
		I left_ind = 0;
		for(I i = 0; i < cut_size; i++)
		{
			if(!((1LL<<i) & mask))
			{
				continue;
			}
			if(forget_ind < forget_size && forget_cut_indices[forget_ind] == i)
			{
					forget_mask |= (1LL<<forget_ind);
					forget_ind++;
			}
			else
			{
				left_mask |= (1LL<<left_ind);
				left_ind++;
			}
		}
		assert(left_mask < new_masks_n);
		assert(forget_mask < forget_masks_n);
		I forget_not_in_mask = (forget_masks_n-1) ^ forget_mask;
		curr_sol[left_mask] = min(curr_sol[left_mask], last_sol[mask] + perm_DP[forget_not_in_mask]);
	}
	// remove vertices from cut
	remove_vertices_from_cut(forget_vert, cut, cut_size);
	assert(cut_size == new_size);
}

// Cut are non-fixed endpoints of cut edges 
// Vertices are only forgotten when no right neighbors
// Implication: No vertex is forgotten and reintroduced.
void run_solver(const VV& graph, const V& arrangement, const V& index, const VP& neighbor_range, const PP& parameters)
{
	I n = graph.size();
	I cutw = parameters.second.second;
	I dp_size = (1LL<<cutw);
	V sol[2] = {V(dp_size), V(dp_size)};
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
			forget_vertices(forget_vert, cut, cut_mask, cut_size, graph, sol[curr_par], sol[other_par]);
#ifdef __DEBUG
			for(I mask = 0; mask < count_masks(cut_size); mask++)
			{
				cout << "[" << get_set_from_mask(cut, mask, cut_size) <<"]: " << sol[curr_par][mask] << endl;
			}
			cout << endl << endl;
#endif
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
#ifdef __DEBUG
				cout << "Fixed partition\nAdded vertices: ";
				for(I h = prev_size; h < cut_size; h++)
				{
					cout << cut[h] << " ";
				}
				cout << endl;
#endif
			} 
			else
			{
				if(!cut_mask[v])
				{
					debug("vertex added to partition.");
					cut[cut_size++] = v;
					cut_mask[v] = 1;
				}
				else
				{
					debug("vertex already in partition");
				}
			}
		}
		if(cut_size == prev_size)
		{
			debug("No introduced vertices.\n\n");
			continue;
		}
		// Update the DP table
		V& curr_sol = sol[curr_par];
		const V& last_sol = sol[other_par];
		curr_sol.assign(count_masks(cut_size), INF);
		I n_masks = count_masks(cut_size);
		I old_bits = count_masks(prev_size) - 1;
		// [TODO] this computes permutation for the same sets multiple times
		// -> Precompute the dp table and access its values globally.
		for(I mask = 0; mask < n_masks; mask++)
		{
			I old_mask = mask & old_bits;
			V added_vertices;
			for(I j = prev_size; j < cut_size; j++)
			{
				if(mask & (1LL<<j))
				{
					added_vertices.push_back(cut[j]);
				}
			}
#ifdef __DEBUG
			cout << "Full mask: " << get_set_from_mask(cut, mask, cut_size, 0) << endl;
			cout << "Crossings with mask: "<< crossings_with_mask(mask, cut, cut_size, prev_size, graph, index) <<endl;
			cout << "Best permutation: " << best_permutation_fixed_mask(mask, cut, cut_size, prev_size, graph, index) << endl;
#endif
			curr_sol[mask] = last_sol[old_mask] 
				+ crossings_with_mask(mask, cut, cut_size, prev_size, graph, index)
				+ best_permutation_fixed_mask(mask, cut, cut_size, prev_size, graph, index);
		}
#ifdef __DEBUG
		cout << "Solutions:" << endl;
		for(I mask = 0; mask < count_masks(cut_size); mask++)
		{
			cout << "[" << get_set_from_mask(cut, mask, cut_size) <<"]: " << curr_sol[mask] << endl;
		}
		cout << endl << endl;
#endif
		swap(curr_par, other_par);
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
//
// // Test add_indiced back (obsolete)
// 	cout << 0b10111 << " " <<  add_indices_back(0b101, {0,2}, 5) << endl;
// 	cout << 0b1101111 << " " <<  add_indices_back(0b1011, {0,1,6}, 7) << endl;
// 	cout << 0b1001111 << " " <<  add_indices_back(0b11, {0,1,6}, 7) << endl;
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
