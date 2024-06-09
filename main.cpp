#include<iterator>
#include<vector>
#include<iostream>
#include<map>
#include<bitset>
#include<fstream>
#include<cassert>
#include<numeric>
#include<algorithm>
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

constexpr I N = 20000;
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

inline void print_current_dp(const V& cut, const I& cut_size, const map<I,I>& sol){
#ifdef __DEBUG
	cout << "Solutions:" << endl;
	for(const auto& it : sol)
	{
		cout << "[" << get_set_from_mask(cut, it.first, cut_size) <<"]: " << it.second << endl;
	}
	cout << endl << endl;
#endif
}

void err(const string& err)
{
#ifdef __DEBUG
cout << "Error " << err << endl;
#endif
exit(-1);
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

map<P, I> committed;
// returns the smaller vertex if commited, or inf otherwise
inline I committed_order(I u, I v, const VP& range)
{
	if(u > v)
	{
		swap(u, v);
	}
	auto it = committed.find(P{u, v});
	if(it != committed.end())
	{
		cout << "found committed" << endl;
		return it->second;
	}
	
	if(range[u].second <= range[v].first) // if last of u is common, still no crossings
	{
		return (committed[{u,v}] = u);
	}
	if(range[v].second <= range[u].first)
	{
		return (committed[{u,v}] = v);
	}
	return (committed[{u,v}] = INF);
}

// Arrangement: is the linear arrangement as given in input
// Index: is the index of each vertex in the linear arrangement
// Graph: adj lists
PP read_input(V& arrangement, VV& graph, istream& in, bool with_arrangement = true)
{
	I n1, n2, m, ctw;
	string s;

	I first_line = 0;
	I arrg_itr = 0;
	while(getline(in, s))
	{
		if(s.size() == 0 || s[0] == '\n' || s[0] == '\r')
		{
			continue;
		}
		if(s[0] == 'c')
		{
			continue;
		}
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
			ctw = with_arrangement? stoi(line[5]) : m;
			graph.assign(n1+n2, V());
			arrangement.resize(n1 + n2);
			// index.resize(n1+n2);
			continue;
		}
		assert(first_line);
		if(with_arrangement)
		{
			if(arrg_itr < n1+ n2)
			{
				assert(line.size() == 1);
				arrangement[arrg_itr] = stoi(line[0]) - 1;
				arrg_itr++;
				continue;
			}
		}
		assert(line.size() == 2);
		I x, y;
		x = stoi(line[0]) - 1;
		y = stoi(line[1]) - 1;
		graph[x].push_back(y);
		graph[y].push_back(x);		
	}
	if(!with_arrangement)
	{
		iota(arrangement.begin(), arrangement.end(), 0);
	}
	return PP(P(n1, n2), P(m, ctw));
}

void remove_isolated_vertices(VV& graph, V& arrangement, V& old_ids, V& solution, PP& parameters)
{
	I& n1 = parameters.first.first;
	I& n2 = parameters.first.second;
	I n = n1 + n2;
	V removed(n, 0);
	V new_ids(n);
	old_ids.clear();
	I count_removed = 0;

	VV newgraph;
	for(I i = 0; i < n; i++)
	{
		if(graph[i].empty())
		{
			if(i >= n1)
			{
				solution.push_back(i);
			}
			removed[i] = 1;
			count_removed++;
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

	assert(count_removed + graph.size() == n);
	if(n1 > 0)
	{
		I last_fixed = n1 - 1;
		while(last_fixed > 0 && removed[last_fixed])
		{
			last_fixed--;
		}
		if(last_fixed == 0 && removed[0])
		{
			n1 = 0;
		}
		else
		{
			n1 = new_ids[last_fixed] + 1;
		}
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

V suffix_lower_bound;
bitset<N> introduced; // indexed by vertex itself
V intro_order;
I first_unintroduced; // index of vertex in intro_order
void compute_lower_bounds(const VP& neighbor_range, const PP& parameters, const VV& graph, const V& index){
	I n1 = parameters.first.first;
	I n = parameters.first.second;
	first_unintroduced = 0;
	introduced.reset();
	intro_order.resize(n);
	iota(intro_order.begin(), intro_order.end(), n1);
	sort(intro_order.begin(), intro_order.end(), [&](const I& x, const I& y){
		return neighbor_range[x].first < neighbor_range[y].first;
	});
	
	suffix_lower_bound.assign(n, 0);
	for(long long int i = n-1; i >= 0; i--)
	{
		I u = intro_order[i];
		suffix_lower_bound[i] = suffix_lower_bound[i+1];
		for(I j = i + 1; j < n; j++)
		{
			I v = intro_order[j];
			// All vertices afterwards have possibly smaller index than v but not that its introduce index
			if(neighbor_range[u].second < neighbor_range[v].first)
			{
				break;
			}
			// u is neighbor of v
			suffix_lower_bound[u] += min(count_crossings(u, v, graph, index), count_crossings(v, u, graph, index));
		}
	}
}

I compute_lb_set_to_unintro(const V& cut, const I& cut_size, const VV& graph, const V& index)
{
	I ans = 0;
	for(I i = first_unintroduced; i < cut_size; i++)
	{
		I v = cut[i];
		for(auto w : graph[v])
		{
			if(!introduced[w])
			{
				ans += min(count_crossings(v, w, graph, index), count_crossings(w, v, graph, index));
			}
		}
	}
#ifdef __DEBUG
	cout << "LB cut to right: [" << cut << "]: " << ans << endl;
#endif
	return ans;
}

// [TODO]
map<I,I> perm_DP;
V permutation_backtrack;
V permutation_last_vertex;
void compute_best_permutation(const V& permutation_vertices, const I& end, const VV& graph, const V& index, bool with_backtrack = false)
{
	I start = 0;
	I n = end - start;
	I m = count_masks(n);
	if(perm_DP.size()<m)
	{
		perm_DP.resize(m);
		permutation_backtrack.resize(m);
		permutation_last_vertex.resize(m);
	}
	fill(perm_DP.begin(), perm_DP.begin()+m, INF);
	for(I mask = 0; mask < m; mask++)
	{
		if(__builtin_popcountll(mask) <= 1)
		{
			perm_DP[mask] = 0;
			if(with_backtrack)
			{
				permutation_backtrack[mask] = 0;
				for(I i = 0; i < n; i++)
				{
					if(mask & (1LL<<i))
					{
						permutation_last_vertex[mask] = permutation_vertices[start + i];
						break;
					}
				}
			}
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
			if(cost < perm_DP[mask])
			{
				perm_DP[mask] = cost;
				if(with_backtrack)
				{
					permutation_backtrack[mask] = prev_mask;
					permutation_last_vertex[mask] = u;
				}
			}
		}
	}
}

V valid_masks;
I n_valid_masks;
void geenrate_valid_submasks(I mask, I ind, const I& alive_bits, const V& cut, const I& cut_size,
		const I& forget_mask, const VP& neighbor_range)
{
	if(ind == cut.size())
	{
		valid_masks[n_valid_masks++] = mask;
		return;
	}

	if(!(alive_bits & (1LL<<ind)))
	{
		geenrate_valid_submasks(mask, ind+1, alive_bits, cut, cut_size, forget_mask, neighbor_range);
		return;
	}

	I u = cut[ind];
	bool can_add = true;
	bool can_rem = (forget_mask & (1LL<<ind));
	for(I i = 0; i < ind; i++)
	{
		if(!(alive_bits & (1LL<<i)))
		{
			continue; // by induction
		}
		I w = cut[i];
		I ord = committed_order(u, w, neighbor_range);
		if(ord == u && ((1LL<<i) & mask))
		{
			can_rem = false;
		}
		if(ord == w && !((1LL<<i)&mask))
		{
			can_add = false;
		}
	}
	I bit = 1LL<<ind;
	if(can_add)
	{
		geenrate_valid_submasks(mask | bit, ind+1, alive_bits, cut, cut_size, forget_mask, neighbor_range);
	}
	if(can_rem)
	{
		geenrate_valid_submasks(mask, ind+1, alive_bits, cut, cut_size, forget_mask, neighbor_range);
	}
}

// Forget vertex.
// For each subset, try subset in mask and find best permutation over rest
// Since dp[subset] is local over vertices in the subset, can reuse values.
// i.e. first compute dp for the whole set and then try all subsets in mask 
// - sol is indexed by after-forgetting
// by cut_history and sol_vertices values from current cut (index corresp to after-forgetting but value to before)
// Since between iterations we only insert vertices, previous masks are still valid for the new cut
void forget_vertices(VP& forget_data, V& cut, I& cut_size, const VV& graph, const V& index, const VP& neighbor_range,
		map<I, I>& curr_sol, I& curr_size, const map<I, I>& last_sol, const I& last_size,
		VV& cut_sol_masks, VV& sol_back_pointer, VV& cut_history, const PP& parameters, I upperbound)
{
	I cut_lower_bound = first_unintroduced <= parameters.first.first ? suffix_lower_bound[first_unintroduced] : 0;
	
	cut_history.push_back(V(cut_size));
	copy(cut.begin(), cut.begin()+cut_size, cut_history.back().begin());
	// cout << "Cut: " << cut_history.back() << endl;

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
	I total_bits = total_masks_n - 1;

	curr_sol.clear();
	cut_sol_masks.push_back(V(curr_size));
	sol_back_pointer.push_back(V(curr_size));

	compute_best_permutation(cut, cut_size, graph, index);

	if(valid_masks.size() < count_masks(cut_size))
	{
		valid_masks.resize(count_masks(cut_size));
	}
	n_valid_masks = 0;

	for(const auto& it : last_sol)
	{
		const I& prev_mask = it.first;
		const I& prev_cost = it.second;
		assert(total_bits & prev_mask == prev_mask);
		I left_bits = total_bits ^ prev_mask;
		geenrate_valid_submasks(0, 0, left_bits, cut, cut_size, cut_forget_mask, neighbor_range);
		for(I mask_ind = 0; mask_ind < n_valid_masks; mask_ind++)
		{
			I suffix_mask = valid_masks[mask_ind];
			auto perm_it = perm_DP.find(suffix_mask);
			if(perm_it == perm_DP.end()) // no permutation for this set
			{
				continue;
			}
			I suffix_cost = perm_it->second;


			I mask = prev_mask | suffix_mask;
			assert(total_bits & mask == mask);
			assert(cut_forget_mask & mask == cut_forget_mask);
			I anti_mask = total_bits ^ mask;

			I prev_to_suffix_cost =  two_masks_crossings(prev_mask, suffix_mask, cut, cut_size, graph, index);
			I forget_to_non_mask = two_masks_crossings(cut_forget_mask, anti_mask, cut, cut_size, graph, index);

			I cost = prev_cost + suffix_cost + prev_to_suffix_cost + forget_to_non_mask;

			I lb_no_forget_to_rest = compute_lb_set_to_unintro(cut, cut_size, graph, index);
			if(cost + cut_lower_bound + lb_no_forget_to_rest > upperbound) // [TODO] + minimum cost in antimask from from cut to right of cut?
				continue;
			if(curr_sol[mask] > cost)
			{
				curr_sol[mask] = cost;
				cut_sol_masks.back()[mask] = mask;
				sol_back_pointer.back()[mask] = prev_mask;
			}
		}
	}

	I non_forget_bits = (total_masks_n - 1) ^ cut_forget_mask;
	for(I non_forget_mask = non_forget_bits;; non_forget_mask = (non_forget_mask - 1) & non_forget_bits)
	{
		I mask = non_forget_mask | cut_forget_mask;
		I anti_mask = (total_masks_n - 1) ^ mask;
		for(I prev_mask = mask;; prev_mask = (prev_mask-1)&mask) // mask that was already in solution
		{
			I suffix_mask = mask ^ prev_mask;
			auto last_it = last_sol.find(prev_mask);
			if(last_it == last_sol.end())
			{
				continue;
			}
			I prev_cost = last_it->second;
			
			auto perm_it = perm_DP.find(suffix_mask);
			if(perm_it == perm_DP.end())
			{
				continue;
			}
			I suffix_cost = perm_it->second;
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
			if(curr_sol[output_mask] > cost)
			{
				// index output-mask but value is mask
				curr_sol[output_mask] = cost;
				cut_sol_masks.back()[output_mask] = suffix_mask;
				sol_back_pointer.back()[output_mask] = prev_mask;
			}

			// end loop here to go over all and none
			if(prev_mask == 0)
			{
				break;
			}
		}
		// end loop here to go over all and none
		if(non_forget_mask == 0)
		{
			break;
		}
	}
	// remove vertices from cut
	remove_vertices_from_cut(forget_vert, cut, cut_size);
}

// Cut are non-fixed endpoints of cut edges 
// Vertices are only forgotten when no right neighbors
// Implication: No vertex is forgotten and reintroduced.
I run_solver(const VV& graph, const V& arrangement, const V& index, const VP& neighbor_range,
		const PP& parameters, VV& cut_sol_masks, VV& sol_back_points, VV& cut_history, I upper_bound)
{
	I n = graph.size();
	map<I, I> sol[2];
	sol[1][0] = 0;

	I sol_size[2] = {0, 1};
	V cut(graph.size());
	I cut_size = 0;

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
			debug("Calling forget.");
			forget_vertices(forget_vert, cut, cut_size, graph, index, neighbor_range,
				sol[curr_par], sol_size[curr_par], sol[other_par], sol_size[other_par],
				cut_sol_masks, sol_back_points, cut_history, parameters, upper_bound);
			if(sol[curr_par].size() == 0)
			{
				return false;
			}
			print_current_dp(cut, cut_size, sol[curr_par]);
			swap(curr_par, other_par);
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
					assert(!is_fixed_partition(w, parameters));
					if(index[w] > ind && !introduced[w])
					{
						cut[cut_size++] = w;
						introduced[w] = 1;
					}
				}
			} 
			else
			{
				if(!introduced[v])
				{
					cut[cut_size++] = v;
					introduced[v] = 1;
				}
			}
		}
		while(first_unintroduced < parameters.first.second 
			&& introduced[intro_order[first_unintroduced]])
		{
			first_unintroduced++;
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
	assert(cut_size < parameters.second.second);

	I ans = sol[other_par][0]; // empty mask
#ifdef __DEBUG
	cout << endl << "Optimaml crossings: " << ans << endl << endl;
#endif
	return ans;
}

void print_solution_backwards(const VV& cut_sol_masks, const VV& sol_back_points, const VV& cut_history,
		const VV& graph, const V& index, V& out)
{
#ifdef __DEBUG

	cout << "Cut history: " << cut_history.size() << endl;;
	for(const auto& v : cut_history)
	{
		cout << v << endl;
	}
	cout << endl;
	cout << "Solutions masks: " << cut_sol_masks.size() << endl;
	for(const auto& v : cut_sol_masks)
	{
		cout << v << endl;
	}
	cout << endl;
#endif

	I n = cut_sol_masks.size();
	for(long long int  i = n-1, mask = 0; i >= 0; i--)
	{
		I sol = cut_sol_masks[i][mask];

		V vertices = get_set_from_mask(cut_history[i], sol, cut_history[i].size());
		compute_best_permutation(vertices, vertices.size(), graph, index, true);
		V ordered_vertices(vertices.size());
		for(I j = 0, perm_mask = count_masks(vertices.size()) - 1; j < vertices.size(); j++)
		{
			ordered_vertices[j] = permutation_last_vertex[perm_mask];
			perm_mask = permutation_backtrack[perm_mask];
		}
		#ifdef __DEBUG
		cout << "Batch: " << ordered_vertices << endl;
		#endif

		copy(ordered_vertices.rbegin(), ordered_vertices.rend(), back_inserter(out));
		mask = sol_back_points[i][mask];
	}
	reverse(out.begin(), out.end());
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

#ifdef __FILEIO
	string in_file = argv[1];
	string out_file = argv[2];
	ifstream fin(in_file);
	ofstream fout(out_file);
	istream& in = fin;
	ostream& out = fout;
#else
	istream& in = cin;
	ostream& out = cout;
#endif

V arrangement;
VV graph;

// #ifndef __LITE
// #define __LITE
// #endif

#ifdef __LITE
	PP parameters = read_input(arrangement, graph, in, false);
#else
	PP parameters = read_input(arrangement, graph, in);
#endif
	
	// Remove isolated vertices
	V sol_isolated_vertices;
	V old_ids;
	remove_isolated_vertices(graph, arrangement, old_ids, sol_isolated_vertices, parameters);

	V index;
	VP neighbor_range;
	compute_index_and_sort(arrangement, index, graph, neighbor_range);

	compute_lower_bounds(neighbor_range, parameters, graph, index);

#ifdef __DEBUG
	cout << "Solution: " << sol_isolated_vertices << endl;
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

	VV cut_sol_masks, sol_back_pointer, cut_history;
	I ub = suffix_lower_bound[0];
	while(!run_solver(graph, arrangement, index, neighbor_range, parameters, cut_sol_masks, sol_back_pointer, cut_history, ub))
	{
		ub++;
	}
	
	V out_arr;
	print_solution_backwards(cut_sol_masks, sol_back_pointer, cut_history, graph, index, out_arr);
	for(auto& v : out_arr)
	{
		assert(v >= parameters.first.first);
		v = old_ids[v];
	}
	copy(sol_isolated_vertices.begin(), sol_isolated_vertices.end(), back_inserter(out_arr));
	for(auto v : out_arr)
	{
		assert(v >= parameters.first.first);
		out << v+1 << endl;
	}
	return 0;
}
