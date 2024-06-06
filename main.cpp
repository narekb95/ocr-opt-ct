#include<bits/stdc++.h>

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
PP read_input(V& arrangement, VV& graph)
{
	I n1, n2, m, ctw;
	string s;

	I first_line = 0;
	I arrg_itr = 0;
	while(getline(cin, s))
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
#ifdef __DEBUG
 	cout << "Error " << err << endl;
	exit(-1);
#else
	while(true);
#endif
 }
 I count_masks(I size)
 {
	if(size > 64)
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

// Returns cuv and cvu.
// cvu is number of pairs i, j in permanent such that i < j, i neighbor of u and j neighbor of v.
// i.e. c(u,v) is the number of crossings of (v, u)
P count_crossings(I u, I v, const VV& graph, const V& index)
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
	cout << "u: " << u + 1 << " v: " << v + 1 << " cuv: " << cuv << " cvu: " << cvu << " common: " << common << endl;
	assert(cuv + cvu + common == n1 * n2);
	return P(cuv, cvu);
}
// Actually new vertices do not have edges with forgotten edges, so only crossings introduced are between old mask and new mask
// Build the function C (count crossings between a pair) dynamically, because for many pairs will never be computed
I crossings_with_mask(const V& cut, I sep_index, const VV& graph, const V& index)
{
	I n = cut.size();
	I i = 0, j = sep_index;
	static map<P,I> crossings;
	map<P,I>::iterator it;
	for(i = 0 ; i < sep_index; i++)
	{
		for(j = sep_index; j < n; j++)
		{
			I u = cut[i];
			I v = cut[j];
			if((it = crossings.find({u,v})) == crossings.end())
			{
				P c = count_crossings(u,v, graph, index);
				// swap left and right since c is defined as "well ordered paris".
				crossings[{v, u}] = c.first;
				crossings[{u, v}] = c.second;
			}
			return crossings[{u,v}];
		}
	}
	return 0;
}

// Use dynamic programming over subsets of new vertices (Hamiltonian cycle)
I best_permutation(const V& vertices, const VV& graph, const V& index)
{
	I n = vertices.size();
	V submasks;
	map<I, I> dp;
	I m = count_masks(n) - 1;
	for(I s = m; s != 0; s = (s-1)&m)
	{
		submasks.push_back(s);
	}
	reverse(submasks.begin(), submasks.end());
	for(I mask : submasks){
		if(__builtin_popcountll(mask) == 1)
		{
			dp[mask] = 0;
			continue;
		}
		dp[mask] = INF;
		for(I i = 0; i < n; i++)
		{
			I bit = (1LL<<i);
			I u = vertices[i];
			if(!(mask & bit))
			{
				continue;
			}
			I new_mask = mask ^ bit;
			I cost = dp[new_mask];
			for(I j = 0; j < n; j++)
			{
				if(i == j)
				{
					continue;
				}
				I bit2 = (1LL<<j);
				if(!(mask & bit2))
				{
					continue;
				}
				I w = vertices[j];
				P c = count_crossings(u, w, graph, index);
				cost += c.first; // We put u after w so we count pairs with u's neighbors comes first
			}
			dp[mask] = min(dp[mask], cost);	
		}
	}
	return dp[m];
}


// [TODO] Do a forget step before the next introduce.
// When forget v: T[X] = T'[X] + T'[XU{v}]
// Problem: X is a bitset. We want to add T'[X] and T'[XU{v}] to $T[X']$
// where X' is the mask of X in the new set resulting from removing v from X
// i.e. elements after v are shifted back by one.
// split pattern in two halfs before and after v, shift and concatenate.
// xor and or should yield the same answer since vs bit was zero.
// Iterate over all forgotten vertices and do this process one by one
// - Remove vertices from cut accordingly.
// i.e., after each step iterate over cut and check which vertices do not survive
// update them in the mentioned way and remove them.
// Q: can I update a batch  together?
// After forget all vertices I introduce the neighbors of v_i.

// Q: make sure that vi is handled well (not forgotten and reintroduced)
void run_solver(const VV& graph, const V& arrangement, const V& index, const VP& neighbor_range, const PP& parameters)
{
	I cutw = parameters.second.second;
	I dp_size = (1LL<<cutw);
	V sol[2] = {V(dp_size), V(dp_size)};
	V cut(graph.size());
	I cut_size = 0;
	bitset<15000> cut_mask; // this mask is over all vertices but later masks are over cut vertices only

	//swap after you get a new solution
	I curr_par = 0;
	I other_par = 1;

	for(const auto& v : arrangement)
	{
		I ind = index[v];

		// Forget vertices with no right neighbors
		for(I i = 0; i < cut_size;)
		{
			const auto& w = cut[i];
			if(index[w] > ind || neighbor_range[w].second > ind)
			{
				i++;
				continue;
			}

			// Forget w
			V& curr_sol = sol[curr_par];
			const V& last_sol = sol[other_par];
			I n_masks = count_masks(cut_size);
			curr_sol.assign(n_masks, INF);
			for(I mask = 0; mask < n_masks; mask++)
			{
				I new_mask = remove_index(mask, i);
				assert(new_mask < curr_sol.size());
				curr_sol[new_mask] = min(curr_sol[new_mask], last_sol[mask]);
			}
			swap(curr_par, other_par);
			// remove w from cut
			cut_mask[w] = 0;
			for(I j = i; j < cut_size -1; j++)
			{
				cut[j] = cut[j+1];
			}
			cut.pop_back();
			cut_size--;			
		}

		// Introduce v or its right neighbors if it has right neighbors
		// Note that either v or its neighbors are fixed partition vertices
		// so only one of them is in cut
		// anything introduced is suited anyway so just permute added vertices and append them.
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
		if(cut_size == prev_size)
		{
			continue;
		}
		// Update the DP table
		// For each partition of S, the new vertices of S come after old.
		// If some old vertices come after these, they will not be in partition.
		V& curr_sol = sol[curr_par];
		const V& last_sol = sol[other_par];
		curr_sol.assign(count_masks(cut_size), INF);
		I n_masks = count_masks(cut_size);
		I old_bits = count_masks(prev_size) - 1;
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
			curr_sol[mask] = last_sol[old_mask] + crossings_with_mask(cut, prev_size, graph, index) + best_permutation(added_vertices, graph, index);
		}
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
int main()
{
	ios_base::sync_with_stdio(0);
	V arrangement;
	VV graph;
	PP parameters = read_input(arrangement, graph);
	
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
	cout << "Neighbor range: " << neighbor_range << endl;
#endif
	run_solver(graph, arrangement, index, neighbor_range, parameters);
	return 0;
}
