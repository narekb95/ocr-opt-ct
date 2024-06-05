#include<bits/stdc++.h>

using namespace std;

using I = long long int;
using P = pair<I, I>;
using PP = pair<P, P>;

using V = vector<I>;
using VV = vector<V>;
using VP = vector<P>;
using VS = vector<string>;

using VV = vector<V>;
using VVP = vector<VP>;

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
	I i = 0, j =0 ;
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
PP read_input(V& arrangement, V&index, VV& graph)
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
			index.resize(n1+n2);
			continue;
		}
		assert(first_line);
		if(arrg_itr < n1+ n2)
		{
			assert(line.size() == 1);
			arrangement[arrg_itr] = stoi(line[0]) - 1;
			index[arrangement[arrg_itr]] = arrg_itr;
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


bool is_fixed_partition(const I& v, const PP& parameters)
{
	return v < parameters.first.first;
}

void compute_cut_vertices(const I& x, const I& ind, const V& arrangement, const V& index, const VV& graph, V& cut, const PP& parameters)
{
	cut.clear();
	for(I i = 0; i <= ind; i++)
	{
		I v = arrangement[i];
		bool v_in_cut  = 0;
		for(auto w : graph[v])
		{
			if(index[w] > ind)
			{
				v_in_cut = 1;
				if(!is_fixed_partition(w, parameters))
				{
					cut.push_back(w);
				}
			}
		}
		if(v_in_cut && !is_fixed_partition(v, parameters))
		{
			cut.push_back(v);
		}
	}
}
void solve(const VV& graph, const V& arrangement, const V& index, const PP& parameters)
{
	I cutw = parameters.second.second;
	I dp_size = 1LL<<cutw;
	V sol[2] = {V(dp_size), V(dp_size)};
	V cut;
	for(const auto& v : arrangement)
	{
		I ind = index[v];
		compute_cut_vertices(v, ind, arrangement, index, graph, cut, parameters);
		I c = cut.size();
		I n_masks = 1LL<<c;
		for(I b_msk = 0; b_msk < n_masks; b_msk++)
		{
			V vertices;
			for(I i = 0; i < c; i++)
			{
				if((1LL<<i)&b_msk)
				{
					vertices.push_back(cut[i]);
				}
			}
			// todo update solutions at sol[ind%2]
			I ind_par = ind & 1;
			I last_par = 1 - ind_par;
			V& curr_sol = sol[ind_par];
			const V& last_sol = sol[last_par];
			fill(curr_sol.begin(), curr_sol.end(), INFINITY);

		}
	}

}

int main()
{
	//Make IO fast
	ios_base::sync_with_stdio(0);

	V arrangement;
	V index;
	VV graph;
	PP parameters = read_input(arrangement, index, graph);

	solve(graph, arrangement, index, parameters);
	return 0;
}
