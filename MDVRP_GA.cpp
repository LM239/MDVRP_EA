#include <iostream>
#include <fstream>
#include <sstream>
#include <string.h>
#include <vector>
#include <cmath> 
#include <chrono>
#include <algorithm>
#include <limits>
#include <stdlib.h>
#include <numeric>
#include <random>
#include <thread>

using namespace std;

#define POP_SIZE 300
#define CROSSOVER_P 0.75
#define INTRA_MUTAION_P 0.5
#define INTER_MUTATION_FREQ 10
#define ELITISM_COUNT 5
#define GENS 135
#define SWAP_DIST 2.0
#define SWAP_FREQ 75
#define PHASE2_ITERATIONS 2
#define TOURNAMENT_GREED 0.8
#define POPULATIONS 1
#define MIGRATION_FREQ 75
#define TOUR_SIZE 200
#define TOUR_GREED 0.8
#define NON_TOUR_MUTATION_P 0.9
#define MAX_LENGTH_PUNISHMENT 1.4


struct route {
    vector<unsigned int> path;
    unsigned int load;
    float length; 
};

struct depot {
    const unsigned int id;
    const signed int x;
    const signed int y;
};

struct customer {
    unsigned int id;
    signed int x;
    signed int y;
    unsigned int load;
    vector<unsigned int> swap_depots;
};


unsigned int max_vehicles;
unsigned int n_customers;
unsigned int num_depots;
vector<depot> depots;
vector<customer> customers;
vector<vector<unsigned int>> chromosomes;
vector<vector<float>> distances;
unsigned int max_load;
float max_length;
string problem;

vector<vector<vector<vector<customer>>>> populations;
vector<vector<vector<customer>>> init_population(mt19937& rng);
void insert_missing(vector<vector<route>>& child_routes, vector<vector<customer>>& child_inst, vector<customer>& additions, int depot, mt19937& rng);
void crossover(vector<vector<route>>& p1_routes, vector<vector<route>>& p2_routes, vector<vector<customer>>& p1_inst, vector<vector<customer>>& p2_inst, mt19937& rng);
void random_mutation(vector<vector<customer>>& instance, vector<vector<route>>& route_inst, mt19937& rng);
void inter_mutation(vector<vector<customer>>& instance, mt19937& rng);
void get_route(vector<vector<route>>& new_routes, vector<vector<customer>>& instance);
float get_fitness(vector<vector<route>>& route_inst);
void print_instance(vector<vector<route>>& instance, float fitness);


void visualize() {
    string cmd = "python visualize.py " + problem;
    system(cmd.c_str());
}

string trim(string str) {
	const char* trim = " \t\n\r\f\v ";
	str.erase(str.find_last_not_of(trim) + 1);
	str.erase(0,str.find_first_not_of(trim));
	return str;
}

float randFloat(const float& low, const float& high, mt19937& rng) {
    uniform_real_distribution<float> urd(low, high);
    return urd(rng);
}

int randInt(const int& low, const int& high, mt19937& rng) {
    uniform_int_distribution<int> urd(low, high);
    return urd(rng);
}

vector<signed int> split(string &s, char split) {     
    vector<signed int> tokens;     
    string token;
    istringstream tokenStream(s);     
    while (getline(tokenStream, token, split)) { 
        token = trim(token);
        if (token == "") continue;     
        tokens.push_back(stoi(token));     
    }     
    return tokens;  
}

template<class T = std::mt19937, std::size_t N = T::state_size * sizeof(typename T::result_type)>
auto ProperlySeededRandomEngine () -> typename std::enable_if<N, T>::type {
    std::random_device source;
    std::random_device::result_type random_data[(N - 1) / sizeof(source()) + 1];
    std::generate(std::begin(random_data), std::end(random_data), [&](){return source();});
    std::seed_seq seeds(std::begin(random_data), std::end(random_data));
    return T(seeds);
}

float euclidean_dist(int x1, int x2, int y1, int y2) {
    return sqrt(pow((x1 - x2), 2) + pow((y1 - y2), 2));
}


void train_population(int p_id, vector<int> migrate_dests) {
    static thread_local mt19937* rng_p = nullptr;
    unsigned long long now = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count();
    if (!rng_p) rng_p = new mt19937(now + p_id);
    static mt19937 rng = *rng_p;
    vector<vector<vector<customer>>> population = init_population(rng);
    vector<vector<vector<customer>>> prev_best_is = vector<vector<vector<customer>>>(ELITISM_COUNT);
    vector<vector<vector<route>>> prev_best_is_route = vector<vector<vector<route>>>(ELITISM_COUNT);
    vector<float> prev_fitnesses;
    vector<vector<vector<route>>> routes = vector<vector<vector<route>>>(POP_SIZE);
    vector<float> fitnesses = vector<float>(POP_SIZE);
    vector<int> n_best_i = vector<int>(ELITISM_COUNT, 0);
    vector<float> n_best_fitnesses = vector<float>(ELITISM_COUNT, numeric_limits<float>::max());
    vector<int> n_worst_i = vector<int>(ELITISM_COUNT, 0);
    vector<float> n_worst_fitnesses = vector<float>(ELITISM_COUNT, 0.0);
    vector<int> tour_matchups = vector<int>(POP_SIZE);
    vector<int> new_instances = vector<int>(POP_SIZE);
    vector<int> parents = vector<int>();
    parents.reserve(TOUR_SIZE / 2);
    iota(tour_matchups.begin(), tour_matchups.end(), 0);
    iota(new_instances.begin(), new_instances.end(), 0);
    populations[p_id] = population;
    for (int i = 0; i < POP_SIZE; i++) {
        routes[i] = vector<vector<route>>(num_depots);
        get_route(routes[i], population[i]);
        fitnesses[i] = get_fitness(routes[i]);
        int test;
        while (fitnesses[i] > 1000000000.0) {
            vector<int> depot_loads = vector<int>(num_depots);
            int max_depot_load = 0;
            int max_depot;
            for (int d = 0; d < num_depots; d++) {
                int total_load = 0;
                for (int c = 0; c < population[i][d].size(); c++) {
                    total_load += population[i][d][c].load;
                }
                depot_loads[d] = total_load;
                if (total_load > max_depot_load) {
                    max_depot_load = total_load;
                    max_depot = d;
                }
            }
            int lowest_load = 2000000000;
            int least_diff = 2000000000;
            int best_swap_depot;
            int best_swap_cust_index;
            for (int c = 0; c < population[i][max_depot].size(); c++) {
                for (int sd = 0; sd < population[i][max_depot][c].swap_depots.size(); sd++) {
                    int cand_depot = population[i][max_depot][c].swap_depots[sd];
                    if (cand_depot == max_depot) continue;
                    int diff = max_depot_load - 2 * population[i][max_depot][c].load - depot_loads[cand_depot];
                    if (diff < least_diff && diff >= 0) {
                        least_diff = diff;
                        best_swap_depot = cand_depot;
                        best_swap_cust_index = c;
                        if (diff == 0) lowest_load = depot_loads[cand_depot];
                    } else if (-diff < least_diff && diff < 0) {
                        best_swap_depot = cand_depot;
                        least_diff = -diff;
                        best_swap_cust_index = c;
                    }
                    if (least_diff == 0 && diff == 0 && depot_loads[cand_depot] < lowest_load) {
                        least_diff = diff;
                        best_swap_depot = cand_depot;
                        best_swap_cust_index = c;
                        lowest_load = depot_loads[cand_depot];
                    }
                }
            }
            population[i][best_swap_depot].insert(population[i][best_swap_depot].end(), population[i][max_depot][best_swap_cust_index]);
            population[i][max_depot].erase(population[i][max_depot].begin() + best_swap_cust_index);
            for (int d = 0; d < num_depots; d++) {
                shuffle(population[i][d].begin(), population[i][d].end(), rng);
            }
            get_route(routes[i], population[i]);
            fitnesses[i] = get_fitness(routes[i]);
            int test = population[i][1][1].id;
        }
    }

    for (int gen = 0; gen < GENS; gen++) {

        shuffle(tour_matchups.begin(), tour_matchups.end(), rng);
        for (int t = 0; t < TOUR_SIZE - 1; t+=2) {
            int p1 = tour_matchups[t];
            int p2 = tour_matchups[t + 1];
            float p = randFloat(0.0, 1.0, rng);
            if (p < TOURNAMENT_GREED) {
                if (fitnesses[p1] < fitnesses[p2]) {
                    parents.push_back(p1);
                    continue;
                }
                parents.push_back(p2);
            } else {
                p = randFloat(0.0, 1.0, rng);
                if (p < 0.5 && fitnesses[p1] < 1000000000.0) {
                    parents.push_back(p1);
                    continue;
                }
                parents.push_back(p2);
            }
        }
        for (int p = 0; p < parents.size() - 1; p += 2) {
            if (fitnesses[parents[p]] > 10000000000.0 || fitnesses[parents[p + 1]] > 10000000000.0) continue;
            crossover(routes[parents[p]], 
                    routes[parents[p + 1]],
                    population[parents[p]],
                    population[parents[p + 1]],
                    rng);
            fitnesses[parents[p]] = get_fitness(routes[parents[p]]);
            fitnesses[parents[p + 1]] = get_fitness(routes[parents[p + 1]]);
        }
               if (gen > 0) {
            for (int i = 0; i < population.size(); i++) {
                fitnesses[i] = get_fitness(routes[i]);
            }
       }
        if (!(gen % INTER_MUTATION_FREQ)) {
            for (int t = TOUR_SIZE; t < POP_SIZE; t++) {
                float p = randFloat(0.0, 1.0, rng);
                if (p < INTRA_MUTAION_P) {
                    inter_mutation(population[tour_matchups[t]], rng);
                    get_route(routes[tour_matchups[t]], population[tour_matchups[t]]);
                    fitnesses[tour_matchups[t]] = get_fitness(routes[tour_matchups[t]]);
                }
            }
        } else {
            for (int t = TOUR_SIZE; t < POP_SIZE; t++) {
                float p = randFloat(0.0, 1.0, rng);
                if (p < NON_TOUR_MUTATION_P && fitnesses[tour_matchups[t]] < numeric_limits<float>::max()) {
                    random_mutation(population[tour_matchups[t]], routes[tour_matchups[t]], rng);
                    fitnesses[tour_matchups[t]] = get_fitness(routes[tour_matchups[t]]);
                }
            }
        }

        double sum = 0.0;
        for (int i = 0; i < POP_SIZE; i++) {
            sum += fitnesses[i];
            if (fitnesses[i] < n_best_fitnesses[ELITISM_COUNT - 1]) {
                signed int index = ELITISM_COUNT - 2;
                while(index >= 0 && fitnesses[i] < n_best_fitnesses[index]) {
                    index--;
                }
                index++;
                n_best_i.erase(n_best_i.end() - 1);
                n_best_fitnesses.erase(n_best_fitnesses.end() - 1);
                n_best_i.insert(n_best_i.begin() + index, i);
                n_best_fitnesses.insert(n_best_fitnesses.begin() + index, fitnesses[i]);
            } 
            if (fitnesses[i] > n_worst_fitnesses[ELITISM_COUNT - 1] && gen > 0) {
                signed int index = ELITISM_COUNT - 2;
                while(index >= 0 && fitnesses[i] > n_worst_fitnesses[index]) {
                    index--;
                }
                index++;
                n_worst_i.erase(n_worst_i.end() - 1);
                n_worst_fitnesses.erase(n_worst_fitnesses.end() - 1);
                n_worst_i.insert(n_worst_i.begin() + index, i);
                n_worst_fitnesses.insert(n_worst_fitnesses.begin() + index, fitnesses[i]);
            }
        }
        if (gen > 0) {
            for (int n = 0; n < ELITISM_COUNT; n++) {
                population[n_worst_i[n]] = prev_best_is[n];
                routes[n_worst_i[n]] = prev_best_is_route[n];

                fitnesses[n_worst_i[n]] = prev_fitnesses[n];
                sum -= n_worst_fitnesses[n] - prev_fitnesses[n];

                if (prev_fitnesses[n] < n_best_fitnesses[ELITISM_COUNT - 1]) {
                    signed int index = ELITISM_COUNT - 2;
                    while(index >= 0 && prev_fitnesses[n] < n_best_fitnesses[index]) {
                        index--;
                    }
                    index++;
                    n_best_i.erase(n_best_i.end() - 1);
                    n_best_fitnesses.erase(n_best_fitnesses.end() - 1);
                    n_best_i.insert(n_best_i.begin() + index, n_worst_i[n]);
                    n_best_fitnesses.insert(n_best_fitnesses.begin() + index, prev_fitnesses[n]);
                } 
            }
        }
        for (int n = 0; n < ELITISM_COUNT; n++) {
            prev_best_is[n] = population[n_best_i[n]];
            prev_best_is_route[n] = routes[n_best_i[n]];
        }
        prev_fitnesses = n_best_fitnesses;
        if (!(gen % 10)) cout << "Gen: " << gen << " Min fitness: " << n_best_fitnesses[0] << " Parent size :/  :" << parents.size() << endl;
        fill(n_best_fitnesses.begin(), n_best_fitnesses.end(), numeric_limits<float>::max());
        fill(n_worst_fitnesses.begin(), n_worst_fitnesses.end(), 0.0);
    }
    print_instance(routes[n_best_i[0]], fitnesses[n_best_i[0]]);
    visualize();
    cout  << "Hello, World!";
}



float get_fitness(vector<vector<route>>& route_inst) {
    float fitness = 0;
    for (int d = 0; d < num_depots; d++) {
        if (route_inst[d].size() > max_vehicles && max_vehicles > 0) return numeric_limits<float>::max();
        for (int r = 0; r < route_inst[d].size(); r++) {
            fitness += route_inst[d][r].length > max_length ? MAX_LENGTH_PUNISHMENT * route_inst[d][r].length : route_inst[d][r].length;
        }
    }
    return fitness;
}

void inter_mutation(vector<vector<customer>>& instance, mt19937& rng) {
    int s_d = randInt(0, num_depots - 1, rng);
    int s_c = randInt(0, instance[s_d].size() - 1, rng);
    int count = 0;
    while (instance[s_d][s_c].swap_depots.size() == 1) {
        count ++;
        if (count > 5) return;
        s_d = randInt(0, num_depots - 1, rng);
        s_c = randInt(0, instance[s_d].size() - 1, rng);
    }
    customer cust = instance[s_d][s_c];
    int t_d = randInt(0, cust.swap_depots.size() - 1, rng);
    while (s_d == cust.swap_depots[t_d]) {
        t_d = randInt(0, cust.swap_depots.size() - 1, rng);
    }
    int t_c = randInt(0, instance[cust.swap_depots[t_d]].size(), rng);
    instance[cust.swap_depots[t_d]].insert(instance[cust.swap_depots[t_d]].begin() + t_c, cust);
    instance[s_d].erase(instance[s_d].begin() + s_c);
}

void crossover(vector<vector<route>>& p1_routes, vector<vector<route>>& p2_routes, vector<vector<customer>>& p1_inst, vector<vector<customer>>& p2_inst, mt19937& rng) {
    int orig_d = randInt(0, num_depots - 1, rng);
    int r1 = randInt(0, p1_routes[orig_d].size() - 1, rng);
    int r2 = randInt(0, p2_routes[orig_d].size() - 1, rng);

    vector<vector<customer>> c1_inst = vector<vector<customer>>(num_depots);
    vector<vector<customer>> c2_inst = vector<vector<customer>>(num_depots);

    vector<customer> c1_missing = vector<customer>();
    vector<customer> c2_missing = vector<customer>();
    c1_missing.reserve(p2_routes[orig_d][r2].path.size());
    c2_missing.reserve(p1_routes[orig_d][r1].path.size());
    
    vector<vector<route>> c1_routes = vector<vector<route>>(num_depots);
    vector<vector<route>> c2_routes = vector<vector<route>>(num_depots);

    for (int d = 0; d < num_depots; d++) {
        c1_inst[d].reserve(p1_inst[d].size());
        c2_inst[d].reserve(p2_inst[d].size());
        c1_routes[d].reserve(p1_routes[d].size());
        c2_routes[d].reserve(p2_routes[d].size());
        for (int r = 0; r < p1_routes[d].size(); r++) {
            int num_removed = 0;

            route new_route;
            new_route.path = vector<unsigned int>();
            new_route.load = 0;
            new_route.length = 0;
            for (int c = 0; c < p1_routes[d][r].path.size(); c++) {
                bool removed = false;
                for (int r2_i = 0; r2_i < p2_routes[orig_d][r2].path.size(); r2_i++) {
                    if (p2_routes[orig_d][r2].path[r2_i] == p1_routes[d][r].path[c]) {
                        c1_missing.push_back(customers[p1_routes[d][r].path[c]]);
                        removed = true;
                        num_removed++;
                        break;
                    }
                }
                if (!removed) {
                    customer cust = customers[p1_routes[d][r].path[c]];
                    c1_inst[d].push_back(cust);
                    new_route.load += cust.load;
                    if (c - num_removed > 0) {
                        new_route.length += distances[new_route.path[c - num_removed - 1]][cust.id];
                    } else {
                        new_route.length = distances[n_customers + d][cust.id];
                    }
                    new_route.path.push_back(p1_routes[d][r].path[c]);
                }
            }
            if (new_route.length > 0) {
                new_route.length += distances[n_customers + d][new_route.path[new_route.path.size() - 1]];
                c1_routes[d].push_back(new_route);
            }
        }
        for (int r = 0; r < p2_routes[d].size(); r++) {
            int num_removed = 0;

            route new_route;
            new_route.path = vector<unsigned int>();
            new_route.load = 0;
            new_route.length = 0;
            for (int c = 0; c < p2_routes[d][r].path.size(); c++) {
                bool removed = false;
                for (int r1_i = 0; r1_i < p1_routes[orig_d][r1].path.size(); r1_i++) {
                    if (p1_routes[orig_d][r1].path[r1_i] == p2_routes[d][r].path[c]) {
                        c2_missing.push_back(customers[p2_routes[d][r].path[c]]);
                        removed = true;
                        num_removed++;
                        break;
                    }
                }
                if (!removed) {
                    customer cust = customers[p2_routes[d][r].path[c]];
                    c2_inst[d].push_back(cust);
                    new_route.load += cust.load;
                    if (c - num_removed > 0) {
                        new_route.length += distances[new_route.path[c - num_removed - 1]][cust.id];
                    } else {
                        new_route.length = distances[n_customers + d][cust.id];
                    }
                    new_route.path.push_back(p2_routes[d][r].path[c]);
                }
            }
            if (new_route.length > 0) {
                new_route.length += distances[n_customers + d][new_route.path[new_route.path.size() - 1]];
                c2_routes[d].push_back(new_route);
            }
        }
    }

    insert_missing(c1_routes, c1_inst, c1_missing, orig_d, rng);
    insert_missing(c2_routes, c2_inst, c2_missing, orig_d, rng);

    float p1_fitness = get_fitness(p1_routes);
    float p2_fitness = get_fitness(p2_routes);

    if (get_fitness(c1_routes) < p1_fitness) {
        p1_routes = c1_routes;
        p1_inst = c1_inst;
    }
    if (get_fitness(c2_routes) < p2_fitness) {
        p2_routes = c2_routes;
        p2_inst = c2_inst;
    }
}

void insert_missing(vector<vector<route>>& child_routes, vector<vector<customer>>& child_inst, vector<customer>& additions, int depot, mt19937& rng) {
    for (int c = 0; c < additions.size(); c++) {
        customer add_cust = additions[c];
        bool no_routes = true;
        int best_r;
        int best_c;
        float current_fitness = get_fitness(child_routes);
        float best_fitness = numeric_limits<float>::max();
        float best_length;
        for (int r = 0; r < child_routes[depot].size(); r++) {
            if (child_routes[depot][r].load + add_cust.load > max_load && max_load > 0) continue;
            no_routes = false;
            float old_length = child_routes[depot][r].length;
            for (int c_i = 0; c_i <= child_routes[depot][r].path.size(); c_i++) {
                float new_length;
                int code;
                if (c_i == 0) {
                    if (child_routes[depot][r].path.size() == 0) {
                        new_length = 2 * distances[n_customers + depot][add_cust.id];
                    } else { 
                        new_length = old_length 
                                - distances[n_customers + depot][child_routes[depot][r].path[0]]
                                + distances[n_customers + depot][add_cust.id]
                                + distances[add_cust.id][child_routes[depot][r].path[0]];
                    }
                                code = 1;
                } else if (c_i == child_routes[depot][r].path.size()) {
                    new_length = old_length
                                - distances[n_customers + depot][child_routes[depot][r].path[c_i - 1]]
                                + distances[n_customers + depot][add_cust.id]
                                + distances[add_cust.id][child_routes[depot][r].path[c_i - 1]];
                                code = 2;
                } else {
                    new_length = old_length
                                - distances[child_routes[depot][r].path[c_i - 1]][child_routes[depot][r].path[c_i]]
                                + distances[add_cust.id][child_routes[depot][r].path[c_i]]
                                + distances[add_cust.id][child_routes[depot][r].path[c_i - 1]];
                                code = 3;
                }
                if (current_fitness + new_length - old_length < best_fitness) {
                    best_fitness = current_fitness + new_length - old_length;
                    best_length = new_length;
                    best_r = r;
                    best_c = c_i;
                }
            }
        }
        if (!no_routes) {
            int inst_i = best_c;
            for (int i = 0; i < best_r; i++) {
                inst_i += child_routes[depot][i].path.size();
            }
            child_routes[depot][best_r].path.insert(child_routes[depot][best_r].path.begin() + best_c, add_cust.id);
            child_inst[depot].insert(child_inst[depot].begin() + inst_i, add_cust);
            child_routes[depot][best_r].load += add_cust.load;
            child_routes[depot][best_r].length = best_length;
        } else {
            int route_depot = depot;
            int count = 0;
            while (child_routes[route_depot].size() >= max_vehicles && max_vehicles > 0) {
                int d_i = randInt(0, add_cust.swap_depots.size() - 1, rng);
                route_depot = add_cust.swap_depots[d_i];
                count++;
                if (count > 4) break;
            }
            route new_route;
            new_route.load = add_cust.load;
            new_route.path = vector<unsigned int>();
            new_route.path.push_back(add_cust.id);
            new_route.length = 2 * distances[n_customers + route_depot][add_cust.id];
            child_routes[route_depot].push_back(new_route);
            child_inst[route_depot].push_back(add_cust);
            if (child_routes[route_depot].size() > max_vehicles && max_vehicles > 0) {
                return;
            }
        }
    }
}

void random_mutation(vector<vector<customer>>& instance, vector<vector<route>>& route_inst, mt19937& rng) {
    float p = randFloat(0.0, 1.0, rng);
    int orig_d = randInt(0, num_depots - 1, rng);
    if (p < (1.0 / 3.0)) {
        int count = 0;
        while ((int)instance[orig_d].size() == 1) {
            if (count > 3) return;
            count++;
            orig_d = randInt(0, num_depots - 1, rng);
        }
        int i1 = randInt(1, (int)instance[orig_d].size() - 1, rng);
        int i2 =  randInt(0, i1 - 1, rng);
        reverse(instance[orig_d].begin() + i2, instance[orig_d].begin() + i1);
        get_route(route_inst, instance);
    } else if (p > (2.0 / 3.0)) {
        int orig_r = randInt(0, (int) route_inst[orig_d].size() - 1, rng);
        while (route_inst[orig_d][orig_r].path.size() == 1) {
            orig_d = randInt(0, num_depots - 1, rng);
            orig_r = randInt(0, (int) route_inst[orig_d].size() - 1, rng);
        }
        int orig_c = randInt(0, (int) route_inst[orig_d][orig_r].path.size() - 1, rng);
        int inst_i = orig_c;
        customer moved_cust = customers[route_inst[orig_d][orig_r].path[orig_c]];
        for (int r = 0; r < orig_r; r++) {
            inst_i += route_inst[orig_d][r].path.size();
        }
        route_inst[orig_d][orig_r].load -= moved_cust.load;

        if (orig_c == 0) {
            route_inst[orig_d][orig_r].length = route_inst[orig_d][orig_r].length 
                        - distances[moved_cust.id][n_customers + orig_d]
                        - distances[moved_cust.id][route_inst[orig_d][orig_r].path[1]]
                        + distances[n_customers + orig_d][route_inst[orig_d][orig_r].path[1]];
        } else if (orig_c == route_inst[orig_d][orig_r].path.size() - 1) {
            route_inst[orig_d][orig_r].length = route_inst[orig_d][orig_r].length 
                        - distances[moved_cust.id][n_customers + orig_d]
                        - distances[moved_cust.id][route_inst[orig_d][orig_r].path[orig_c - 1]]
                        + distances[n_customers + orig_d][route_inst[orig_d][orig_r].path[orig_c - 1]];
        } else {
            route_inst[orig_d][orig_r].length = route_inst[orig_d][orig_r].length 
                        - distances[moved_cust.id][route_inst[orig_d][orig_r].path[orig_c + 1]]
                        - distances[moved_cust.id][route_inst[orig_d][orig_r].path[orig_c - 1]]
                        + distances[route_inst[orig_d][orig_r].path[orig_c - 1]][route_inst[orig_d][orig_r].path[orig_c + 1]];
        }
        route_inst[orig_d][orig_r].path.erase(route_inst[orig_d][orig_r].path.begin() + orig_c);
        instance[orig_d].erase(instance[orig_d].begin() + inst_i);

        int best_d;
        int best_r;
        int best_c;
        float current_fitness = get_fitness(route_inst);
        float best_fitness = numeric_limits<float>::max();
        float best_length;
        int final_code;
        for (int d = 0; d < num_depots; d++) {
            for (int r = 0; r < route_inst[d].size(); r++) {
                if (route_inst[d][r].load + moved_cust.load > max_load && max_load > 0) continue;
                float old_length = route_inst[d][r].length;
                for (int c = 0; c <= route_inst[d][r].path.size(); c++) {
                    float new_length;
                    if (c == 0) {
                        new_length = old_length 
                                    - distances[n_customers + d][route_inst[d][r].path[0]]
                                    + distances[moved_cust.id][n_customers + d]
                                    + distances[moved_cust.id][route_inst[d][r].path[0]];
                    } else if (c == route_inst[d][r].path.size()) {
                        new_length = old_length
                                    - distances[n_customers + d][route_inst[d][r].path[c - 1]]
                                    + distances[moved_cust.id][n_customers + d]
                                    + distances[moved_cust.id][route_inst[d][r].path[c - 1]];
                    } else {
                        new_length = old_length
                                    - distances[route_inst[d][r].path[c - 1]][route_inst[d][r].path[c]]
                                    + distances[moved_cust.id][route_inst[d][r].path[c]]
                                    + distances[moved_cust.id][route_inst[d][r].path[c - 1]];
                    }
                    if (current_fitness + new_length - old_length < best_fitness) {
                        best_fitness = current_fitness + new_length - old_length;
                        best_length = new_length;
                        best_d = d;
                        best_r = r;
                        best_c = c;
                    }
                }
            }
        }

        inst_i = best_c;
        for (int i = 0; i < best_r; i++) {
            inst_i += route_inst[best_d][i].path.size();
        }
        route_inst[best_d][best_r].path.insert(route_inst[best_d][best_r].path.begin() + best_c, moved_cust.id);
        instance[best_d].insert(instance[best_d].begin() + inst_i, moved_cust);
        route_inst[best_d][best_r].load += moved_cust.load;
        route_inst[best_d][best_r].length = best_length;
    } else {
        int count = 0;
        while (route_inst[orig_d].size() < 2) {
            if (count > 8) return;
            orig_d = randInt(0, num_depots - 1, rng);
            count++;
        }
        int r1 = randInt(0, route_inst[orig_d].size() - 1, rng);
        int r2 = randInt(0, route_inst[orig_d].size() - 1, rng);
        count = 0;
        int c_1i = randInt(0, route_inst[orig_d][r1].path.size() - 1, rng);
        int c_2i = randInt(0, route_inst[orig_d][r2].path.size() - 1, rng);
        while (r1 == r2 || route_inst[orig_d][r1].load + customers[route_inst[orig_d][r2].path[c_2i]].load - customers[route_inst[orig_d][r1].path[c_1i]].load > max_load && max_load > 0
               || route_inst[orig_d][r2].load + customers[route_inst[orig_d][r1].path[c_1i]].load - customers[route_inst[orig_d][r2].path[c_2i]].load > max_load && max_load > 0) {
            if(count > 3) {
                return;
            }
            r2 = randInt(0, route_inst[orig_d].size() - 1, rng);
            c_1i = randInt(0, route_inst[orig_d][r1].path.size() - 1, rng);
            c_2i = randInt(0, route_inst[orig_d][r2].path.size() - 1, rng);
            count++;
        }
        customer c1 = customers[route_inst[orig_d][r1].path[c_1i]];
        customer c2 = customers[route_inst[orig_d][r2].path[c_2i]];

        count = 0;
        float new_r1_length = route_inst[orig_d][r1].length;
        float new_r2_length = route_inst[orig_d][r2].length;
        while (true) {
            if (route_inst[orig_d][r1].path.size() == 1) {
                new_r1_length +=
                            2 * distances[n_customers + orig_d][c2.id]
                            - 2 * distances[n_customers + orig_d][c1.id];
            }
            else if (c_1i == 0) {
                new_r1_length += 
                            distances[c2.id][n_customers + orig_d]
                            + distances[c2.id][route_inst[orig_d][r1].path[1]]
                            - distances[c1.id][n_customers + orig_d]
                            - distances[c1.id][route_inst[orig_d][r1].path[1]];
            } else if (c_1i == route_inst[orig_d][r1].path.size() - 1) {
                new_r1_length +=  
                            distances[c2.id][n_customers + orig_d]
                            + distances[c2.id][route_inst[orig_d][r1].path[c_1i - 1]]
                            - distances[c1.id][n_customers + orig_d]
                            - distances[c1.id][route_inst[orig_d][r1].path[c_1i - 1]];
            } else {
                new_r1_length += 
                            distances[c2.id][route_inst[orig_d][r1].path[c_1i + 1]]
                            + distances[c2.id][route_inst[orig_d][r1].path[c_1i - 1]]
                            - distances[c1.id][route_inst[orig_d][r1].path[c_1i - 1]]
                            - distances[c1.id][route_inst[orig_d][r1].path[c_1i + 1]];
            }
            if (route_inst[orig_d][r2].path.size() == 1) {
                new_r2_length +=
                            2 * distances[n_customers + orig_d][c1.id]
                            - 2 * distances[n_customers + orig_d][c2.id];
            }
            else if (c_2i == 0) {
               new_r2_length +=
                            distances[c1.id][n_customers + orig_d]
                            + distances[c1.id][route_inst[orig_d][r2].path[1]]
                            - distances[c2.id][n_customers + orig_d]
                            - distances[c2.id][route_inst[orig_d][r2].path[1]];
            } else if (c_2i == route_inst[orig_d][r2].path.size() - 1) {
                new_r2_length +=
                            distances[c1.id][n_customers + orig_d]
                            + distances[c1.id][route_inst[orig_d][r2].path[c_2i - 1]]
                            - distances[c2.id][n_customers + orig_d]
                            - distances[c2.id][route_inst[orig_d][r2].path[c_2i - 1]];
            } else {
                new_r2_length +=
                            distances[c1.id][route_inst[orig_d][r2].path[c_2i - 1]]
                            + distances[c1.id][route_inst[orig_d][r2].path[c_2i + 1]]
                            - distances[c2.id][route_inst[orig_d][r2].path[c_2i + 1]]
                            - distances[c2.id][route_inst[orig_d][r2].path[c_2i - 1]];
            }
            count++;
            if (new_r1_length <= max_length && new_r2_length <= max_length || count > 4) {
                break;
            } else {
                r2 = randInt(0, route_inst[orig_d].size() - 1, rng);
                c_1i = randInt(0, route_inst[orig_d][r1].path.size() - 1, rng);
                c_2i = randInt(0, route_inst[orig_d][r2].path.size() - 1, rng);
                while (r1 == r2 || route_inst[orig_d][r1].load + customers[route_inst[orig_d][r2].path[c_2i]].load - customers[route_inst[orig_d][r1].path[c_1i]].load > max_load && max_load > 0
               || route_inst[orig_d][r2].load + customers[route_inst[orig_d][r1].path[c_1i]].load - customers[route_inst[orig_d][r2].path[c_2i]].load > max_load && max_load > 0) {
                    if(count > 3) {
                        return;
                    }
                    r2 = randInt(0, route_inst[orig_d].size() - 1, rng);
                    c_1i = randInt(0, route_inst[orig_d][r1].path.size() - 1, rng);
                    c_2i = randInt(0, route_inst[orig_d][r2].path.size() - 1, rng);
                    count++;
                }
                c1 = customers[route_inst[orig_d][r1].path[c_1i]];
                c2 = customers[route_inst[orig_d][r2].path[c_2i]];
            }
            new_r1_length = route_inst[orig_d][r1].length;
            new_r2_length = route_inst[orig_d][r2].length;
        }

        int inst_i1 = c_1i;
        for (int i = 0; i < r1; i++) {
            inst_i1 += route_inst[orig_d][i].path.size();
        }
        int inst_i2 = c_2i;
        for (int i = 0; i < r2; i++) {
            inst_i2 += route_inst[orig_d][i].path.size();
        }
        route_inst[orig_d][r1].length = new_r1_length;
        route_inst[orig_d][r2].length = new_r2_length;
        route_inst[orig_d][r2].load += c1.load - c2.load;
        route_inst[orig_d][r1].load += c2.load - c1.load;
        instance[orig_d][inst_i1] = c2;
        instance[orig_d][inst_i2] = c1;
        route_inst[orig_d][r1].path[c_1i] = c2.id;
        route_inst[orig_d][r2].path[c_2i] = c1.id;
    }
}


void get_route(vector<vector<route>>& new_routes, vector<vector<customer>>& instance) {
    for (int d = 0; d < num_depots; d++) {
        new_routes[d] = vector<route>();
        new_routes[d].reserve(max_vehicles);
        vector<unsigned int> current_path = vector<unsigned int>();
        unsigned int current_load = 0;
        float current_length = 0.0;
        unsigned int new_load = instance[d][0].load;
        float new_length = distances[n_customers + d][instance[d][0].id];
        for (int c = 1; c <= instance[d].size() + 1; c++) {
            if (current_load + new_load > max_load && max_load > 0 || c > instance[d].size()) {
                route new_route;
                new_route.path = current_path;
                new_route.load = current_load;
                new_route.length = current_length + distances[n_customers + d][instance[d][c - 2].id];
                new_routes[d].push_back(new_route);
                if (c > instance[d].size()) break;
                
                current_path = vector<unsigned int>();
                current_load = 0;
                current_length = 0.0;
                new_length = distances[n_customers + d][instance[d][c - 1].id];
            }
            current_load += new_load;
            current_length += new_length;
            current_path.push_back(instance[d][c - 1].id);
            if (c < instance[d].size()) {
                new_load = instance[d][c].load;
                new_length = distances[instance[d][c - 1].id][instance[d][c].id];
            }
        }
        bool changed = true;
        for (int k = 0; k < PHASE2_ITERATIONS; k++) {
            if (changed) {
                for (int r = new_routes[d].size() - 1; r > 0; r--) {
                    changed = false;
                    if (new_routes[d][r - 1].path.size() < 2) continue;
                    customer last_customer = customers[new_routes[d][r - 1].path[new_routes[d][r - 1].path.size() - 1]];

                    unsigned int new_load_r1 = new_routes[d][r - 1].load - last_customer.load;
                    unsigned int new_load_r2 = new_routes[d][r].load + last_customer.load;
                    if (new_load_r2 > max_load && max_load > 0) continue;
                    
                    float new_length_r1 = new_routes[d][r - 1].length 
                                        - distances[last_customer.id][n_customers + d]
                                        - distances[last_customer.id][new_routes[d][r - 1].path[new_routes[d][r - 1].path.size() - 2]]
                                        + distances[n_customers + d][new_routes[d][r - 1].path[new_routes[d][r - 1].path.size() - 2]];

                    float new_length_r2 = new_routes[d][r].length
                                        + distances[last_customer.id][n_customers + d]
                                        + distances[last_customer.id][new_routes[d][r].path[0]]
                                        - distances[n_customers + d][new_routes[d][r].path[0]];

                    if (new_length_r1 + new_length_r2 < new_routes[d][r - 1].length + new_routes[d][r].length) {
                        new_routes[d][r].length = new_length_r2;
                        new_routes[d][r].load = new_load_r2;
                        new_routes[d][r].path.insert(new_routes[d][r].path.begin(), last_customer.id);

                        new_routes[d][r - 1].length = new_length_r1;
                        new_routes[d][r - 1].load = new_load_r1;
                        new_routes[d][r - 1].path.erase(new_routes[d][r - 1].path.end() - 1);
                        changed = true;                       
                    }
                }
            }
        }
    }
}

vector<vector<vector<customer>>> init_population(mt19937& rng) {
    vector<vector<vector<customer>>> population;
    population.reserve(POP_SIZE);
    for (int i = 0; i < POP_SIZE; i++) {
        population.push_back(vector<vector<customer>>(num_depots));
        for (int d = 0; d < num_depots; d++) {
            shuffle(chromosomes[d].begin(), chromosomes[d].end(), rng);
            vector<customer> new_customers = vector<customer>();
            new_customers.reserve(chromosomes[d].size());
            for (int k = 0; k < chromosomes[d].size(); k++) {
                new_customers.push_back(customers[chromosomes[d][k]]);
            }
            population[i][d] = new_customers;
        }
    }
    return population;
}

int main(int argc, char* argv[])
{   
    if (argc == 2) {
        problem = argv[1];
    } else {
        problem = "p23";
    }
    string text, dump;
    ifstream data("data_files\\" + problem);
    if (data.is_open())
    {
        getline(data, text);
        vector<signed int> line1 = split(text, ' ');
        max_vehicles = line1[0];
        n_customers = line1[1];
        num_depots = line1[2];
        depots.reserve(num_depots);
        for (short d = 0; d < num_depots; d++) {
            getline(data, text);
        }
        line1 = split(text, ' ');
        max_load = line1[1];
        max_length = (float) line1[0];
        if (max_length == 0.0) max_length = numeric_limits<float>::max();
        customers.reserve(n_customers);
        for (unsigned int c = 0; c < n_customers; c++) {
            getline(data, text);
            line1 = split(text, ' ');
            const struct customer new_customer {
                c,
                line1[1],
                line1[2],
                (unsigned int) line1[4],
                vector<unsigned int>(),
            };
            customers.push_back(new_customer);
        }
        depots.reserve(num_depots);
        for (unsigned int d = 0; d < num_depots; d++) {
            getline(data, text);
            line1 = split(text, ' ');
            const struct depot new_depot {
                d,
                line1[1],
                line1[2]
            };
            depots.push_back(new_depot);
        }
        data.close();
    }

    distances.reserve(num_depots + n_customers);
    chromosomes.reserve(num_depots);
    for (int d = 0; d < num_depots; d++) {
        chromosomes.push_back(vector<unsigned int>());
    }
    for (int i = 0; i < num_depots + n_customers; i++) {
        distances.push_back(vector<float>(num_depots + n_customers));
    }
    for (int c = 0; c < n_customers; c++) {
        int x1 = customers[c].x;
        int y1 = customers[c].y;

        int best_depot;
        float best_dist = numeric_limits<float>::max();
        for (int d = 0; d < num_depots; d++) {
            const float dist = euclidean_dist(x1, depots[d].x, y1, depots[d].y);
            distances[c][n_customers + d] = dist;
            distances[n_customers + d][c] = dist;
            if (dist < best_dist) {
                best_dist = dist;
                best_depot = d;
            }
        }
        chromosomes[best_depot].push_back(c);
        for (int d = 0; d < num_depots; d++) {
            float dist = distances[c][n_customers + d];
            if ((dist - best_dist) / best_dist <= SWAP_DIST) {
                customers[c].swap_depots.push_back(d);
            }
        }
        for (int j = 0; j < n_customers; j++) {
            int x2 = customers[j].x;
            int y2 = customers[j].y;

            float dist = euclidean_dist(x1, x2, y1, y2);
            distances[c][j] = dist;
        }
    }
    populations = vector<vector<vector<vector<customer>>>>(POPULATIONS);
    vector<int> my_dests = vector<int>(GENS / MIGRATION_FREQ);
    int p_id = 0;
    train_population(p_id, my_dests);
    /*
    vector<vector<int>> migration_dest_vals = vector<vector<int>>(GENS / MIGRATION_FREQ);
    for (int p = 0; p < GENS / MIGRATION_FREQ; p++) {
        vector<int> range = vector<int>(POPULATIONS);
        iota(range.begin(), range.end(), 0);
        for (int i = range.size() - 1; i > 0; i--) {
            int j = rand_r(unsigned(time(NULL))) % i;
            int temp = range[i];
            range[i] = range[j];
            range[j] = temp;
        }
        migration_dest_vals[p] = range;
    }
    for (int p = 0; p < POPULATIONS; p++) {
        vector<int> my_dests = vector<int>(GENS / MIGRATION_FREQ);
        for (int d = 0; d < GENS / MIGRATION_FREQ; d++) {
            my_dests[d] = migration_dest_vals[d][p];
        }
        train_population(p, my_dests, rand());
    }
    */
}

string pad_int(int num) {
    if (num < 0) {
        if (num < -9) return " " + to_string(num);
        return "  " + to_string(num);
    }
    if (num < 10) return "   " + to_string(num);
    if (num > 99) return " " + to_string(num);
    return "  " + to_string(num);
}

string pad_float(float num) {
    string text = to_string(num);
    if (num < 10) return "   " + text.substr(0, text.find(".")+3);;
    if (num > 99) return " " + text.substr(0, text.find(".")+3);;
    return "  " + text.substr(0, text.find(".")+3);;
}


void print_instance(vector<vector<route>>& instance, float fitness) {
    ofstream file = ofstream("solution_files/" + problem + ".txt");
    cout << "Max_length: " << max_length << endl;
    cout << fitness << endl;
    file << fitness << endl;
    for (int d = 0; d < num_depots; d++) {
        for (int r = 0; r < instance[d].size(); r++) {
            string out = pad_int(d + 1) + pad_int(r + 1) + pad_float(instance[d][r].length) + pad_int(instance[d][r].load) + "  0";
            for (int c = 0; c < instance[d][r].path.size(); c++) {
                out += pad_int(instance[d][r].path[c] + 1);
            }
            out += "  0";
            cout << out;
            if (instance[d][r].length > max_length) cout << " Exceeded max_length: " << max_length;
            cout << endl;
            file << out << endl;
        }
    }
    file.close();
}

/*
    for (int i = 0; i < POP_SIZE; i++) {
        for (int d = 0; d < num_depots; d++){
            for (int r = 0; r < routes[i][d].size(); r++) {
                int load = 0;
                float length = distances[n_customers + d][routes[i][d][r].path[0]];
                for (int c = 1; c <= routes[i][d][r].path.size(); c++) {
                    load += customers[routes[i][d][r].path[c - 1]].load;
                    if (c < routes[i][d][r].path.size()) {
                        length += distances[routes[i][d][r].path[c - 1]][routes[i][d][r].path[c]];
                    } else {
                        length += distances[routes[i][d][r].path[c - 1]][n_customers + d];
                    }
                }
                calc_fitness += length;
                if (routes[i][d][r].load != load) {
                    cout << "Load: " << routes[i][d][r].load << " Calculated: " << load << endl;
                }
                if (routes[i][d][r].length - length < -0.001 || routes[i][d][r].length - length > 0.001) {
                    cout << "Length: " << routes[i][d][r].length << " Calculated: " << length << endl;
                }
            }
        }
    }
*/
