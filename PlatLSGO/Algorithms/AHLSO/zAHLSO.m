function [gbestx, bestever, gbesthistory] = zAHLSO(popsize, dimension, xmax, xmin, vmax, vmin, maxiter, fCalculation, FuncId)
% AHLSO - Agent-assisted Heterogeneous Learning Swarm Optimizer
% Based on the paper: "An agent-assisted heterogeneous learning swarm optimizer 
%            for large-scale optimization" (2024)
% 
% Main innovations:
% 1. Heterogeneous learning structure (HL): local search + global search
% 2. Agent-assisted evolutionary search: Q-learning dynamically selects the learning structure
% 3. Level-based resource allocation (LRA): dynamically allocates computational resources
% 4. Level-based sample selection (LSS): prevents premature convergence

%% ================ Parameter settings ================
popsize = 500;          % population size (as recommended in the paper)
phi = 0.35;             % global search weight (paper Table 2)
alpha = 0.4;            % Q-learning learning rate
gamma = 0.8;            % Q-learning discount factor
epsilon = 0.6;          % epsilon-greedy exploration rate

% Set of learning structures (paper Algorithm 1, line 3)
S_local = [4, 6, 8, 10, 20, 50];    % set of local search levels
S_global = [4, 6, 8, 10];           % set of global search levels

% State space: 4 discrete states (paper Section 3.4.1)
n_states = 4;

% Initialize the Q-tables (paper Algorithm 4, lines 1-5)
Q_local = zeros(n_states, length(S_local));   % objective-space state × local action
Q_global = zeros(n_states, length(S_global)); % decision-space state × global action

ComputeFitness = fCalculation;
FEs = 0;
MaxFEs = 3e6;

%% ================ Population initialization ================
p = zeros(popsize, dimension);
v = zeros(popsize, dimension);
fitness = zeros(popsize, 1);

for i = 1:popsize
    p(i,:) = xmin + (xmax - xmin) .* rand(1, dimension);
    v(i,:) = vmin + (vmax - vmin) .* rand(1, dimension);
    fitness(i) = ComputeFitness(p(i,:)', FuncId);
end
FEs = FEs + popsize;

[bestever, id] = min(fitness);
gbestx = p(id,:);
gbesthistory = bestever * ones(MaxFEs, 1);

fprintf('AHLSO initialization complete, initial best = %e\n', bestever);

%% ================ Main loop ================
gen = 1;

while FEs < MaxFEs
    
    %% Step 1: compute the reference point (paper Eq. 8-10)
    best_val = min(fitness);
    worst_val = max(fitness);
    
    % Compute the weights (Eq. 8)
    w = zeros(popsize, 1);
    for i = 1:popsize
        w(i) = best_val / (fitness(i) + 1e-10);
    end
    w_sum = sum(w);
    
    % Compute the reference point (Eq. 9)
    p_ref = zeros(1, dimension);
    for d = 1:dimension
        p_ref(d) = sum(w .* p(:, d)) / w_sum;
    end
    
    % Compute the distance from each particle to the reference point (Eq. 10)
    dist_ref = zeros(popsize, 1);
    for i = 1:popsize
        dist_ref(i) = sqrt(sum((p(i,:) - p_ref).^2));
    end
    
    %% Step 2: sort the population (paper Algorithm 5, lines 5-6)
    % Sort by fitness (used for local search) - ascending
    [fitness_sorted, idx_fitness] = sort(fitness, 'ascend');
    P_fitness = p(idx_fitness, :);
    
    % Sort by distance to the reference point (used for global search) - ascending
    [dist_sorted, idx_dist] = sort(dist_ref, 'ascend');
    P_reference = p(idx_dist, :);
    
    %% Step 3: update the particles one by one (from worst to best)
    for i = popsize:-1:1  % paper Algorithm 5, line 7: i = NP → 1
        
        % Get the index of the current particle in the original population
        current_idx = idx_fitness(i);
        
        %% Step 3.1: compute the particle state (paper Algorithm 5, line 9; Eq. 12-13)
        
        % Decision-space state (Eq. 12) - used for global search structure selection
        min_dist = min(dist_ref);
        max_dist = max(dist_ref);
        if max_dist > min_dist
            s_D_norm = (dist_ref(current_idx) - min_dist) / (max_dist - min_dist);
        else
            s_D_norm = 0;
        end
        
        % Map to 4 states: {nearest, nearer, farther, farthest}
        if s_D_norm <= 0.25
            state_D = 1;  % nearest
        elseif s_D_norm <= 0.5
            state_D = 2;  % nearer
        elseif s_D_norm <= 0.75
            state_D = 3;  % farther
        else
            state_D = 4;  % farthest
        end
        
        % Objective-space state (Eq. 13) - used for local search structure selection
        if worst_val > best_val
            s_O_norm = (fitness(current_idx) - best_val) / (worst_val - best_val);
        else
            s_O_norm = 0;
        end
        
        % Map to 4 states: {smallest, small, larger, largest}
        if s_O_norm <= 0.25
            state_O = 1;  % smallest
        elseif s_O_norm <= 0.5
            state_O = 2;  % small
        elseif s_O_norm <= 0.75
            state_O = 3;  % larger
        else
            state_O = 4;  % largest
        end
        
        %% Step 3.2: the agent selects an action (paper Algorithm 5, line 10; Eq. 14)
        
        % Select the local search structure (based on the objective-space state)
        if rand < epsilon
            action_L = randi(length(S_local));  % exploration
        else
            [~, action_L] = max(Q_local(state_O, :));  % exploitation
        end
        SL = S_local(action_L);
        
        % Select the global search structure (based on the decision-space state)
        if rand < epsilon
            action_G = randi(length(S_global));  % exploration
        else
            [~, action_G] = max(Q_global(state_D, :));  % exploitation
        end
        SG = S_global(action_G);
        
        %% Step 3.3: compute the level of the current particle (paper Algorithm 5, lines 11-14)
        
        % Level in the local search structure (based on the fitness sort)
        level_L = ceil(i / (popsize / SL));
        level_L = min(level_L, SL);  % ensure it does not exceed the maximum level
        
        % Level in the global search structure (based on the distance sort)
        rank_in_dist = find(idx_dist == current_idx, 1);
        level_G = ceil(rank_in_dist / (popsize / SG));
        level_G = min(level_G, SG);  % ensure it does not exceed the maximum level
        
        %% Step 3.4: level-based resource allocation (LRA) (paper Algorithm 5, lines 15-19; Eq. 17)
        
        % The best level is kept directly and consumes no resources
        if level_L == 1
            continue;
        end
        
        % Compute the resource allocation probability (Eq. 17)
        prob_i = (level_L / SL) - (1 - 2*level_L/SL) * (FEs / MaxFEs);
        
        % Probabilistic resource allocation
        if rand > prob_i
            continue;  % no resource allocated; the particle goes directly to the next generation
        end
        
        %% Step 3.5: generate the local learning exemplar (paper Algorithm 2)
        
        LG_local = ceil(popsize / SL);  % number of particles per level
        
        if level_L >= 2
            % Randomly select two levels from the first (level_L - 1) levels (Algorithm 2, line 3)
            m = randi(level_L - 1);
            n = randi(level_L - 1);
            m = min(m, n);  % select the better level (Algorithm 2, line 4)
            
            % Randomly select a particle from the chosen level (Algorithm 2, line 5)
            start_idx = (m - 1) * LG_local + 1;
            end_idx = min(m * LG_local, popsize);
            k1 = randi([start_idx, end_idx]);
            p_local = P_fitness(k1, :);
        else
            % The case level_L == 1 has already continued earlier
            % This branch is never reached
            p_local = P_fitness(1, :);
        end
        
        %% Step 3.6: generate the global learning exemplar (paper Algorithm 3 + LSS strategy)
        
        LG_global = ceil(popsize / SG);  % number of particles per level
        
        % Level-based sample selection strategy (LSS) - Algorithm 5, line 21
        max_attempts = 20;
        p_global = p_local;  % default value
        
        for attempt = 1:max_attempts
            % Randomly select one level from the first level_G levels (Algorithm 3, line 4)
            m = randi(level_G);
            
            % Randomly select a particle from that level (Algorithm 3, line 5)
            start_idx = (m - 1) * LG_global + 1;
            end_idx = min(m * LG_global, popsize);
            k1 = randi([start_idx, end_idx]);
            p_global_candidate = P_reference(k1, :);
            
            % Ensure the global exemplar differs from the local exemplar and the current particle (Algorithm 3, line 2)
            if ~isequal(p_global_candidate, p_local) && ...
               ~isequal(p_global_candidate, p(current_idx, :))
                p_global = p_global_candidate;
                break;
            end
        end
        
        %% Step 3.7: update the particle (paper Eq. 6-7)
        
        r1 = rand(1, dimension);
        r2 = rand(1, dimension);
        r3 = rand(1, dimension);
        
        % Velocity update (Eq. 6)
        v(current_idx, :) = r1 .* v(current_idx, :) + ...
                            r2 .* (p_local - p(current_idx, :)) + ...
                            phi * r3 .* (p_global - p(current_idx, :));
        
        % Velocity boundary handling
        v(current_idx, :) = max(v(current_idx, :), vmin);
        v(current_idx, :) = min(v(current_idx, :), vmax);
        
        % Position update (Eq. 7)
        p(current_idx, :) = p(current_idx, :) + v(current_idx, :);
        
        % Position boundary handling
        p(current_idx, :) = max(p(current_idx, :), xmin);
        p(current_idx, :) = min(p(current_idx, :), xmax);
        
        % Compute the new fitness (Algorithm 5, line 22)
        fitness_old = fitness(current_idx);
        fitness(current_idx) = ComputeFitness(p(current_idx, :)', FuncId);
        FEs = FEs + 1;
        
        %% Step 3.8: compute the reward (paper Algorithm 5, line 24; Eq. 15-16)
        
        % Local search reward (Eq. 15) - based on the fitness improvement
        if fitness_old ~= 0
            r_local = (fitness_old - fitness(current_idx)) / abs(fitness_old);
        else
            r_local = fitness_old - fitness(current_idx);
        end
        
        % Global search reward (Eq. 16) - based on the change in distance to the reference point
        dist_new = sqrt(sum((p(current_idx, :) - p_ref).^2));
        
        if r_local > 0  % if the fitness has improved
            % Reward movement toward the reference point
            r_global = abs(dist_ref(current_idx) - dist_new) / ...
                       (max_dist - min_dist + 1e-10);
        else
            % Otherwise reward according to the distance change
            r_global = (dist_ref(current_idx) - dist_new) / ...
                       (max_dist - min_dist + 1e-10);
        end
        
        %% Step 3.9: update the Q-tables (paper Algorithm 5, line 25; Eq. 11)
        
        % Compute the new state
        if max_dist > min_dist
            s_D_norm_new = (dist_new - min_dist) / (max_dist - min_dist);
        else
            s_D_norm_new = 0;
        end
        
        if s_D_norm_new <= 0.25
            state_D_next = 1;
        elseif s_D_norm_new <= 0.5
            state_D_next = 2;
        elseif s_D_norm_new <= 0.75
            state_D_next = 3;
        else
            state_D_next = 4;
        end
        
        if worst_val > best_val
            s_O_norm_new = (fitness(current_idx) - best_val) / (worst_val - best_val);
        else
            s_O_norm_new = 0;
        end
        
        if s_O_norm_new <= 0.25
            state_O_next = 1;
        elseif s_O_norm_new <= 0.5
            state_O_next = 2;
        elseif s_O_norm_new <= 0.75
            state_O_next = 3;
        else
            state_O_next = 4;
        end
        
        % Q-learning update (Bellman equation, Eq. 11)
        % Q(s_t, a_t) = Q(s_t, a_t) + α·(r + γ·Max(Q(s_{t+1})) - Q(s_t, a_t))
        
        Q_local(state_O, action_L) = Q_local(state_O, action_L) + ...
            alpha * (r_local + gamma * max(Q_local(state_O_next, :)) - ...
            Q_local(state_O, action_L));
        
        Q_global(state_D, action_G) = Q_global(state_D, action_G) + ...
            alpha * (r_global + gamma * max(Q_global(state_D_next, :)) - ...
            Q_global(state_D, action_G));
        
        %% Step 3.10: update the global best
        if fitness(current_idx) < bestever
            bestever = fitness(current_idx);
            gbestx = p(current_idx, :);
        end
        
        gbesthistory(FEs) = bestever;
        
        % Output progress
        if mod(FEs, 10000) == 0
            fprintf('Gen %d, FEs=%d, Best=%.6e, AvgFit=%.6e\n', ...
                    gen, FEs, bestever, mean(fitness));
        end
        
        % Check the termination condition
        if FEs >= MaxFEs
            break;
        end
    end
    
    gen = gen + 1;
    
    % Check the termination condition
    if FEs >= MaxFEs
        break;
    end
end

%% ================ Fill the history record ================
if FEs < MaxFEs
    gbesthistory(FEs+1:MaxFEs) = bestever;
elseif FEs > MaxFEs
    gbesthistory = gbesthistory(1:MaxFEs);
end

fprintf('\n========================================\n');
fprintf('AHLSO finished.\n');
fprintf('Total evaluations: %d\n', FEs);
fprintf('Final best fitness: %.6e\n', bestever);
fprintf('========================================\n');

end