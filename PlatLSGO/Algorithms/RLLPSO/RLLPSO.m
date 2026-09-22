function [gbestX, gbestfitness, gbesthistory] = RLLPSO(PopSize, D, xmax, xmin, vmax, vmin, MaxIter, fCal, FuncId)
% RLLPSO: Reinforcement Learning Level-based PSO for LSOPs
% Platform interface: [gbestX, gbestfitness, gbesthistory] = RLLPSO(PopSize, D, xmax, xmin, vmax, vmin, MaxIter, fCal, FuncId)
% Settings used in the paper:  φ=0.4, α=0.4, γ=0.8, ε=0.9, S={4,6,8,10,20,50}; p_lcomp = (FEcount/MaxFEs)^2
% FE counting strictly follows the platform: MaxFEs = MaxIter * PopSize; +1 per evaluated individual, written into gbesthistory(FEs)
PopSize=500;
% -------------------- Parameters / constants --------------------
phi = 0.4;           % φ in Eq.(4)
alpha = 0.4;         % Q-learning α
gamma = 0.8;         % Q-learning γ
epsilon = 0.9;       % ε-greedy
S_cand = [4 6 8 10 20 50];              % candidate set of level numbers
S_valid = S_cand(S_cand <= PopSize);    % must not exceed the population size
if isempty(S_valid), S_valid = min(4, PopSize); end
if numel(S_valid)==1, epsilon = 1; end  % with a single action, always "exploit"

MaxFEs = 3*10^6;
FEs = 0;
tiny = 1e-12;        % guard against division by zero

% -------------------- Initialize swarm --------------------
% position / velocity
X = xmin + (xmax - xmin) .* rand(PopSize, D);
V = zeros(PopSize, D);

% initial evaluation
fitness = inf(PopSize,1);
gbestfitness = inf; 
gbestX = zeros(1, D);
for i = 1:PopSize
    fi = fCal(X(i,:)', FuncId);
    fitness(i) = fi;
    if fi < gbestfitness
        gbestfitness = fi;
        gbestX = X(i,:);
    end
end

% history record
gbesthistory = inf(1, MaxFEs);

% Q-table (both states and actions are "level indices")
ns = numel(S_valid);
Q = zeros(ns, ns);
state_idx = 1;                    % s0 = l1
prev_gbest = gbestfitness;        % used when computing the reward

% -------------------- Main loop (driven by FEs) --------------------
while FEs <= MaxFEs
    % 1) ε-greedy selection of the level-number action (the number of levels for the next round)
    if rand < epsilon
        [~, action_idx] = max(Q(state_idx, :));   % exploitation
    else
        action_idx = randi(ns);                   % exploration
    end
    L = S_valid(action_idx);       % number of levels for this round
    
    % 2) Sort by fitness and divide equally into L levels (the last level takes the remainder and is the worst level)
    [~, ord] = sort(fitness, 'ascend');           % smaller is better
    levelIdx = partition_levels(ord, PopSize, L); % cell{1..L}, each holding an index array
    
    % 3) Generate the new generation (based on inter-level learning and inter-level competition)
    Xnew = X;
    Vnew = V;
    % 3.1 Level 2: learns only from level 1; level 1: frozen
    if L >= 2 && ~isempty(levelIdx{2})
        ex1_pool = levelIdx{1};
        for k = levelIdx{2}(:)'   % particle by particle
            [e1_idx, e2_idx] = pick_two_from_level(ex1_pool);
            r1 = rand(1, D); r2 = rand(1, D); r3 = rand(1, D);
            Vnew(k,:) = r1 .* V(k,:) + r2 .* (X(e1_idx,:) - X(k,:)) + phi .* r3 .* (X(e2_idx,:) - X(k,:));
            Vnew(k,:) = clip_by(Vnew(k,:), vmin, vmax);
            Xnew(k,:) = X(k,:) + Vnew(k,:);
            Xnew(k,:) = min(max(Xnew(k,:), xmin), xmax);
        end
    end
    % 3.2 Levels 3..L: learn from two "higher" levels, with inter-level competition
    for li = 3:L
        cur_pool = levelIdx{li};
        if isempty(cur_pool), continue; end
        for k = cur_pool(:)'   % particle by particle
            % Select two exemplar levels (Algorithm 2: Level competition mechanism)
            probl = (FEs / MaxFEs)^2; % estimate the trigger probability from the FEs used so far
            [le1, le2] = select_two_exemplar_levels(li, L, probl);
            % Randomly take 1 exemplar particle from each of levels le1 and le2
            e1 = levelIdx{le1}( randi(numel(levelIdx{le1})) );
            e2 = levelIdx{le2}( randi(numel(levelIdx{le2})) );
            % Update velocity / position (Eq.4 & Eq.5)
            r1 = rand(1, D); r2 = rand(1, D); r3 = rand(1, D);
            Vnew(k,:) = r1 .* V(k,:) + r2 .* (X(e1,:) - X(k,:)) + phi .* r3 .* (X(e2,:) - X(k,:));
            Vnew(k,:) = clip_by(Vnew(k,:), vmin, vmax);
            Xnew(k,:) = X(k,:) + Vnew(k,:);
            Xnew(k,:) = min(max(Xnew(k,:), xmin), xmax);
        end
    end
    
    % 4) Evaluate the new population and record (consistent with the platform convention: +1 per evaluation, written to gbesthistory)
    newfitness = zeros(PopSize,1);
    for i = 1:PopSize
        if FEs >= MaxFEs, break; end
        FEs = FEs + 1;
        fi = fCal(Xnew(i,:)', FuncId);
        newfitness(i) = fi;
        if fi < gbestfitness
            gbestfitness = fi;
            gbestX = Xnew(i,:);
        end
        gbesthistory(FEs) = gbestfitness;
        if mod(FEs, floor(MaxFEs/10)) == 0 && FEs <= MaxFEs
            fprintf('RLLPSO  FE %d  best = %e\n', FEs, gbestfitness);
        end
    end
    
    % 5) Environmental feedback (reward) and Q update (Eq.7 & Eq.8)
    reward = abs(prev_gbest - gbestfitness) / max(abs(prev_gbest), tiny);
    prev_gbest = gbestfitness;
    if ns > 1
        Q(state_idx, action_idx) = Q(state_idx, action_idx) + ...
            alpha * (reward + gamma * max(Q(action_idx, :)) - Q(state_idx, action_idx));
        state_idx = action_idx; % s_{t+1} = a_{t+1}
    end
    
    % 6) Prepare the next round
    X = Xnew; V = Vnew; fitness = newfitness;
    
    % 7) Loop termination condition (by FEs)
    if FEs >= MaxFEs, break; end
end

% Pad at the end (the platform requires gbesthistory to have length exactly MaxFEs)
if FEs < MaxFEs
    gbesthistory(FEs+1:MaxFEs) = gbestfitness;
elseif FEs > MaxFEs
    gbesthistory(MaxFEs+1:end) = [];
end

end % ===== end of main function =====

% -------------------- Helper functions --------------------
function cells = partition_levels(order_idx, N, L)
% Divide the sorted indices equally into L levels; the last level takes the remainder (and is the "worst level")
LS = floor(N / L);
remN = mod(N, L);
cells = cell(L,1);
startPos = 1;
for li = 1:L
    take = LS + (li==L) * remN;
    if take == 0
        cells{li} = [];
    else
        cells{li} = order_idx(startPos : startPos + take - 1);
        startPos = startPos + take;
    end
end
end

function [le1, le2] = select_two_exemplar_levels(curL, L, probl)
% Algorithm 2: level competition. Returns two "higher" levels (more elite: smaller index)
le1 = choose_one_level(curL, L, probl);
le2 = choose_one_level(curL, L, probl);
% If le2 is higher (more elite) than le1, swap them so that le1 is not more elite than le2
if le2 < le1
    tmp = le1; le1 = le2; le2 = tmp;
end
end

function le = choose_one_level(curL, L, probl)
% Select one level from {1,2,...,curL-1}; with probability probl trigger a "two-level competition" and take the higher level
if curL <= 2
    le = 1; return;
end
if rand < probl && (curL - 1) >= 2
    c1 = randi(curL-1); c2 = randi(curL-1);
    while c2 == c1, c2 = randi(curL-1); end
    le = min(c1, c2); % the "higher" level (smaller index) wins
else
    le = randi(curL-1);
end
end

function [i1, i2] = pick_two_from_level(pool)
% Randomly draw two exemplar particles from the same level (they may be identical)
n = numel(pool);
if n == 1
    i1 = pool(1); i2 = pool(1);
else
    i1 = pool(randi(n));
    i2 = pool(randi(n));
end
end

function v = clip_by(v, vmin, vmax)
% Velocity clipping (if the platform does not actually use vmax/vmin, it may remain zero or infinite)
if ~isempty(vmax) && ~isempty(vmin)
    v = min(max(v, vmin), vmax);
end
end
