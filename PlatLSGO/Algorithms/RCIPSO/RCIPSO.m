function [gbestx, bestever, gbesthistory] = RCIPSO(popsize, dimension, xmax, xmin, vmax, vmin, maxiter, fCalculation, FuncId)

% --- Parameter initialization ---
popsize = 800;
% dimension = 1000;
phi = 0.3;

% --- Framework variables ---
ComputeFitness = fCalculation;
FEs = 0;
MaxFEs = 3000000;

% --- Population initialization ---
p = zeros(popsize, dimension);
v = zeros(popsize, dimension);
fitness = zeros(popsize, 1);
for i = 1:popsize
    p(i,:) = xmin + (xmax - xmin) .* rand(1, dimension);
    v(i,:) = zeros(1, dimension);
    fitness(i) = ComputeFitness(p(i,:)', FuncId);
end
FEs = FEs + popsize;
[bestever, id] = min(fitness);
gbestx = p(id,:); 
gbesthistory = zeros(MaxFEs, 1); 
gbesthistory(1:FEs) = bestever;

% --- Main loop ---
while FEs < MaxFEs
    topology_size = floor(2 + (10 - 2) * sqrt(FEs / MaxFEs));

    for i = 1:popsize
        if FEs >= MaxFEs, break; end

        population_indices = 1:popsize;
        population_indices(i) = [];
        shuffled_indices = population_indices(randperm(popsize - 1));
        topo_indices = shuffled_indices(1:topology_size);

        topo_fitness = fitness(topo_indices);
        [sorted_topo_fitness, sort_idx] = sort(topo_fitness);
        sorted_topo_indices = topo_indices(sort_idx);

        if sorted_topo_fitness(2) <= fitness(i)
            exemplar1_idx = sorted_topo_indices(1);

            % --- exemplar2 selection logic (bug fixed) ---
            exemplar2_idx = -1;
            for j = topology_size:-1:1
                if sorted_topo_fitness(j) <= fitness(i)
                    exemplar2_idx = sorted_topo_indices(j);
                    break;
                end
            end

            % [Key fix] Ensure exemplar2_idx is always a valid index
            if exemplar2_idx == -1
                exemplar2_idx = sorted_topo_indices(topology_size);
            end

            % --- Velocity and position update (keep the better-performing vectorized random numbers) ---
            r1 = rand(1, dimension);
            r2 = rand(1, dimension);
            r3 = rand(1, dimension);

            v(i,:) = r1 .* v(i,:) + ...
                     r2 .* (p(exemplar1_idx, :) - p(i, :)) + ...
                     phi .* r3 .* (p(exemplar2_idx, :) - p(i, :));

            p(i,:) = p(i,:) + v(i,:);

            p(i,:) = max(p(i,:), xmin);
            p(i,:) = min(p(i,:), xmax);

            fitness(i) = ComputeFitness(p(i,:)', FuncId);
            FEs = FEs + 1;

            if fitness(i) < bestever
                bestever = fitness(i);
                gbestx = p(i,:);
            end

            if FEs <= MaxFEs
                gbesthistory(FEs) = bestever;
                 if mod(FEs, floor(MaxFEs/10)) == 0 && FEs <= MaxFEs
                 fprintf("RCIPSO  FE %d  best = %e  topology size = %d\n", FEs, bestever, topology_size);
                 end
            end
        end
    end
end

% --- Post-processing ---
if FEs < MaxFEs
    gbesthistory(FEs+1:end) = bestever;
else
    if length(gbesthistory) > MaxFEs
        gbesthistory = gbesthistory(1:MaxFEs);
    end
end
end




% function [gbestx, bestever, gbesthistory] = rcipso_corrected(mainHandle, popsize, dimension, xmax, xmin, vmax, vmin, maxiter, fCalculation, FuncId, VisualSwitch)
% % RCIPSO - version corrected strictly following the C++ code logic
% 
% % --- Parameter initialization (fully matching C++) ---
% popsize = 800;              % const int populationsize = 800
% dimension = 1000;           % const int dim = 1000
% phi = 0.3;                  % const double phi = 0.3
% 
% % --- Framework variables ---
% ComputeFitness = fCalculation;
% FEs = 0;
% MaxFEs = 3000 * dimension;  % const int MaxFEs= 3000*dim
% 
% % --- Population initialization (corresponds to the initialization function) ---
% p = zeros(popsize, dimension);
% v = zeros(popsize, dimension);
% fitness = zeros(popsize, 1);
% 
% for i = 1:popsize
%     p(i,:) = xmin + (xmax - xmin) .* rand(1, dimension);
%     v(i,:) = zeros(1, dimension);  % corresponds to v[i][j] = 0 in C++
% end
% 
% % --- Initial fitness evaluation (corresponds to the Fitness_Computation function) ---
% fitness(1) = ComputeFitness(p(1,:)', FuncId);
% bestever = fitness(1);
% gl_best = 1;
% FEs = FEs + 1;
% 
% for i = 2:popsize
%     fitness(i) = ComputeFitness(p(i,:)', FuncId);
%     if fitness(i) < bestever
%         bestever = fitness(i);
%         gl_best = i;
%     end
%     FEs = FEs + 1;
% end
% 
% gbestx = p(gl_best,:);
% final_val = bestever;
% 
% % --- History initialization ---
% gbesthistory = zeros(MaxFEs, 1);
% gbesthistory(1:FEs) = bestever;
% 
% % --- Main loop ---
% while FEs <= MaxFEs
%     % Topology size computation (exactly matching the C++ formula)
%     topology_size = floor(2 + (10 - 2) * power((1.0*FEs)/MaxFEs, 0.5));
% 
%     for i = 1:popsize
%         if FEs > MaxFEs, break; end
% 
%         % --- Key fix: neighborhood selection logic (strictly matching C++) ---
%         % Create and randomly shuffle the population indices
%         population_index = randperm(popsize);
% 
%         % Obtain the neighborhood (corresponds to the complex logic in C++)
%         topology_ids = zeros(topology_size, 1);
%         topology_fitness = zeros(topology_size, 1);
% 
%         m = 1;
%         for j = 1:topology_size
%             % Skip the current particle (corresponds to the C++ logic)
%             if population_index(m) ~= i
%                 topology_ids(j) = population_index(m);
%                 topology_fitness(j) = fitness(population_index(m));
%                 m = m + 1;
%             else
%                 m = m + 1;
%                 if m <= popsize
%                     topology_ids(j) = population_index(m);
%                     topology_fitness(j) = fitness(population_index(m));
%                     m = m + 1;
%                 else
%                     % Boundary handling
%                     topology_ids(j) = population_index(1);
%                     topology_fitness(j) = fitness(population_index(1));
%                     m = 2;
%                 end
%             end
%         end
% 
%         % Sort the topology
%         [sorted_fitness, sort_idx] = sort(topology_fitness);
%         sorted_ids = topology_ids(sort_idx);
% 
%         % Update condition check (corresponds to if(topology[1].data <= results[i]) in C++)
%         if length(sorted_fitness) >= 2 && sorted_fitness(2) <= fitness(i)
%             % exemplar1 selection (best neighbor)
%             exemplar1_idx = sorted_ids(1);
% 
%             % --- Key fix: exemplar2 selection logic ---
%             % Corresponds to C++: search backwards for the first neighbor better than the current particle
%             j = topology_size;
%             while j >= 1 && sorted_fitness(j) > fitness(i)
%                 j = j - 1;
%             end
%             exemplar2_idx = sorted_ids(j);
% 
%             % --- Velocity and position update ---
%             % Generate random numbers dimension by dimension (matching C++ more strictly)
%             for d = 1:dimension
%                 r1 = rand();
%                 r2 = rand();
%                 r3 = rand();
% 
%                 v(i,d) = r1 * v(i,d) + ...
%                          r2 * (p(exemplar1_idx, d) - p(i, d)) + ...
%                          phi * r3 * (p(exemplar2_idx, d) - p(i, d));
% 
%                 p(i,d) = p(i,d) + v(i,d);
% 
%                 % Boundary handling (corresponds to the strict check in C++)
%                 if p(i,d) < xmin
%                     p(i,d) = xmin;
%                 end
%                 if p(i,d) > xmax
%                     p(i,d) = xmax;
%                 end
%             end
% 
%             % Function evaluation
%             FEs = FEs + 1;
%             fitness(i) = ComputeFitness(p(i,:)', FuncId);
% 
%             % Update the global best
%             if fitness(i) < final_val
%                 final_val = fitness(i);
%                 bestever = final_val;
%                 gbestx = p(i,:);
%             end
% 
%             % Record the history
%             if FEs <= MaxFEs
%                 gbesthistory(FEs) = bestever;
%                 fprintf("RCIPSO  FE %d  best = %e  topology size = %d\n", FEs, bestever, topology_size);
%             end
%         end
%     end
% end
% 
% % --- Post-processing ---
% if FEs < MaxFEs
%     gbesthistory(FEs+1:MaxFEs) = bestever;
% else
%     if length(gbesthistory) > MaxFEs
%         gbesthistory = gbesthistory(1:MaxFEs);
%     end
% end
% 
% end





% function [gbestx, bestever, gbesthistory] = RCIPSO(mainHandle, popsize, dimension, xmax, xmin, vmax, vmin, maxiter, fCalculation, FuncId, VisualSwitch)
% 
% 
% % ---------------- Parameters and basic variables ----------------
% popsize=800;
% phi     = 0.3;                 % corresponds to const double phi = 0.3
% MaxFEs  = 3000 * dimension;    % corresponds to const int MaxFEs = 3000*dim
% FEs     = 0;
% 
% % Record points (corresponds to record[i] = 5e5*(i+1) in C++)
% recordPoints = (5e5:5e5:5e5*10);
% recordCount  = numel(recordPoints);
% recordIndex  = 1;
% checkpointBest = zeros(recordCount,1);  %#ok<NASGU> % can be output externally if needed
% 
% % ---------------- Initialization (corresponds to initialization + Fitness_Computation) ----------------
% % Position: uniform random; velocity: 0
% position = xmin + (xmax - xmin) .* rand(popsize, dimension);
% velocity = zeros(popsize, dimension);
% 
% fitness  = zeros(popsize,1);
% 
% % Initial whole-population evaluation (corresponds to Fitness_Computation)
% % In C++: FEs is initially increased by populationsize (once per particle)
% for i = 1:popsize
%     fitness(i) = fCalculation(position(i,:)', FuncId);
% end
% FEs = FEs + popsize;
% 
% % Find the initial global best
% [bestever, gl_best] = min(fitness);
% gbestx = position(gl_best,:);
% 
% % gbesthistory: fixed length MaxFEs
% gbesthistory = zeros(MaxFEs,1);
% % Fill the FEs already used (here initial FEs=popsize; fill in the best of the first popsize evaluations)
% gbesthistory(1:FEs) = bestever;
% 
% % population_index array (reshuffled every round in C++; here a temporary randperm is used directly)
% % No separate persistence needed
% % The topology structure is represented in MATLAB by two arrays: topo_ids, topo_fits
% 
% if VisualSwitch
%     fprintf('RCIPSO (Func %d) init: FEs=%d best=%e\n', FuncId, FEs, bestever);
% end
% 
% % ---------------- Main loop (corresponds to while(FEs <= MaxFEs)) ----------------
% while FEs <= MaxFEs
%     rng('shuffle'); 
%     % Dynamic topology size (C++: int, truncated downward)
%     rawTopo = 2 + (10 - 2) * ((FEs / MaxFEs)^0.5);
%     topology_size = floor(rawTopo);
%     if topology_size < 2
%         topology_size = 2;
%     elseif topology_size > 10
%         topology_size = 10;
%     end
% 
%     % Loop over each particle
%     for i = 1:popsize
%         if FEs > MaxFEs
%             break;
%         end
% 
%         % ---------------- Build the topology (strictly mimicking the C++ "skip self then take the next" logic) ----------------
%         perm = randperm(popsize);   % random_shuffle
%         topo_ids  = zeros(topology_size,1);
%         topo_fits = zeros(topology_size,1);
% 
%         m = 1;
%         for j = 1:topology_size
%             if perm(m) ~= i
%                 topo_ids(j)  = perm(m);
%                 topo_fits(j) = fitness(perm(m));
%                 m = m + 1;
%             else
%                 m = m + 1;  % skip self
%                 % Potential out-of-bounds risk here in the C++ code: direct access when m exceeds popsize goes out of bounds
%                 % Safe handling: wrap around if out of bounds
%                 if m > popsize
%                     m = 1;
%                     % If wrapping around hits self again, move further on
%                     if perm(m) == i
%                         m = m + 1;
%                         if m > popsize
%                             m = 1; % handling for the extreme case
%                         end
%                     end
%                 end
%                 topo_ids(j)  = perm(m);
%                 topo_fits(j) = fitness(perm(m));
%                 m = m + 1;
%                 if m > popsize
%                     m = 1;
%                 end
%             end
%         end
% 
%         % ---------------- Sort (corresponds to sort(topology, topology+topology_size, Compare_NewType)) ----------------
%         [sorted_fits, idx_in_sorted] = sort(topo_fits);
%         sorted_ids = topo_ids(idx_in_sorted);
% 
%         % ---------------- Update condition: fitness of the second-best neighbor <= current particle ----------------
%         if sorted_fits(2) <= fitness(i)
% 
%             exemplar1 = sorted_ids(1);  % best neighbor
% 
%             % exemplar2: search backwards from the end for the first neighbor with fitness <= current particle
%             jj = topology_size;
%             while jj >= 1 && fitness(sorted_ids(jj)) > fitness(i)
%                 jj = jj - 1;
%             end
%             % Entering this if block theoretically guarantees at least two neighbors <= current, so jj >= 1 always holds
%             exemplar2 = sorted_ids(jj);
% 
%             % ---------------- Velocity & position update (per dimension, independent random numbers, matching the successive random_real_num_r() calls in C++) ----------------
%             for d = 1:dimension
%                 r1 = rand();
%                 r2 = rand();
%                 r3 = rand();
%                 velocity(i,d) = r1 * velocity(i,d) ...
%                               + r2 * (position(exemplar1,d) - position(i,d)) ...
%                               + phi * r3 * (position(exemplar2,d) - position(i,d));
%                 position(i,d) = position(i,d) + velocity(i,d);
% 
%                 % Boundary handling (corresponds to if() clamp)
%                 if position(i,d) < xmin
%                     position(i,d) = xmin;
%                 elseif position(i,d) > xmax
%                     position(i,d) = xmax;
%                 end
%             end
% 
%             % ---------------- Evaluation & FEs counting (corresponds to results[i] = fp->compute(...); FEs++) ----------------
%             FEs = FEs + 1;
%             fitness(i) = fCalculation(position(i,:)', FuncId);
% 
%             % Update the global best
%             if fitness(i) < bestever
%                 bestever = fitness(i);
%                 gbestx   = position(i,:);
%             end
% 
%             % Record the history (if capacity remains)
%             if FEs <= MaxFEs
%                 gbesthistory(FEs) = bestever;
%             end
% 
%             % Check whether a record point is crossed (corresponds to record & final_results[counter] in C++)
%             if recordIndex <= recordCount && FEs >= recordPoints(recordIndex)
%                 % checkpointBest(recordIndex) = bestever;  % uncomment if output is needed
%                 recordIndex = recordIndex + 1;
%             end
% 
%             % Visualization/logging (optional)
%             if  mod(FEs, 1000) == 0
%                 fprintf('RCIPSO Func %d | FEs=%d | best=%e | topo=%d\n', FuncId, FEs, bestever, topology_size);
%             end
%         end
%     end
% end
% 
% % ---------------- Tail filling (if FEs stops at < MaxFEs) ----------------
% if FEs < MaxFEs
%     gbesthistory(FEs+1:MaxFEs) = bestever;
% end
% end








% function [gbestx, bestever, gbesthistory] = rcipso(mainHandle, popsize, dimension, xmax, xmin, vmax, vmin, maxiter, fCalculation, FuncId, VisualSwitch)
% % RCIPSO  Random Contrastive / Dual-Exemplar Dynamic-Topology PSO (single-file version)
% % Follows the HCLPSO platform function format you provided and implements the dual-exemplar topology strategy of the C++ code.
% %
% % INPUT (all kept for platform compatibility; the usage of some parameters in this algorithm is described below):
% %   mainHandle    : optional GUI/external handle (called internally if non-empty and a callback exists)
% %   popsize       : population size (defaults to 800 if empty or < 1)
% %   dimension     : dimension
% %   xmax, xmin    : scalar or vector lower and upper bounds (length = dimension, or a scalar that will be expanded)
% %   vmax, vmin    : velocity lower and upper bounds (optional; if empty, either unbounded or set to 20% of the range)
% %   maxiter       : iteration limit (for interface compatibility; this algorithm is controlled by the number of evaluations MaxFEs, so maxiter can be ignored)
% %   fCalculation  : fitness function handle, called as fCalculation(columnVector, FuncId)
% %   FuncId        : function index passed to fCalculation
% %   VisualSwitch  : whether to print logs (true/false)
% %
% % OUTPUT:
% %   gbestx        : global best position
% %   bestever      : global best fitness
% %   gbesthistory  : history of best values with length = MaxFEs (positions without a new evaluation are filled with the most recent best)
% %
% % Key differences:
% %   - The original C++ uses MaxFEs = 3000*dim; this implementation is consistent with it.
% %   - Dynamic topology: topology_size = floor( 2 + (10-2)*sqrt(FEs/MaxFEs) ) bounded to [2,10]
% %   - A particle is updated only when the second-best neighbor's fitness <= the current particle's fitness (producing one new fitness evaluation + FEs++)
% %   - exemplar1 = best in the neighborhood; exemplar2 = the first neighbor with fitness <= the current particle, searching backwards from the worst
% %   - Random coefficients r1, r2, r3 are resampled per dimension (consistent with the calls inside the C++ for(j) loop)
% %
% % Usage example:
% %   [gbestx, bestval, curve] = RCIPSO([], 800, 1000, 100, -100, [], [], [], @myFunc, 3, true);
% %
% % -----------------------------------------------------------------------------
% 
% % ---------------------- Argument checking and defaults ----------------------
% if nargin < 3 || isempty(dimension),  error('dimension must be provided'); end
% if nargin < 4 || isempty(xmax),       xmax = 100; end
% if nargin < 5 || isempty(xmin),       xmin = -100; end
% if nargin < 6 || isempty(vmax),       vmax = []; end
% if nargin < 7 || isempty(vmin),       vmin = []; end
% if nargin < 8 || isempty(maxiter),    maxiter = inf; end %#ok<NASGU>
% if nargin < 9 || isempty(fCalculation), error('an fCalculation handle must be provided'); end
% if nargin <10 || isempty(FuncId),     FuncId = 1; end
% if nargin <11 || isempty(VisualSwitch), VisualSwitch = false; end
% 
% % Scalar bound expansion
% if isscalar(xmax), xmax = repmat(xmax,1,dimension); end
% if isscalar(xmin), xmin = repmat(xmin,1,dimension); end
% if numel(xmax) ~= dimension || numel(xmin) ~= dimension
%     error('xmax/xmin must be of size 1 or dimension');
% end
% 
% % Velocity bounds (generated as 20% of the range if not given; a scalar will be expanded)
% if isempty(vmax) || isempty(vmin)
%     span = xmax - xmin;
%     vmax = 0.2 * span;
%     vmin = -vmax;
% else
%     if isscalar(vmax), vmax = repmat(vmax,1,dimension); end
%     if isscalar(vmin), vmin = repmat(vmin,1,dimension); end
% end
% 
% % ---------------------- Global control ----------------------
% popsize = 800;
% phi    = 0.3;                 % same as C++ const double phi = 0.3
% MaxFEs = 3000 * dimension;    % same as C++ const int MaxFEs = 3000*dim
% FEs    = 0;
% 
% % Record points
% recordPoints = (5e5:5e5:5e5*10);
% recordCount  = numel(recordPoints);
% recordIndex  = 1;                 % pointer to the next record point to fill
% % checkpointBest = zeros(recordCount,1); % can be exposed for output later if needed
% 
% % ---------------------- Initialization (position + velocity + initial fitness) ----------------------
% position = xmin + rand(popsize, dimension).*(xmax - xmin);  % uniform random
% velocity = zeros(popsize, dimension);
% fitness  = zeros(popsize,1);
% 
% for i = 1:popsize
%     fitness(i) = fCalculation(position(i,:)', FuncId);
% end
% FEs = FEs + popsize;
% 
% [bestever, bestIdx] = min(fitness);
% gbestx = position(bestIdx,:);
% 
% % Preallocate history (length = MaxFEs). Positions not reached by FEs are left as 0 and filled in later.
% gbesthistory = zeros(MaxFEs,1);
% gbesthistory(1:FEs) = bestever;
% 
% 
% % ---------------------- Main loop (driven by FEs rather than iteration count) ----------------------
% while FEs <= MaxFEs
%     % Dynamic topology size
%     rawTopo = 2 + (10 - 2) * sqrt(FEs / MaxFEs);
%     topology_size = floor(rawTopo);
%     if topology_size < 2
%         topology_size = 2;
%     elseif topology_size > 10
%         topology_size = 10;
%     end
% 
%     % Loop over the population
%     for i = 1:popsize
%         if FEs > MaxFEs
%             break;
%         end
% 
%         % ---------- Build the topology (mimicking C++: random shuffle + skip self and take the next) ----------
%         perm = randperm(popsize);
%         topo_ids  = zeros(topology_size,1);
%         topo_fits = zeros(topology_size,1);
%         m = 1;
%         for j = 1:topology_size
%             if perm(m) ~= i
%                 topo_ids(j)  = perm(m);
%                 topo_fits(j) = fitness(perm(m));
%                 m = m + 1;
%             else
%                 m = m + 1;
%                 if m > popsize  % wrap-around to prevent out-of-bounds (the original C++ code is not rigorous here; protection added)
%                     m = 1;
%                     if perm(m) == i
%                         m = m + 1;
%                         if m > popsize
%                             m = 1;
%                         end
%                     end
%                 end
%                 topo_ids(j)  = perm(m);
%                 topo_fits(j) = fitness(perm(m));
%                 m = m + 1;
%                 if m > popsize
%                     m = 1;
%                 end
%             end
%         end
% 
%         % ---------- Sort ----------
%         [sorted_fits, sortIdx] = sort(topo_fits);
%         sorted_ids = topo_ids(sortIdx);
% 
%         % ---------- Trigger condition ----------
%         if sorted_fits(2) <= fitness(i)
%             exemplar1 = sorted_ids(1);
% 
%             % exemplar2: search backwards from the end for fitness <= fitness(i)
%             jj = topology_size;
%             while jj >= 1 && fitness(sorted_ids(jj)) > fitness(i)
%                 jj = jj - 1;
%             end
%             exemplar2 = sorted_ids(jj);
% 
%             % ---------- Per-dimension update (independent r1, r2, r3 for each dimension) ----------
%             for d = 1:dimension
%                 r1 = rand();
%                 r2 = rand();
%                 r3 = rand();
%                 velocity(i,d) = r1 * velocity(i,d) ...
%                               + r2 * (position(exemplar1,d) - position(i,d)) ...
%                               + phi * r3 * (position(exemplar2,d) - position(i,d));
% 
%                 % Velocity bounds
%                 if velocity(i,d) > vmax(d), velocity(i,d) = vmax(d); end
%                 if velocity(i,d) < vmin(d), velocity(i,d) = vmin(d); end
% 
%                 position(i,d) = position(i,d) + velocity(i,d);
% 
%                 % Position bounds
%                 if position(i,d) < xmin(d), position(i,d) = xmin(d); end
%                 if position(i,d) > xmax(d), position(i,d) = xmax(d); end
%             end
% 
%             % ---------- Evaluation + FEs ----------
%             FEs = FEs + 1;
%             fitness(i) = fCalculation(position(i,:)', FuncId);
% 
%             % Update the global best
%             if fitness(i) < bestever
%                 bestever = fitness(i);
%                 gbestx   = position(i,:);
%             end
% 
%             % Write the history
%             if FEs <= MaxFEs
%                 gbesthistory(FEs) = bestever;
%             end
% 
%             % Record point check
%             if recordIndex <= recordCount && FEs >= recordPoints(recordIndex)
%                 % checkpointBest(recordIndex) = bestever; % can be exposed if needed
%                 recordIndex = recordIndex + 1;
%             end
% 
%             % Visualization output
%             if  mod(FEs, 50000) == 0
%                 fprintf('[RCIPSO] Func=%d FEs=%d gbest=%e topo=%d\n', FuncId, FEs, bestever, topology_size);
%             end
% 
%         end % if condition
%     end % for i
% end % while
% 
% % Tail filling: if MaxFEs is not exhausted, fill the remaining positions with the final bestever
% if FEs < MaxFEs
%     gbesthistory(FEs+1:MaxFEs) = bestever;
% end
% 
% end % function RCIPSO