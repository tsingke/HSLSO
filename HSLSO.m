% =========================================================================
%  HSLSO: Heterogeneous Selection Learning Swarm Optimization (for LSGO)
%  Copyright (c) 2026, Qingke Zhang @ SDNU CILab
%  This code is released for academic and research use.
%  Please cite the related paper when using or modifying.
% =========================================================================
%  Platform version.
%  HSLSO.m at the repository root is the same algorithm in standalone form.
%  The two are identical -- same parameter list, same body -- so this file
%  can be swapped for any comparison algorithm on the platform.
% =========================================================================
function [gbestx,bestever,gbesthistory] = HSLSO(popsize,dimension,xmax,xmin,vmax,vmin,maxiter,fCalculation,FuncId)
%  Inputs
%     popsize       population size, a multiple of N (fixed to 400 below)
%     dimension     decision dimension D
%     xmax, xmin    box constraints, scalar or 1-by-D
%     vmax, vmin    velocity bounds, used to seed the initial velocity
%     maxiter       kept for interface compatibility; the budget actually
%                   consumed is MaxFEs, set below
%     fCalculation  objective handle, f = fCalculation(x,FuncId) with x a
%                   column vector
%     FuncId        problem id forwarded to fCalculation
%  Outputs
%     gbestx        best solution found, 1-by-D
%     bestever      best fitness value
%     gbesthistory  best-so-far value indexed by FEs, length MaxFEs
%

%% ===================== 1. Parameter settings ===========================
% All algorithm constants live in this block, so a run is re-configured
% from one place. The population is split into N fitness levels of m
% particles each.
popsize = 400;                      % Fixed population size, a multiple of N
if nargin < 2
    popsize = 400;
end

N = 10;                             % Number of fitness levels NL
m = popsize/N;                      % Number of particles in each level PopL
phi = 0.3;                          % phi in Eq. (4)
FEs = 0;                            % Fitness evaluations consumed so far
MaxFEs = 3e6;                       % Evaluation budget of one run

ComputeFitness = fCalculation;      % Objective handle used throughout

% Update probability of each level at the start and at the end of a run.
% The two vectors are linearly interpolated in between: Eq. (7).
PLinit = (1:N)/N;
PLfina = 1-PLinit;

% Column offsets used for vectorized dimension-wise exemplar construction.
% For a row-index vector rows, pbest(rows+baseIndex) extracts one sampled
% personal-best component from each decision dimension.
baseIndex = (0:dimension-1)*popsize;

%% ===================== 2. Population initialization ====================
% Positions are sampled uniformly in the box, velocities uniformly in the
% velocity range. Velocities are never clamped during the search.
p = zeros(popsize,dimension);       % Particle positions
v = zeros(popsize,dimension);       % Particle velocities
fitness = zeros(1,popsize);         % Fitness of the current positions

for i = 1:popsize
    p(i,:) = xmin+(xmax-xmin).*rand(1,dimension);
    v(i,:) = vmin+(vmax-vmin).*rand(1,dimension);
    fitness(i) = ComputeFitness(p(i,:)',FuncId);
end

FEs = FEs+popsize;                  % Initialization costs popsize FEs

% The personal best starts at the current state; the global best is the
% best of the personal bests.
pbest = p;
pbestfitness = fitness;

[bestever,id] = min(fitness);
gbestx = p(id,:);
gbesthistory = bestever*ones(FEs,1);    % Flat best-so-far so far

%% ===================== 3. Main optimization loop =======================
% One pass over all fitness levels constitutes one generation. Level 1 is
% retained without position updates during the current generation, whereas
% particles in levels 2,...,NL participate in updating according to SAUC,
% while their learning information is constructed through HSL.
while FEs < MaxFEs

    %% -- 3.1 Sort by fitness, then re-form the levels -------------------
    % After sorting, particles are partitioned into fitness levels from best
    % to worst. Level 1 contains the best-ranked particles and is retained
    % without position updates during the current generation.
    [fitness,rank] = sort(fitness);
    p = p(rank,:);
    v = v(rank,:);
    pbest = pbest(rank,:);
    pbestfitness = pbestfitness(rank);

    %% -- 3.2 Update every level below level 1 ---------------------------
    for sub = 2:N

        % Eq. (7): Probability Stage-Aware Update Control.
        % The level-wise update probability is linearly interpolated from PLinit
        % to PLfina according to the consumed FE ratio. Inferior levels receive
        % larger update probabilities in the early stage, whereas relatively
        % superior non-elite levels receive larger update probabilities in the
        % later stage.
        PL = PLinit(sub)+(PLfina(sub)-PLinit(sub))*(FEs/MaxFEs);

        for k = 1:m

            if rand < PL

                i = (sub-1)*m+k;    % Row index of this particle

                %% -- 3.2.1 Heterogeneous Selection Learning -----------------------------
                if sub == 2
                    % Algorithm 1 -- level 2 has only level 1 above it, so
                    % both exemplars are built dimension-wise from the
                    % personal bests of level 1.
                    rowsA = randi(m,1,dimension);
                    rowsB = randi(m,1,dimension);
                    learna = pbest(rowsA+baseIndex);
                    learnb = pbest(rowsB+baseIndex);
                else
                    % Algorithm 2 -- the admissible range of superior levels
                    % k_l contracts from (sub-1) towards 2 as FEs/MaxFEs goes
                    % from 0 to 1, so late updates learn from the top levels.
                    % Eq. (6): progress-dependent admissible superior-level range.
                    k_layers = ceil((sub-1)*(1-(FEs/MaxFEs)^2));
                    k_layers = max(2,k_layers);

                    % Draw two distinct superior levels e1 < e2 <= k_l.
                    r = randperm(k_layers,2);
                    if r(1) < r(2)
                        a = r(1);
                        b = r(2);
                    else
                        a = r(2);
                        b = r(1);
                    end

                    rowsA = (a-1)*m+randi(m,1,dimension);
                    rowsB = (b-1)*m+randi(m,1,dimension);
                    learna = pbest(rowsA+baseIndex);
                    learnb = pbest(rowsB+baseIndex);
                end

                %% -- 3.2.2 Velocity and position update ------------------
                % Eq. (4): HSLSO velocity update.
                % The two attraction terms are guided by learna and learnb,
                % respectively, with the second exemplar weighted by phi.
                r1 = rand(1,dimension);
                r2 = rand(1,dimension);
                r3 = rand(1,dimension);
                v(i,:) = r1.*v(i,:) ...
                    + r2.*(learna-p(i,:)) ...
                    + phi.*r3.*(learnb-p(i,:));

                % Eq. (5): HSLSO position update.
                p(i,:) = p(i,:)+v(i,:);

                %% -- 3.2.3 Boundary handling ----------------------------
                % Positions are clamped into the box; velocities are not.
                p(i,:) = max(p(i,:),xmin);
                p(i,:) = min(p(i,:),xmax);

                %% -- 3.2.4 Evaluation ------------------------------------
                fitness(i) = ComputeFitness(p(i,:)',FuncId);
                FEs = FEs+1;

                %% -- 3.2.5 Personal best update --------------------------
                if fitness(i) < pbestfitness(i)
                    pbestfitness(i) = fitness(i);
                    pbest(i,:) = p(i,:);
                end

                %% -- 3.2.6 Global best update ----------------------------
                if fitness(i) < bestever
                    bestever = fitness(i);
                    gbestx = p(i,:);
                end

                %% -- 3.2.7 Progress and termination ----------------------
                gbesthistory(FEs) = bestever;
                if mod(FEs, floor(MaxFEs/10)) == 0 && FEs <= MaxFEs
                    fprintf("HSLSO  FE %d  best = %e\n",FEs,bestever);
                end

                if FEs >= MaxFEs
                    break;              % Budget exhausted inside the level
                end
            end
        end

        if FEs >= MaxFEs
            break;                      % Budget exhausted between levels
        end
    end
end

%% ===================== 4. History completion ===========================
% Defensive history handling. Under the current FE-based termination
% logic, the optimization normally terminates exactly when FEs reaches
% MaxFEs.
if FEs < MaxFEs
    gbesthistory(FEs+1:MaxFEs) = bestever;
elseif FEs > MaxFEs
    gbesthistory(MaxFEs+1:end) = [];
end

end
