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
% particles each, re-formed by the sort at the top of every iteration (3.1).
popsize = 400;                      % Fixed population size, a multiple of N
if nargin < 2
    popsize = 400;                  % No-op: popsize is already fixed above
end

N = 10;                             % Number of fitness levels NL
m = popsize/N;                      % Number of particles in each level PopL
phi = 0.3;                          % phi in Eq. (4)
FEs = 0;                            % Fitness evaluations consumed so far
MaxFEs = 3e6;                       % Evaluation budget of one run

ComputeFitness = fCalculation;      % Objective handle used throughout

% Update probability of each level at the start and at the end of a run:
% PLinit runs 0.1..1.0 and PLfina 0.9..0.0 from level 1 to level N. The two
% are linearly interpolated in between: Eq. (7). Entry 1 of both is never
% read, because the update loop below starts at level 2.
PLinit = (1:N)/N;
PLfina = 1-PLinit;

% Linear-index offsets into the popsize-by-dimension pbest matrix, one per
% dimension: added to a row index they pick out that row in every column,
% which is how 3.2.1 builds an exemplar one dimension at a time.
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
gbesthistory = bestever*ones(FEs,1);    % Constant over initialization

%% ===================== 3. Main optimization loop =======================
% One pass over the levels is one iteration. Every pass re-sorts first, so
% level 1 holds the best m particles and level N the worst. Level 1 is never
% updated; each particle below it either follows the hierarchical selection
% rule or is left untouched, according to its level's probability PL.
while FEs < MaxFEs

    %% -- 3.1 Sort by fitness, then re-form the levels -------------------
    % Ascending sort -- minimization -- applied to positions, velocities and
    % personal bests alike, so the rows stay aligned. Level 1 is the best m
    % rows and is skipped by the update loop below. Membership follows the
    % ranking rather than a fixed index, which is what makes these "fitness
    % levels".
    [fitness,rank] = sort(fitness);
    p = p(rank,:);
    v = v(rank,:);
    pbest = pbest(rank,:);
    pbestfitness = pbestfitness(rank);

    %% -- 3.2 Update every level below level 1 ---------------------------
    for sub = 2:N

        % Level-wise update probability: Eq. (7). Each level interpolates
        % linearly from PLinit(sub) to PLfina(sub) over the budget. PLinit
        % rises with the level index while PLfina falls, so the update effort
        % migrates from the worst levels towards the better ones: levels 2-4
        % end up updated more often than they start, level 5 stays at 1/2
        % throughout, and levels 6-N progressively freeze.
        PL = PLinit(sub)+(PLfina(sub)-PLinit(sub))*(FEs/MaxFEs);

        for k = 1:m

            if rand < PL

                i = (sub-1)*m+k;    % Row index of this particle

                %% -- 3.2.1 Hierarchical selection of two exemplars ------
                if sub == 2
                    % Algorithm 1 -- level 2 has only one superior level
                    % (level 1), while the generic branch below requires two
                    % distinct superior levels. Both exemplars are therefore
                    % drawn from level 1, independently per dimension, so the
                    % two differ from one dimension to the next. That branch's
                    % max(2,...) pin does not cover this case: for sub == 2 it
                    % would yield levels 1 and 2, and level 2 is the particle's
                    % own level rather than a superior one.
                    rowsA = randi(m,1,dimension);
                    rowsB = randi(m,1,dimension);
                    learna = pbest(rowsA+baseIndex);
                    learnb = pbest(rowsB+baseIndex);
                else
                    % Algorithm 2 -- the admissible range of superior levels
                    % k_l contracts from (sub-1) towards 2 as FEs/MaxFEs goes
                    % from 0 to 1, so late updates learn from the top levels.
                    % The square holds k_l wide early and closes it sharply at
                    % the end; max() pins it at 2 so that two distinct
                    % superior levels always remain available.
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
                % r1 keeps the previous velocity, r2 weights learna and
                % phi*r3 weights learnb: Eq. (4). For sub > 2, learna comes
                % from the better of the two levels drawn above.
                r1 = rand(1,dimension);
                r2 = rand(1,dimension);
                r3 = rand(1,dimension);
                v(i,:) = r1.*v(i,:) ...
                    + r2.*(learna-p(i,:)) ...
                    + phi.*r3.*(learnb-p(i,:));

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
                % Indexed by FEs, so entry k is the best value seen after k
                % evaluations. The print fires at each 10% of the budget,
                % which is what produces the 10 log lines per run.
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
% The returned history must be exactly MaxFEs long. As written, 3.2.7 breaks
% out the moment FEs reaches MaxFEs and every increment writes its own entry,
% so FEs always ends exactly at MaxFEs and neither branch below ever fires.
% They are kept as a guard: pad the tail if a run ever stopped short, trim it
% if one ever ran long.
if FEs < MaxFEs
    gbesthistory(FEs+1:MaxFEs) = bestever;
elseif FEs > MaxFEs
    gbesthistory(MaxFEs+1:end) = [];
end

end
