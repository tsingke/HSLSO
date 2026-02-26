% =========================================================================
%  HSLSO: Heterogeneous Selection Learning Swarm Optimization (for LSGO)
%  Copyright (c) 2026, Qingke Zhang @ SDNU CILab
%  This code is released for academic and research use.
%  Please cite the related paper when using or modifying.
% =========================================================================
%  Function
%     [gbestX, gbestFitness, gbestHistory] = HSLSO(popsize, dimension, xmax, xmin, maxiter, Func, FuncId, opts)
%
%  Inputs
%     popsize    : population size (e.g., 400)
%     dimension  : decision dimension
%     xmax, xmin : box constraints (scalar or 1-by-D vectors)
%     maxiter    : maximum iterations (MaxFEs = popsize * maxiter)
%     Func       : function handle, f = Func(x, FuncId); x is column vector
%     FuncId     : benchmark/problem id passed to Func
%     opts       : (optional) struct with fields:
%                  .NLayers  (default: 10)         number of hierarchy layers
%                  .phi      (default: 0.3)        second-exemplar weight
%                  .verbose  (default: true)       print progress
%                  .seed     (default: [])         rng seed (e.g., 42)
%                  .vmaxRate (default: 0.2)        vmax = vmaxRate*(xmax-xmin)
%
%  Outputs
%     gbestX        : best solution found (1-by-D)
%     gbestFitness  : best fitness value
%     gbestHistory  : best-so-far curve over FEs (length = MaxFEs)
% =========================================================================
function [gbestX, gbestFitness, gbestHistory] = HSLSO(popsize, dimension, xmax, xmin, maxiter, Func, FuncId, opts)

    % -------------------------
    % Parameters & preparation
    % -------------------------
    if nargin < 8 || isempty(opts), opts = struct(); end
    NLayers  = getOpt(opts, 'NLayers', 10);
    phi      = getOpt(opts, 'phi', 0.3);
    verbose  = getOpt(opts, 'verbose', true);
    seed     = getOpt(opts, 'seed', []);
    vmaxRate = getOpt(opts, 'vmaxRate', 0.2);

    if ~isempty(seed), rng(seed); end

    MaxFEs = popsize * maxiter;
    ComputeFitness = Func;

    % Broadcast bounds if scalar
    if isscalar(xmax), xmax = repmat(xmax, 1, dimension); end
    if isscalar(xmin), xmin = repmat(xmin, 1, dimension); end

    % Velocity bounds (fraction of search range)
    vmag  = vmaxRate * (xmax - xmin);
    vmax  =  vmag;
    vmin  = -vmag;

    % Layering
    N = NLayers;
    m = floor(popsize / N);                  % individuals per layer
    if m * N ~= popsize
        % enforce exact layering by truncating tail; or pad if desired
        popsize = m * N;
        if verbose
            fprintf('[HSLSO] popsize adjusted to %d for %d uniform layers.\n', popsize, N);
        end
    end

    % Probability schedule for heterogeneous learning (layer-wise)
    PLinit = (1:N) / N;
    PLfina = 1 - PLinit;

    % -------------------------
    % Initialization
    % -------------------------
    P  = xmin + (xmax - xmin) .* rand(popsize, dimension);   % positions
    V  = vmin + (vmax - vmin) .* rand(popsize, dimension);   % velocities
    F  = zeros(popsize, 1);

    FEs = 0;
    for i = 1:popsize
        F(i) = ComputeFitness(P(i, :)', FuncId);
    end
    FEs = FEs + popsize;

    Pbest        = P;
    PbestFitness = F;

    [gbestFitness, id] = min(F);
    gbestX = P(id, :);
    gbestHistory = gbestFitness * ones(MaxFEs, 1);
    gbestHistory(1:FEs) = gbestFitness;

    % -------------------------
    % Main loop
    % -------------------------
    while FEs < MaxFEs

        % ---- Rank & layer assignment (ascending fitness) ----
        [F, rank]       = sort(F);
        P               = P(rank, :);
        V               = V(rank, :);
        Pbest           = Pbest(rank, :);
        PbestFitness    = PbestFitness(rank);

        % ---- Heterogeneous selection-learning by layers ----
        for sub = 2:N        % the best layer (sub==1) remains as elite buffer
            PL = PLinit(sub) + (PLfina(sub) - PLinit(sub)) * (FEs / MaxFEs);

            for k = 1:m
                if rand < PL
                    idx = (sub - 1) * m + k;

                    % ----- Choose learning exemplars (hierarchical selection) -----
                    if sub == 2
                        % learn from the top layer only
                        aLayer = 1; bLayer = 1;
                    else
                        % learn from two layers within the upper hierarchy
                        k_layers = ceil((sub - 1) * (1 - (FEs / MaxFEs)^2));
                        k_layers = max(k_layers, 2);
                        rr = randperm(k_layers, 2);
                        aLayer = min(rr); bLayer = max(rr);
                    end

                    % random exemplars within chosen layers (use pbest)
                    aIdx = (aLayer - 1) * m + randi(m);
                    bIdx = (bLayer - 1) * m + randi(m);
                    learnA = Pbest(aIdx, :);
                    learnB = Pbest(bIdx, :);

                    % ----- Velocity & position update (level-aware heterogeneity) -----
                    r1 = rand(1, dimension);
                    r2 = rand(1, dimension);
                    r3 = rand(1, dimension);

                    V(idx, :) = r1 .* V(idx, :) ...
                              + r2 .* (learnA - P(idx, :)) ...
                              + phi * r3 .* (learnB - P(idx, :));

                    % clamp velocity
                    V(idx, :) = max(min(V(idx, :), vmax), vmin);

                    % position update + box constraints
                    P(idx, :) = P(idx, :) + V(idx, :);
                    P(idx, :) = max(min(P(idx, :), xmax), xmin);

                    % evaluate
                    F(idx) = ComputeFitness(P(idx, :)', FuncId);
                    FEs = FEs + 1;

                    % personal best
                    if F(idx) < PbestFitness(idx)
                        PbestFitness(idx) = F(idx);
                        Pbest(idx, :)     = P(idx, :);
                    end

                    % global best
                    if F(idx) < gbestFitness
                        gbestFitness = F(idx);
                        gbestX       = P(idx, :);
                    end

                    % history
                    if FEs <= MaxFEs
                        gbestHistory(FEs) = gbestFitness;
                    end

                    if verbose && mod(FEs, 1000) == 0
                        fprintf('[HSLSO] FEs=%8d | gbest=%.8e\n', FEs, gbestFitness);
                    end

                    if FEs >= MaxFEs
                        break;
                    end
                end
            end
            if FEs >= MaxFEs
                break;
            end
        end
    end

    % fill tail (if early exits happen exactly at MaxFEs, this is a no-op)
    if FEs < MaxFEs
        gbestHistory(FEs+1:MaxFEs) = gbestFitness;
    end
end

% =========================
% Helpers (local functions)
% =========================
function val = getOpt(s, name, defaultVal)
    if isfield(s, name) && ~isempty(s.(name))
        val = s.(name);
    else
        val = defaultVal;
    end
end

