%% ======================= Parameter settings ============================

popsize = 50;
N = 10;
m = popsize/N;

phi = 0.3;

xmax = 1;
xmin = 0;
vmax = 1;
vmin = 0;

historyPoints = max(1,round(historyPoints));

% Default maximum FE budget.
% This corresponds to the previous maximum possible FE count
% under 200 generations, including initial population evaluations.
if nargin < 7 || isempty(MaxFEs)
    MaxFEs = popsize + historyPoints*(popsize-m);
end

if ~isnumeric(MaxFEs) || ~isscalar(MaxFEs) || ...
        ~isreal(MaxFEs) || isnan(MaxFEs) || isinf(MaxFEs) || ...
        MaxFEs < popsize || MaxFEs ~= floor(MaxFEs)

    error('HSLSO:InvalidMaxFEs', ...
        'MaxFEs must be an integer not smaller than popsize.');
end

if dimension < 1 || dimension ~= floor(dimension)
    error('HSLSO:InvalidDimension', ...
        'Dimension must be a positive integer.');
end

if mod(popsize,N) ~= 0
    error('HSLSO:InvalidPopulation', ...
        'Population size must be divisible by the number of levels.');
end

% Initial and final level-wise update probabilities.
% PL_{l,init}  = l/N
% PL_{l,final} = 1-l/N
PLinit = (1:N)/N;
PLfina = 1-PLinit;

% Auxiliary index for dimension-wise exemplar construction.
baseIndex = (0:dimension-1)*popsize;

%% ======================= Population initialization =====================

p = zeros(popsize,dimension);
v = zeros(popsize,dimension);
fitnessx = zeros(1,popsize);

for i = 1:popsize

    p(i,:) = xmin+(xmax-xmin).*rand(1,dimension);

    v(i,:) = vmin+(vmax-vmin).*rand(1,dimension);

    fitnessx(i) = fitness( ...
        sequence,aa,lengthdata,L,p(i,:));

end

% Count all initial population evaluations.
FEs = popsize;

pbest = p;
pbestfitness = fitnessx;

% MSA uses SPS maximization.
[bestever,id] = max(fitnessx);
gbestx = p(id,:);

%% ======================= FE history initialization =====================

% Record the best-so-far SPS against the actual FE count.
feTrace = zeros(MaxFEs-popsize+1,1);
bestTrace = zeros(MaxFEs-popsize+1,1);

traceCount = 1;

feTrace(traceCount) = FEs;
bestTrace(traceCount) = bestever;

%% ======================= Main optimization loop ========================

generation = 0;

while FEs < MaxFEs

    generation = generation+1;

    % Rank particles from best to worst (SPS maximization).
    [fitnessx,rank] = sort(fitnessx,'descend');

    p = p(rank,:);
    v = v(rank,:);
    pbest = pbest(rank,:);
    pbestfitness = pbestfitness(rank);

    % Update particles from level 2 to level N.
    for sub = 2:N

        if FEs >= MaxFEs
            break;
        end

        %% Eq. (7): Probability Stage-Aware Update Control

        PL = PLinit(sub) ...
            +(PLfina(sub)-PLinit(sub))*(FEs/MaxFEs);

        for k = 1:m

            if FEs >= MaxFEs
                break;
            end

            % Probabilistic particle activation.
            if rand < PL

                i = (sub-1)*m+k;

                %% ================= Level 2 =========================

                if sub == 2

                    % Both exemplars are constructed from level 1.
                    % Donors are independently sampled for each dimension.

                    rowsA = randi(m,1,dimension);
                    rowsB = randi(m,1,dimension);

                    learna = pbest(rowsA+baseIndex);
                    learnb = pbest(rowsB+baseIndex);

                %% =============== Levels 3 to N =====================

                else

                    %% Eq. (6): Progressive Superior-Level Sampling

                    k_layers = ceil((sub-1) ...
                        *(1-(FEs/MaxFEs)^2));

                    k_layers = max(2,k_layers);

                    % Numerical and indexing safety.
                    k_layers = min(sub-1,k_layers);

                    % Sample two distinct superior levels.
                    r = randperm(k_layers,2);

                    a = min(r);
                    b = max(r);

                    % Dimension-wise personal-best donor sampling.
                    rowsA = (a-1)*m+randi(m,1,dimension);
                    rowsB = (b-1)*m+randi(m,1,dimension);

                    learna = pbest(rowsA+baseIndex);
                    learnb = pbest(rowsB+baseIndex);

                end

                %% ================= Velocity update =================

                r1 = rand(1,dimension);
                r2 = rand(1,dimension);
                r3 = rand(1,dimension);

                v(i,:) = r1.*v(i,:) ...
                    +r2.*(learna-p(i,:)) ...
                    +phi.*r3.*(learnb-p(i,:));

                %% ================= Position update =================

                p(i,:) = p(i,:)+v(i,:);

                % Boundary handling.
                p(i,:) = max(p(i,:),xmin);
                p(i,:) = min(p(i,:),xmax);

                %% ================= Fitness evaluation ===============

                fitnessx(i) = fitness( ...
                    sequence,aa,lengthdata,L,p(i,:));

                FEs = FEs+1;

                %% ================= Personal-best update ==============

                if fitnessx(i) > pbestfitness(i)

                    pbestfitness(i) = fitnessx(i);
                    pbest(i,:) = p(i,:);

                end

                %% ================= Global-best update ================

                if fitnessx(i) > bestever

                    bestever = fitnessx(i);
                    gbestx = p(i,:);

                end

                %% ================= FE history ========================

                traceCount = traceCount+1;

                feTrace(traceCount) = FEs;
                bestTrace(traceCount) = bestever;

                if mod(FEs,1000) == 0 || FEs == MaxFEs

                    fprintf(['HSLSO-MSA: FE %d/%d, ' ...
                        'Best SPS = %.8f\n'], ...
                        FEs,MaxFEs,bestever);

                end

            end
        end
    end
end

%% ======================= Final evaluation record =======================

actualFEs = FEs;

% Return a fixed-length convergence history sampled uniformly
% over the consumed FE budget.
if MaxFEs == popsize

    gbesthistory = bestever*ones(historyPoints,1);

else

    if historyPoints == 1
        sampleFEs = MaxFEs;
    else
        sampleFEs = linspace( ...
            popsize,MaxFEs,historyPoints);
    end

    gbesthistory = interp1( ...
        feTrace(1:traceCount), ...
        bestTrace(1:traceCount), ...
        sampleFEs,'previous')';

end

gbesthistory = gbesthistory(:);

fprintf(['HSLSO-MSA completed: FEs = %d/%d, ' ...
    'Generations = %d, Best SPS = %.8f\n'], ...
    actualFEs,MaxFEs,generation,bestever);

end
