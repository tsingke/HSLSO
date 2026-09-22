function [gbestx,bestever,gbesthistory]=HSLSO(mainHandle,popsize,dimension,xmax,xmin,vmax,vmin,maxiter,fCalculation,FuncId,VisualSwitch)

%% ======================= Parameter settings ============================
popsize = 400;
if nargin < 2
    popsize = 400;
end

N = 10;                         % Number of fitness levels NL
m = popsize/N;                  % Number of particles in each level PopL
ComputeFitness = fCalculation;

phi = 0.3;                      % phi in Eq. (4)
FEs = 0;
MaxFEs = 3e6;

% Initial and final level-wise update probabilities in Eq. (7)
% PL_{l,init}  = l / NL
% PL_{l,final} = 1 - PL_{l,init}
PLinit = (1:N)/N;
PLfina = 1-PLinit;

% Auxiliary index used for dimension-wise exemplar construction
baseIndex = (0:dimension-1)*popsize;

%% ======================= Population initialization =====================
p = zeros(popsize,dimension);
v = zeros(popsize,dimension);
fitness = zeros(1,popsize);

for i = 1:popsize
    p(i,:) = xmin+(xmax-xmin).*rand(1,dimension);
    v(i,:) = vmin+(vmax-vmin).*rand(1,dimension);
    fitness(i) = ComputeFitness(p(i,:)',FuncId);
end

FEs = FEs+popsize;

pbest = p;
pbestfitness = fitness;

[bestever,id] = min(fitness);
gbestx = p(id,:);
gbesthistory = bestever*ones(FEs,1);

%% ======================= Main optimization loop ========================
while FEs < MaxFEs

    % Sort particles according to fitness and divide them into NL levels.
    % Level 1 contains the best particles and is preserved without update.
    [fitness,rank] = sort(fitness);
    p = p(rank,:);
    v = v(rank,:);
    pbest = pbest(rank,:);
    pbestfitness = pbestfitness(rank);

    % Update particles from level 2 to level NL
    for sub = 2:N

        %% Eq. (7): Probability Stage-Aware Update Control
        %
        % PL(l) = PL_{l,init}
        %       + (FEs / MaxFEs) * (PL_{l,final} - PL_{l,init})
        %
        PL = PLinit(sub) ...
            +(PLfina(sub)-PLinit(sub))*(FEs/MaxFEs);

        for k = 1:m

            % A particle is updated according to the level-wise
            % probability PL(l) defined in Eq. (7).
            if rand < PL

                i = (sub-1)*m+k;

                if sub == 2
                    %% Algorithm 1:
                    % Dimension-wise exemplar construction for level l = 2.
                    %
                    % Both exemplars are constructed from the personal-best
                    % solutions in level 1. For every dimension j, donor
                    % particles are independently sampled from level 1.

                    rowsA = randi(m,1,dimension);
                    rowsB = randi(m,1,dimension);

                    % exemplar_a,j = pbest^1_{rand_i(PopL),j}
                    % exemplar_b,j = pbest^1_{rand_i(PopL),j}
                    learna = pbest(rowsA+baseIndex);
                    learnb = pbest(rowsB+baseIndex);

                else
                    %% Eq. (6): Control Sampling Hierarchy
                    %
                    % k_l = max(2, ceil((l-1) *
                    %       (1-(FEs/MaxFEs)^2)))
                    %
                    % The admissible range of superior levels gradually
                    % contracts as the search progresses.

                    k_layers = ceil((sub-1) ...
                        *(1-(FEs/MaxFEs)^2));

                    k_layers = max(2,k_layers);

                    %% Algorithm 2:
                    % Randomly sample two distinct superior levels
                    % e1 and e2 from {1,...,k_l}, where e1 < e2 < l.

                    r = randperm(k_layers,2);

                    if r(1) < r(2)
                        a = r(1);       % e1
                        b = r(2);       % e2
                    else
                        a = r(2);       % e1
                        b = r(1);       % e2
                    end

                    % Algorithm 2: dimension-wise exemplar construction.
                    %
                    % For each dimension j, independently select one
                    % personal-best donor from superior level e1/e2.
                    %
                    % exemplar_a,j = pbest^{e1}_{rand_i(PopL),j}
                    % exemplar_b,j = pbest^{e2}_{rand_i(PopL),j}

                    rowsA = (a-1)*m+randi(m,1,dimension);
                    rowsB = (b-1)*m+randi(m,1,dimension);

                    learna = pbest(rowsA+baseIndex);
                    learnb = pbest(rowsB+baseIndex);
                end

                %% Eq. (4): HSLSO velocity update
                %
                % v^l_{i,j}(t+1)
                % = r1*v^l_{i,j}(t)
                % + r2*(exemplar_a^{e1}_j - x^l_{i,j}(t))
                % + phi*r3*(exemplar_b^{e2}_j - x^l_{i,j}(t))
                %
                % r1, r2 and r3 are independently sampled from U(0,1)
                % for every dimension.

                r1 = rand(1,dimension);
                r2 = rand(1,dimension);
                r3 = rand(1,dimension);

                v(i,:) = r1.*v(i,:) ...
                    + r2.*(learna-p(i,:)) ...
                    + phi.*r3.*(learnb-p(i,:));

                %% Eq. (5): HSLSO position update
                %
                % x^l_{i,j}(t+1)
                % = x^l_{i,j}(t) + v^l_{i,j}(t+1)
                %
                p(i,:) = p(i,:)+v(i,:);

                %% Boundary handling
                p(i,:) = max(p(i,:),xmin);
                p(i,:) = min(p(i,:),xmax);

                %% Fitness evaluation
                fitness(i) = ComputeFitness(p(i,:)',FuncId);
                FEs = FEs+1;

                %% Personal-best update
                if fitness(i) < pbestfitness(i)
                    pbestfitness(i) = fitness(i);
                    pbest(i,:) = p(i,:);
                end

                %% Global-best update
                if fitness(i) < bestever
                    bestever = fitness(i);
                    gbestx = p(i,:);
                end

                gbesthistory(FEs) = bestever;

                fprintf("HSLSO算法,第%d次评价，最佳适应度 = %e\n", ...
                    FEs,bestever);

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

%% ======================= History completion ============================
if FEs < MaxFEs
    gbesthistory(FEs+1:MaxFEs) = bestever;
elseif FEs > MaxFEs
    gbesthistory(MaxFEs+1:end) = [];
end

end