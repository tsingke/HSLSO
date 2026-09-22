function [gbestx,bestever,gbesthistory] = DLLSO(popsize,dimension,xmax,xmin,vmax,vmin,maxiter,fCalculation,FuncId)

% ==============================================================
% Dynamic Level-Based Learning Swarm Optimizer (DLLSO)
%
% Q. Yang et al.,
% "A Level-Based Learning Swarm Optimizer for Large-Scale Optimization"
% IEEE Transactions on Evolutionary Computation, 2018
%
% This version is adapted to the current platform interface:
%
% function [gbestx,bestever,gbesthistory] = DLLSO(...
%     popsize,dimension,xmax,xmin,vmax,vmin,...
%     maxiter,fCalculation,FuncId)
%
% Settings corresponding to the paper's 1000-D experiments:
% NP  = 500
% phi = 0.4
% S   = {4,6,8,10,20,50}
% MaxFEs = 3e6 = 3000 * 1000
%
% ==============================================================


%% ===================== Parameter settings ===============================

% Paper's 1000-D settings
m = 500;                       % NP
phi = 0.4;                     % Controls the influence of the second exemplar

% Candidate set of dynamic level numbers
S = [4, 6, 8, 10, 20, 50];
numS = length(S);

% Performance record for each candidate level number in Eq. (8)
% The paper specifies that all entries are initialized to 1
R = ones(1,numS);

ComputeFitness = fCalculation;

FEs = 0;

% Number of FEs for the main 1000-D CEC2010 / CEC2013 experiments
MaxFEs = 3e6;

% If you later want to strictly follow 3000*D:
% MaxFEs = 3000 * dimension;


%% ===================== Initialize population =============================

p = zeros(m,dimension);
v = zeros(m,dimension);
fitness = zeros(1,m);

for i = 1:m

    p(i,:) = xmin + (xmax-xmin).*rand(1,dimension);

    v(i,:) = vmin + (vmax-vmin).*rand(1,dimension);

    fitness(i) = ComputeFitness(p(i,:)',FuncId);

end

FEs = FEs + m;


%% ===================== Initialize global best ==========================

[bestever,id] = min(fitness);

gbestx = p(id,:);


%% ===================== Convergence history ===============================

gbesthistory = zeros(MaxFEs,1);

% Kept consistent with your current CSO platform:
% The m evaluations of the initial population all record the best value after initialization
gbesthistory(1:FEs) = bestever;


gen = 1;


%% ==============================================================
%                      DLLSO main loop
% ==============================================================

while FEs < MaxFEs


    %% ----------------------------------------------------------
    % 1. Compute the selection probability of each NL according to Eq. (9)
    %
    %             exp(7*r_i)
    % p_i = -----------------------
    %        sum_j exp(7*r_j)
    %
    % -----------------------------------------------------------

    prob = exp(7 .* R);

    prob = prob ./ sum(prob);


    %% ----------------------------------------------------------
    % 2. Roulette Wheel Selection
    %    Dynamically select the level number NL for the current generation
    % -----------------------------------------------------------

    cumulativeProb = cumsum(prob);

    randValue = rand;

    selectedIndex = find(randValue <= cumulativeProb,1,'first');

    NL = S(selectedIndex);


    %% ----------------------------------------------------------
    % Record the global best before this generation's update
    % Used for Eq. (8)
    % -----------------------------------------------------------

    oldBest = bestever;


    %% ----------------------------------------------------------
    % 3. Sort particles in ascending order of fitness
    %
    % Smaller fitness is better
    %
    % L1: best
    % L2: second best
    % ...
    % LNL: worst
    %
    % -----------------------------------------------------------

    [~,rankIndex] = sort(fitness,'ascend');


    %% ----------------------------------------------------------
    % 4. Divide the population into NL levels
    %
    % The paper:
    % LS = floor(NP/NL)
    %
    % If the division is not exact, all remaining particles join the lowest level LNL
    %
    % -----------------------------------------------------------

    LS = floor(m/NL);

    levels = cell(1,NL);

    for level = 1:NL-1

        startIndex = (level-1)*LS + 1;

        endIndex = level*LS;

        levels{level} = rankIndex(startIndex:endIndex);

    end

    % The last level contains all remaining particles
    startIndex = (NL-1)*LS + 1;

    levels{NL} = rankIndex(startIndex:m);


    %% ==========================================================
    % 5. Update LNL, LNL-1, ..., L3
    %
    % Algorithm 1:
    %
    % for i = NL,...,3
    %
    % Randomly select two distinct levels from the first i-1 higher levels:
    %
    % L_rl1 and L_rl2
    %
    % with:
    %
    % rl1 < rl2 < i
    %
    % Randomly select one exemplar from each of the two levels.
    %
    % ==========================================================

    stopFlag = false;

    for level = NL:-1:3

        currentParticles = levels{level};

        numCurrent = length(currentParticles);


        for jj = 1:numCurrent

            currentID = currentParticles(jj);


            %% --------------------------------------------------
            % Randomly select two distinct levels from the higher levels
            % ---------------------------------------------------

            selectedLevels = randperm(level-1,2);

            rl1 = selectedLevels(1);
            rl2 = selectedLevels(2);


            % Ensure rl1 < rl2
            %
            % The smaller the level index, the higher the level
            %
            % Therefore:
            %
            % exemplar1 comes from a better level
            % exemplar2 comes from a relatively worse higher level
            %

            if rl2 < rl1

                temp = rl1;
                rl1 = rl2;
                rl2 = temp;

            end


            %% --------------------------------------------------
            % Randomly select one particle from L_rl1 and one from L_rl2
            % ---------------------------------------------------

            group1 = levels{rl1};
            group2 = levels{rl2};

            k1 = randi(length(group1));
            k2 = randi(length(group2));

            exemplar1 = group1(k1);
            exemplar2 = group2(k2);


            %% --------------------------------------------------
            % Eq. (6)
            %
            % v_i =
            % r1*v_i
            % + r2*(X_rl1,k1 - X_i)
            % + phi*r3*(X_rl2,k2 - X_i)
            %
            % Note:
            % r1, r2, r3 are all dimension-dimensional random vectors
            %
            % ---------------------------------------------------

            r1 = rand(1,dimension);
            r2 = rand(1,dimension);
            r3 = rand(1,dimension);

            v(currentID,:) = ...
                r1 .* v(currentID,:) ...
                + r2 .* (p(exemplar1,:) - p(currentID,:)) ...
                + phi .* r3 .* (p(exemplar2,:) - p(currentID,:));


            %% --------------------------------------------------
            % Eq. (7)
            % ---------------------------------------------------

            p(currentID,:) = p(currentID,:) + v(currentID,:);


            %% --------------------------------------------------
            % Boundary handling
            %
            % Here we follow your CSO platform's handling
            % ---------------------------------------------------

            p(currentID,:) = max(p(currentID,:),xmin);
            p(currentID,:) = min(p(currentID,:),xmax);


            %% --------------------------------------------------
            % Fitness Evaluation
            % ---------------------------------------------------

            fitness(currentID) = ...
                ComputeFitness(p(currentID,:)',FuncId);

            FEs = FEs + 1;


            %% --------------------------------------------------
            % Update global best
            % ---------------------------------------------------

            if fitness(currentID) < bestever

                bestever = fitness(currentID);

                gbestx = p(currentID,:);

            end


            gbesthistory(FEs) = bestever;


            %% Print once every 1000 evaluations
            if mod(FEs,1000) == 0

                fprintf(['DLLSO  FE %d  ' ...
                    'best = %e, NL = %d\n'],...
                    FEs,bestever,NL);

            end


            %% --------------------------------------------------
            % Stop immediately once the maximum number of FEs is reached
            % ---------------------------------------------------

            if FEs >= MaxFEs

                stopFlag = true;

                break;

            end

        end


        if stopFlag
            break;
        end

    end


    if stopFlag

        break;

    end


    %% ==========================================================
    % 6. Update the second level L2 separately
    %
    % Algorithm 1 Lines 22-32
    %
    % L2 has no two distinct higher levels to choose from,
    % so:
    %
    % Both exemplars are randomly selected from L1.
    %
    % The one with better fitness serves as exemplar1,
    % the worse one as exemplar2.
    %
    % ==========================================================

    level2Particles = levels{2};

    level1Particles = levels{1};


    for jj = 1:length(level2Particles)

        currentID = level2Particles(jj);


        %% ------------------------------------------------------
        % Randomly select two distinct particles from the first level
        % -------------------------------------------------------

        selectedParticles = randperm(length(level1Particles),2);

        exemplar1 = level1Particles(selectedParticles(1));
        exemplar2 = level1Particles(selectedParticles(2));


        %% ------------------------------------------------------
        % Algorithm 1:
        %
        % The better particle serves as X_1,k1
        % The worse particle serves as X_1,k2
        % -------------------------------------------------------

        if fitness(exemplar2) < fitness(exemplar1)

            temp = exemplar1;
            exemplar1 = exemplar2;
            exemplar2 = temp;

        end


        %% ------------------------------------------------------
        % Eq. (6)
        % -------------------------------------------------------

        r1 = rand(1,dimension);
        r2 = rand(1,dimension);
        r3 = rand(1,dimension);

        v(currentID,:) = ...
            r1 .* v(currentID,:) ...
            + r2 .* (p(exemplar1,:) - p(currentID,:)) ...
            + phi .* r3 .* (p(exemplar2,:) - p(currentID,:));


        %% Eq. (7)

        p(currentID,:) = p(currentID,:) + v(currentID,:);


        %% Boundary handling consistent with the platform

        p(currentID,:) = max(p(currentID,:),xmin);
        p(currentID,:) = min(p(currentID,:),xmax);


        %% Fitness Evaluation

        fitness(currentID) = ...
            ComputeFitness(p(currentID,:)',FuncId);

        FEs = FEs + 1;


        %% global best

        if fitness(currentID) < bestever

            bestever = fitness(currentID);

            gbestx = p(currentID,:);

        end


        gbesthistory(FEs) = bestever;


        if mod(FEs,1000) == 0

            fprintf(['DLLSO  FE %d  ' ...
                'best = %e, NL = %d\n'],...
                FEs,bestever,NL);

        end


        if FEs >= MaxFEs

            stopFlag = true;

            break;

        end

    end


    %% ==========================================================
    % Note:
    %
    % The first level L1 is not updated at all.
    %
    % According to the paper:
    %
    % "particles in the first level directly enter
    %  the next generation"
    %
    % Therefore there is no L1 update code here.
    % ==========================================================


    %% ==========================================================
    % 7. Eq. (8)
    %
    % Update the performance record r_i of the NL that was just selected
    %
    %                 |F_old - F_new|
    % r_i = --------------------------------
    %                       |F_old|
    %
    % The records of the other candidate NLs remain unchanged.
    %
    % ==========================================================

    if oldBest ~= 0

        R(selectedIndex) = ...
            abs(oldBest-bestever) / abs(oldBest);

    else

        % Once 0 has been reached, the relative improvement rate can no longer be computed.
        % Prevents 0/0 from producing NaN.
        R(selectedIndex) = 0;

    end


    gen = gen + 1;


    if stopFlag
        break;
    end

end


%% ==============================================================
% Handling the convergence history length
% ==============================================================

if FEs < MaxFEs

    gbesthistory(FEs+1:MaxFEs) = bestever;

elseif FEs > MaxFEs

    gbesthistory(MaxFEs+1:end) = [];

end


end