function [gbestx,bestever,gbesthistory] = DPCLSO(popsize,dimension,xmax,xmin,vmax,vmin,maxiter,fCalculation,FuncId)


ComputeFitness = fCalculation;

%% ================= Parameters ======================================

m = 600;

tau   = 0.01;
alpha = 0.1;
beta  = 0.6;
psi   = 0.3;
k     = 1.0;

MaxFEs = 3e6;
FEs = 0;

%% ================= Initialization ====================================

p = zeros(m,dimension);

% --------------------------------------------------------------
% The paper does not state how V(0) is initialized.
%
% Algorithm 1 only explicitly states that the population is randomly initialized in solution space.
% To avoid a large random initial velocity disrupting the early search, it is initialized to 0 here.
% --------------------------------------------------------------
v = zeros(m,dimension);

fitness = zeros(1,m);

for i = 1:m

    p(i,:) = xmin + (xmax-xmin).*rand(1,dimension);

    fitness(i) = ComputeFitness(p(i,:)',FuncId);

end

FEs = FEs + m;

%% ================= Initialize best ================================

[bestever,id] = min(fitness);

gbestx = p(id,:);

gbesthistory = zeros(MaxFEs,1);

gbesthistory(1:FEs) = bestever;

gen = 1;

%% ==============================================================
% Main loop
% ==============================================================

while FEs < MaxFEs

    %% ==========================================================
    % 1. Sort particles from best to worst
    %
    % Corresponds to Algorithm 1 Line 4
    %
    % Note:
    % Here the population is reordered directly,
    % so after sorting:
    %
    % p(1,:) = current best
    % p(2,:) = current second best
    % ...
    % p(m,:) = current worst
    %
    % The particle index itself is then the rank.
    % ==========================================================

    [fitness,order] = sort(fitness,'ascend');

    p = p(order,:);
    v = v(order,:);


    %% ==========================================================
    % 2. Dynamic BPS
    %
    % BPSmin = alpha * PS
    % BPSmax = beta  * PS
    %
    % The formatting of Eq.(5) in the paper is:
    %
    % BPSmax - round(BPSmax-BPSmin)*(FE/FEmax)^k
    %
    % but the main text also says that round is used to round the computed result.
    %
    % BPS itself is defined as "number of better particles",
    % so here the dynamic decrement is rounded so that BPS is always an integer.
    % ==========================================================

    BPSmin = round(alpha*m);
    BPSmax = round(beta*m);

    BPS = BPSmax - ...
        round((BPSmax-BPSmin)*(FEs/MaxFEs)^k);

    % Safety range
    BPS = max(BPSmin,min(BPSmax,BPS));


    %% ==========================================================
    % 3. Update probability
    %
    % Eq.(2)
    %
    % pro(i) = 1/[1+exp(-tau*(rank(i)-BPS))]
    %
    % The current population is already sorted, so rank(i)=i.
    % ==========================================================

    rankVector = 1:m;

    pro = 1 ./ ...
        (1 + exp(-tau*(rankVector-BPS)));


    %% ==========================================================
    % 4. Particle update
    %
    % Algorithm 1:
    %
    % for each particle i (rank(i)>2)
    %
    % Therefore the two best particles are not updated.
    %
    % Here the update is performed in place one by one in the sorted order,
    % consistent with the execution structure of Algorithm 1.
    % ==========================================================

    for i = 3:m

        if FEs >= MaxFEs
            break;
        end

        %% ------------------------------------------------------
        % Probability controlled update
        % -------------------------------------------------------

        if rand < pro(i)

            %% ==================================================
            % Better particle
            %
            % Algorithm:
            % rank(i) < BPS
            %
            % Both learning samples come from better particles
            % and are better than particle i.
            % ==================================================

            if i < BPS

                % ----------------------------------------------
                % ranks 1:i-1 are all better than the current particle i,
                % and they all belong to the better group as well.
                % ----------------------------------------------

                candidate = 1:i-1;

                rr = randperm(length(candidate),2);

                sample1 = candidate(rr(1));
                sample2 = candidate(rr(2));


                % ==============================================
                % Eq.(8)
                %
                % IMPORTANT:
                %
                % A random number is generated independently for each dimension.
                %
                % This is the more reasonable interpretation in high-dimensional PSO/LSO implementations,
                % rather than sharing a single random scalar across the whole 1000D.
                % ==============================================

                r1 = rand(1,dimension);
                r2 = rand(1,dimension);
                r3 = rand(1,dimension);


                v(i,:) = ...
                    r1 .* v(i,:) ...
                    + r2 .* (p(sample1,:) - p(i,:)) ...
                    + psi .* r3 .* ...
                    (p(sample2,:) - p(i,:));


            %% ==================================================
            % Worse particle
            %
            % One learning sample comes from the better group,
            % the other comes from the worse group.
            %
            % Algorithm 1 Lines 12-14.
            % ==================================================

            else

                %% ---------------------------------------------
                % learning sample 1:
                % better particle
                %
                % The paper uses rank < BPS to decide "better".
                % Therefore the better ranks are:
                %
                % 1,...,BPS-1
                % ----------------------------------------------

                betterEnd = BPS - 1;

                if betterEnd < 1
                    continue;
                end

                sample1 = randi(betterEnd);


                %% ---------------------------------------------
                % learning sample 2:
                % worse particle
                %
                % The paper's main text is ambiguous here:
                % if that worse particle is required to be better than i,
                % then for the first worse particle there is no solution.
                %
                % Algorithm 1 only requires "from worse particles".
                %
                % Therefore the choice here is made uniformly at random from the whole worse group,
                % but particle i itself cannot be chosen.
                % ----------------------------------------------

                worseStart = BPS;

                worseCandidate = worseStart:m;

                worseCandidate(worseCandidate == i) = [];

                if isempty(worseCandidate)
                    continue;
                end

                sample2 = worseCandidate( ...
                    randi(length(worseCandidate)) );


                %% =============================================
                % Eq.(6)
                % ==============================================

                r1 = rand(1,dimension);
                r2 = rand(1,dimension);
                r3 = rand(1,dimension);


                v(i,:) = ...
                    r1 .* v(i,:) ...
                    + r2 .* (p(sample1,:) - p(i,:)) ...
                    + psi .* r3 .* ...
                    (p(sample2,:) - p(i,:));

            end


            %% ==================================================
            % Eq.(7) / Eq.(9)
            %
            % X_i(t+1)=X_i(t)+V_i(t+1)
            % ==================================================

            p(i,:) = p(i,:) + v(i,:);


            %% ==================================================
            % Boundary handling
            %
            % The paper does not specify the boundary strategy.
            %
            % Here it is kept consistent with the existing experimental framework:
            % saturation / clipping
            % ==================================================

            p(i,:) = max(p(i,:),xmin);
            p(i,:) = min(p(i,:),xmax);


            %% ==================================================
            % Fitness evaluation
            % ==================================================

            fitness(i) = ComputeFitness(p(i,:)',FuncId);

            FEs = FEs + 1;


            %% ==================================================
            % Best-so-far
            % ==================================================

            if fitness(i) < bestever

                bestever = fitness(i);
                gbestx = p(i,:);

            end


            %% ==================================================
            % History
            % ==================================================

            if FEs <= MaxFEs
                gbesthistory(FEs) = bestever;
            end


            %% ==================================================
            % Print every 10%
            % ==================================================

            if mod(FEs, floor(MaxFEs/10)) == 0 && FEs <= MaxFEs

                fprintf(...
                    'DPCLSO  FE %d  best = %e\n',...
                    FEs,bestever);

            end

        end

    end

    gen = gen + 1;

end


%% ==============================================================
% Pad the history
% ==============================================================

if FEs < MaxFEs

    gbesthistory(FEs+1:MaxFEs) = bestever;

elseif FEs > MaxFEs

    gbesthistory(MaxFEs+1:end) = [];

end

end