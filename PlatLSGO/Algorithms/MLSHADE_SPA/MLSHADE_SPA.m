function [gbestx,bestever,gbesthistory]=MLSHADE_SPA(mainHandle,popsize,dimension,xmax,xmin,vmax,vmin,maxiter,fCalculation,FuncId,VisualSwitch)
% =========================================================================
% MLSHADE-SPA -- platform-adapted version based on the authors' MATLAB code
%
% Original algorithm:
%   A. A. Hadi, A. W. Mohamed, K. M. Jambi,
%   "LSHADE-SPA memetic framework for solving large-scale optimization
%   problems", Complex & Intelligent Systems, 5 (2019), 25-40.
%
% Platform interface follows:
%   [gbestx,bestever,gbesthistory] = Algorithm(mainHandle,popsize,...
%       dimension,xmax,xmin,vmax,vmin,maxiter,fCalculation,FuncId,VisualSwitch)
%
% Adaptation rules:
%   1) Internal MLSHADE-SPA population remains NP=250 and is reduced to 20
%      exactly as in the released source code.
%   2) MaxFEs is fixed at 3e6, matching both the released MLSHADE-SPA code
%      and the platform comparison setting shown by the user.
%   3) Objective calls are changed from benchmark_func(...) to
%      fCalculation(x',FuncId).
%   4) gbesthistory records the best-so-far value at every actual FE.
%   5) The MMTS local FE loop uses an exact < budget condition to prevent
%      the released source's one-evaluation bookkeeping overshoot.
%
% mainHandle, popsize, vmax, vmin, maxiter and VisualSwitch are retained only
% for compatibility with the user's unified experiment interface.
% =========================================================================

ComputeFitness = fCalculation;
D = dimension;

FEs = 0;
MaxFEs = 3e6;
%MaxFEs = 10000 * dimension;

% Preserve platform-compatible inputs without changing the authors'
% algorithm-specific NP and FE settings.
mainHandle = mainHandle; %#ok<NASGU>
popsize = popsize; %#ok<NASGU>
vmax = vmax; %#ok<NASGU>
vmin = vmin; %#ok<NASGU>
maxiter = maxiter; %#ok<NASGU>
VisualSwitch = VisualSwitch; %#ok<NASGU>

% Bounds: support either scalars or 1-by-D vectors.
if isscalar(xmin)
    L = xmin * ones(1,D);
else
    L = xmin(:)';
end
if isscalar(xmax)
    H = xmax * ones(1,D);
else
    H = xmax(:)';
end
if numel(L) ~= D || numel(H) ~= D
    error('MLSHADE_SPA:BoundSize','xmin/xmax must be scalar or length dimension.');
end
lu = [L;H];

% Authors' population settings.
NP = 250;
max_NP = NP;
min_NP = 20.0;

bestever = inf;
gbestx = zeros(1,D);
gbesthistory = inf(MaxFEs,1);

% Initial population.
Pop = repmat(lu(1,:),NP,1) + rand(NP,D) .* ...
      repmat(lu(2,:)-lu(1,:),NP,1);

Fit = EvalPop(Pop);

% If a nonstandard call ever makes the FE budget smaller than initialization.
if FEs >= MaxFEs
    gbesthistory(MaxFEs+1:end) = [];
    return;
end

CC_nfes = MaxFEs/50;

Par = [];
Par.stgCount = zeros(NP,1);
Par.CRNewFlags = zeros(NP,1);
Par.CR = zeros(NP,1);
Par.CRRatio = zeros(11,1);

Par.memory_size = 5;
Par.memory_sf = 0.5 .* ones(Par.memory_size,1);
Par.memory_pos = 1;

archive = [];
archive.NP = NP;
archive.Pop = [];
archive.funvalues = [];

% The released source keeps a separate allocated-FE counter. It begins at NP.
nfes = NP;
flag = 0;

while nfes < MaxFEs

    Par.GenRatio = nfes/MaxFEs;

    EA_nfes = round(CC_nfes/2);
    EA_nfes = EA_nfes - mod(EA_nfes,NP);

    MMTS_nfes = round(CC_nfes/2);
    MMTS_nfes = MMTS_nfes - mod(MMTS_nfes,NP);

    %% EA part: full-dimensional LSHADE-SPA
    CC_Group_Ind = ones(1,D);

    Alg_fit = round(0.5*EA_nfes) - mod(round(0.5*EA_nfes),NP);
    if Alg_fit + nfes > MaxFEs
        Alg_fit = MaxFEs - nfes;
    end

    % If fewer than NP FEs remain, use MMTS to consume the exact residual
    % instead of evaluating and discarding a partial DE generation.
    if Alg_fit > 0 && Alg_fit < NP
        [Pop,Fit,Par] = MMTS(Alg_fit,Pop,Fit,lu,FuncId,CC_Group_Ind==1,Par,@EvalPop);
        nfes = nfes + Alg_fit;
        break;
    end

    [Pop,Fit,archive,Par] = CC_LSHADESPA(Alg_fit,Pop,Fit,lu,FuncId,...
        CC_Group_Ind==1,archive,Par,@EvalPop);
    nfes = nfes + Alg_fit;

    if nfes >= MaxFEs
        break;
    end

    %% Adaptive FE allocation among LSHADE-SPA, ANDE and EADE
    if flag == 0
        Alg_CC_nfes = (0.5*EA_nfes/3);
        LSHADESPA_CC_nfes = round(Alg_CC_nfes);
        ANDE_CC_nfes = round(Alg_CC_nfes);
        EADE_CC_nfes = 0.5*EA_nfes-(LSHADESPA_CC_nfes+ANDE_CC_nfes);
        flag = 1;
    else
        LSHADESPA_CC_nfes = round(0.9*LSHADESPA_CC_nfes + ...
            0.1*0.5*EA_nfes*All_Imp(1));
        ANDE_CC_nfes = round(0.9*ANDE_CC_nfes + ...
            0.1*0.5*EA_nfes*All_Imp(2));
        EADE_CC_nfes = 0.5*EA_nfes-(LSHADESPA_CC_nfes+ANDE_CC_nfes);
    end

    %% Randomly assign decision variables to three CC groups
    Group_No = 3;
    CC_Group_Ind = ceil(Group_No*rand(1,D));
    while length(unique(CC_Group_Ind)) ~= Group_No
        CC_Group_Ind = ceil(Group_No*rand(1,D));
    end

    %% CC-LSHADE-SPA
    Alg_Group_Ind = (CC_Group_Ind==1);
    LSHADESPA_CC_nfes = LSHADESPA_CC_nfes - mod(LSHADESPA_CC_nfes,NP);
    if LSHADESPA_CC_nfes + nfes > MaxFEs
        LSHADESPA_CC_nfes = MaxFEs-nfes;
    end

    if LSHADESPA_CC_nfes > 0 && LSHADESPA_CC_nfes < NP
        [Pop,Fit,Par] = MMTS(LSHADESPA_CC_nfes,Pop,Fit,lu,FuncId,...
            true(1,D),Par,@EvalPop);
        nfes = nfes + LSHADESPA_CC_nfes;
        break;
    end

    [Pop,LSHADESPA_Fit,archive,Par] = CC_LSHADESPA(LSHADESPA_CC_nfes,...
        Pop,Fit,lu,FuncId,Alg_Group_Ind,archive,Par,@EvalPop);
    nfes = nfes + LSHADESPA_CC_nfes;

    if nfes >= MaxFEs
        Fit = LSHADESPA_Fit;
        break;
    end

    %% CC-ANDE
    Alg_Group_Ind = (CC_Group_Ind==2);
    ANDE_CC_nfes = ANDE_CC_nfes - mod(ANDE_CC_nfes,NP);
    if ANDE_CC_nfes + nfes > MaxFEs
        ANDE_CC_nfes = MaxFEs-nfes;
    end

    if ANDE_CC_nfes > 0 && ANDE_CC_nfes < NP
        Fit = LSHADESPA_Fit;
        [Pop,Fit,Par] = MMTS(ANDE_CC_nfes,Pop,Fit,lu,FuncId,...
            true(1,D),Par,@EvalPop);
        nfes = nfes + ANDE_CC_nfes;
        break;
    end

    [Pop,ANDE_Fit,Par] = CC_ANDE(ANDE_CC_nfes,Pop,LSHADESPA_Fit,...
        lu,FuncId,Alg_Group_Ind,Par,@EvalPop);
    nfes = nfes + ANDE_CC_nfes;

    if nfes >= MaxFEs
        Fit = ANDE_Fit;
        break;
    end

    %% CC-EADE
    Alg_Group_Ind = (CC_Group_Ind==3);
    EADE_CC_nfes = EADE_CC_nfes - mod(EADE_CC_nfes,NP);
    if EADE_CC_nfes + nfes > MaxFEs
        EADE_CC_nfes = MaxFEs-nfes;
    end

    if EADE_CC_nfes > 0 && EADE_CC_nfes < NP
        Fit = ANDE_Fit;
        [Pop,Fit,Par] = MMTS(EADE_CC_nfes,Pop,Fit,lu,FuncId,...
            true(1,D),Par,@EvalPop);
        nfes = nfes + EADE_CC_nfes;
        break;
    end

    [Pop,EADE_Fit,Par] = CC_EADE(EADE_CC_nfes,Pop,ANDE_Fit,...
        lu,FuncId,Alg_Group_Ind,Par,@EvalPop);
    nfes = nfes + EADE_CC_nfes;

    if nfes >= MaxFEs
        Fit = EADE_Fit;
        break;
    end

    %% Calculate improvement ratio for each EA (authors' released formula)
    All_Imp = [];
    All_Imp(:,1) = Fit-LSHADESPA_Fit;
    All_Imp(:,2) = LSHADESPA_Fit-ANDE_Fit;
    All_Imp(:,3) = ANDE_Fit-EADE_Fit;
    All_Imp = sum(All_Imp);
    All_Imp = All_Imp./NP;

    if max(All_Imp) ~= 0
        All_Imp = All_Imp./[LSHADESPA_CC_nfes,ANDE_CC_nfes,EADE_CC_nfes];
        All_Imp = All_Imp./sum(All_Imp);
        [~,Imp_Ind] = sort(All_Imp);
        for imp_i = 1:length(All_Imp)-1
            All_Imp(Imp_Ind(imp_i)) = max(All_Imp(Imp_Ind(imp_i)),0.1);
        end
        All_Imp(Imp_Ind(end)) = 1-sum(All_Imp(Imp_Ind(1:end-1)));
    else
        Imp_Ind = 1:length(All_Imp); %#ok<NASGU>
        All_Imp(:) = 1/length(All_Imp);
    end
    Fit = EADE_Fit;

    %% MMTS
    CC_Group_Ind = ones(1,D);
    MMTS_Group_Ind = (CC_Group_Ind==1);
    if MMTS_nfes + nfes > MaxFEs
        MMTS_nfes = MaxFEs-nfes;
    end

    [Pop,Fit,Par] = MMTS(MMTS_nfes,Pop,Fit,lu,FuncId,...
        MMTS_Group_Ind,Par,@EvalPop);
    nfes = nfes + MMTS_nfes;

    if nfes >= MaxFEs
        break;
    end

    %% Linear population size reduction
    plan_NP = round((((min_NP-max_NP)/(0.5*MaxFEs))*nfes)+max_NP);

    if NP > plan_NP
        reduction_ind_num = NP-plan_NP;
        if NP-reduction_ind_num < min_NP
            reduction_ind_num = NP-min_NP;
        end
        NP = NP-reduction_ind_num;

        for r = 1:reduction_ind_num
            [~,indBest] = sort(Fit,'ascend');
            worst_ind = indBest(end);
            Pop(worst_ind,:) = [];
            Fit(worst_ind,:) = [];
            Par.stgCount(worst_ind) = [];
            Par.CRNewFlags(worst_ind) = [];
            Par.CR(worst_ind) = [];
        end

        archive.NP = NP;
        if size(archive.Pop,1) > archive.NP
            rndpos = randperm(size(archive.Pop,1));
            rndpos = rndpos(1:archive.NP);
            archive.Pop = archive.Pop(rndpos,:);
        end
    end
end

% Exact platform output length.
if FEs < MaxFEs
    if FEs == 0
        gbesthistory(:) = bestever;
    else
        gbesthistory(FEs+1:MaxFEs) = bestever;
    end
elseif FEs > MaxFEs
    gbesthistory(MaxFEs+1:end) = [];
end

% -------------------------------------------------------------------------
% Nested population evaluator. Every row of X is one candidate.
% It centralizes the actual FE counter and best-so-far history.
% -------------------------------------------------------------------------
    function Fits = EvalPop(X)
        n = size(X,1);
        Fits = inf(n,1);

        for ee = 1:n
            if FEs >= MaxFEs
                return;
            end

            value = ComputeFitness(X(ee,:)',FuncId);
            value = value(1);
            FEs = FEs+1;
            Fits(ee) = value;

            if value < bestever
                bestever = value;
                gbestx = X(ee,:);
            end

            gbesthistory(FEs) = bestever;

            if mod(FEs, floor(MaxFEs/10)) == 0 && FEs <= MaxFEs
                fprintf('MLSHADE-SPA算法,第%d次评价，最佳适应度 = %e\n',...
                    FEs,bestever);
            end
        end
    end
end


% =========================================================================
% Authors' component routines, adapted only to receive EvalPop.
% =========================================================================



% In this version we test the impact of JADE in our CC framework
function [PopOld,Fit,archive,Par]= CC_LSHADESPA(CC_nfes,Pop,Fit,lu,func_num,CC_Alg_Ind,archive,Par,EvalPop)
global initial_flag
initial_flag=initial_flag;

lu=lu(:,CC_Alg_Ind);

PopOld=Pop;

Eval_Pop=Pop;

Pop=Pop(:,CC_Alg_Ind);

[NP,D]=size(Pop);

X = zeros(NP,D); % trial vector

p_best_rate = 0.1;

stgCount=Par.stgCount;
CRNewFlags=Par.CRNewFlags;
CR=Par.CR;
CRRatio=Par.CRRatio;

memory_size=Par.memory_size;
memory_sf=Par.memory_sf;
memory_pos=Par.memory_pos;

nfes=0;

while nfes<CC_nfes
    
    mem_rand_index = ceil(memory_size * rand(NP, 1));
    mu_sf = memory_sf(mem_rand_index);
    
    [A,CR,stgCount]=Cr_Adaptation(CRNewFlags,Par.GenRatio,stgCount,CRRatio,CR);
    
    [temp_fit, sorted_index] = sort(Fit, 'ascend');
    
    %% for generating scaling factor
    if(nfes <= CC_nfes/2)
        sf=0.45+.1*rand(NP, 1);
        pos = find(sf <= 0);
        
        while ~ isempty(pos)
            sf(pos)=0.45+0.1*rand(length(pos), 1);
            pos = find(sf <= 0);
        end
    else
        sf = mu_sf + 0.1 * tan(pi * (rand(NP, 1) - 0.5));
        
        pos = find(sf <= 0);
        
        while ~ isempty(pos)
            sf(pos) = mu_sf(pos) + 0.1 * tan(pi * (rand(length(pos), 1) - 0.5));
            pos = find(sf <= 0);
        end
    end
    sf = min(sf, 1);
    
    r0 = [1 : NP];
    if(size(archive.Pop,1)~=0)
    Arc_pop=archive.Pop(:,CC_Alg_Ind);
    popAll = [Pop; Arc_pop];
    else
        popAll = Pop;
    end
    [r1, r2] = gnR1R2(NP, size(popAll, 1), r0);
    
    pNP = max(round(p_best_rate * NP), 2); %% choose at least two best solutions
    randindex = ceil(rand(1, NP) .* pNP); %% select from [1, 2, 3, ..., pNP]
    randindex = max(1, randindex); %% to avoid the problem that rand = 0 and thus ceil(rand) = 0
    pbest = Pop(sorted_index(randindex), :); %% randomly choose one of the top 100p% solutions
    
    X = Pop+ sf(:, ones(1, D)) .* (pbest - Pop + Pop(r1, :) - popAll(r2, :));

    X = boundConstraint(X, Pop, lu);
    
    mask = rand(NP, D) > CR(:, ones(1, D)); %mask is used to indicate which elements of ui comes from the parent
    Rnd=ceil(D* rand(NP, 1)); %choose one position where the element of X doesn't come from the parent
    jrand = sub2ind([NP D], (1:NP)', Rnd);
    mask(jrand)=false;
    X(mask) = Pop(mask);
                
    Eval_Pop(:,CC_Alg_Ind)=X;
    Child_Fit = EvalPop(Eval_Pop);
    
    nfes = nfes+NP;
    if nfes > CC_nfes;
        PopOld(:,CC_Alg_Ind)=Pop;
        Par.stgCount=stgCount;
        Par.CRNewFlags=CRNewFlags;
        Par.CR=CR;
        Par.CRRatio=CRRatio;
        
        Par.memory_size=memory_size;
        Par.memory_sf=memory_sf;
        Par.memory_pos=memory_pos;
        return;
    end
    
    %% JADE Archive & F Update
    Fit_imp_inf = (Child_Fit<=Fit);

    goodF = sf(Fit_imp_inf);
    dif = abs(Fit - Child_Fit);
    dif_val = dif(Fit_imp_inf);
    
    Eval_Pop(:,CC_Alg_Ind)=Pop;
    archive = updateArchive(archive, Eval_Pop(Fit_imp_inf, :), Fit(Fit_imp_inf));

    num_success_params = numel(goodF);
    if num_success_params > 0
        sum_dif = sum(dif_val);
        dif_val = dif_val / sum_dif;
        
        %% for updating the memory of scaling factor
        memory_sf(memory_pos) = (dif_val' * (goodF .^ 2)) / (dif_val' * goodF);
        
        memory_pos = memory_pos + 1;
        if memory_pos > memory_size;  memory_pos = 1; end
    end

    %% EADE CR and Pop Update
    CRNewFlags(Fit_imp_inf)=1;
    CRNewFlags(~Fit_imp_inf)=0;
    
    val= Child_Fit./Fit;
    
    val=1-val;
    
    Pop(Fit_imp_inf,:) = X(Fit_imp_inf,:); % replace current by trial
    Fit(Fit_imp_inf) = Child_Fit(Fit_imp_inf) ;
    
    
    for j=1:length(A)
        A_ind=A(j)==CR;
        CRRatio(j)= CRRatio(j)+sum(val(and(A_ind,Fit_imp_inf)));
    end
    
%     fprintf('FES_Perc:%.2f%% FES_No:%1.3e FitiBest:%1.3e SPA\n',(Cnfes+nfes)*100/3.00E+06,Cnfes+nfes,min(Fit));
    
end

PopOld(:,CC_Alg_Ind)=Pop;
Par.stgCount=stgCount;
Par.CRNewFlags=CRNewFlags;
Par.CR=CR;
Par.CRRatio=CRRatio;

Par.memory_size=memory_size;
Par.memory_sf=memory_sf;
Par.memory_pos=memory_pos;
end


function [PopOld,Fit,Par]= CC_ANDE(CC_nfes,Pop,Fit,lu,func_num,CC_Alg_Ind,Par,EvalPop)
global initial_flag
initial_flag=initial_flag;

lu=lu(:,CC_Alg_Ind);

PopOld=Pop;
Eval_Pop=Pop;

Pop=Pop(:,CC_Alg_Ind);

[NP,D]=size(Pop);


X = zeros(NP,D); % trial vector

stgCount=Par.stgCount;
CRNewFlags=Par.CRNewFlags;
CR=Par.CR;
CRRatio=Par.CRRatio;

nfes=0;

while nfes<CC_nfes
    
    
    [A,CR,stgCount]=Cr_Adaptation(CRNewFlags,Par.GenRatio,stgCount,CRRatio,CR);
    
    R = Gen_R(NP,3);
    R(:,1)=[];
    fr=Fit(R);
    [B,I] = sort(fr,2);
    R_S=[];
    for i=1:NP
        R_S(i,:)=R(i,I(i,:));
    end
    rb=R_S(:,1);
    rm=R_S(:,2);
    rw=R_S(:,3);
    
    F = 0.20+0.6*rand(NP,D);
    %     F = repmat(F,1,D);
    
    p1 = ones(NP,1);
    p2 = 0.75 + 0.25*rand(NP,1);
    p3 = 0.50 + 0.25*rand(NP,1);
    
    p=[p1 p2 p3];
    w=p./repmat(sum(p,2),1,3);
    
    w1= repmat(w(:,1),1,D);
    w2= repmat(w(:,2),1,D);
    w3= repmat(w(:,3),1,D);
    
    
    X = w1.*Pop(rb, :)+w2.*Pop(rm, :)+ w3.*Pop(rw, :)...
        + 2*F.*(Pop(rb, :) - Pop(rw, :));
    
    
    X = boundConstraint(X, Pop, lu);
    
    mask = rand(NP, D) > CR(:, ones(1, D)); %mask is used to indicate which elements of ui comes from the parent
    Rnd=ceil(D* rand(NP, 1)); %choose one position where the element of X doesn't come from the parent
    jrand = sub2ind([NP D], (1:NP)', Rnd);
    mask(jrand)=false;
    X(mask) = Pop(mask);
    
    Eval_Pop(:,CC_Alg_Ind)=X;
    Child_Fit = EvalPop(Eval_Pop);
    
    nfes = nfes+NP;
    if nfes > CC_nfes;
        PopOld(:,CC_Alg_Ind)=Pop;
        Par.stgCount=stgCount;
        Par.CRNewFlags=CRNewFlags;
        Par.CR=CR;
        Par.CRRatio=CRRatio;
        return;
    end
    
    Fit_imp_inf = (Child_Fit<=Fit);
    
    CRNewFlags(Fit_imp_inf)=1;
    CRNewFlags(~Fit_imp_inf)=0;
    
    val= Child_Fit./Fit;
    
    val=1-val;
    
    Pop(Fit_imp_inf,:) = X(Fit_imp_inf,:); % replace current by trial
    Fit(Fit_imp_inf) = Child_Fit(Fit_imp_inf) ;
    
    
    for j=1:length(A)
        A_ind=A(j)==CR;
        CRRatio(j)= CRRatio(j)+sum(val(and(A_ind,Fit_imp_inf)));
    end
    
%     fprintf('FES_Perc:%.2f%% FES_No:%1.3e\tFitiBest:%1.3e Tri\n',(Cnfes+nfes)*100/3.00E+06,Cnfes+nfes,min(Fit));
    
end

PopOld(:,CC_Alg_Ind)=Pop;
Par.stgCount=stgCount;
Par.CRNewFlags=CRNewFlags;
Par.CR=CR;
Par.CRRatio=CRRatio;

end


% In this version we test the impact of the original EADE as is with in our CC framework
function [PopOld,Fit,Par]= CC_EADE(CC_nfes,Pop,Fit,lu,func_num,CC_Alg_Ind,Par,EvalPop)
global initial_flag
initial_flag=initial_flag;

format long;

lu=lu(:,CC_Alg_Ind);

PopOld=Pop;

Eval_Pop=Pop;

Pop=Pop(:,CC_Alg_Ind);

[NP,D]=size(Pop);


X = zeros(NP,D); % trial vector

stgCount=Par.stgCount;
CRNewFlags=Par.CRNewFlags;
CR=Par.CR;
CRRatio=Par.CRRatio;

nfes=0;

while nfes<CC_nfes
    
    
    [A,CR,stgCount]=Cr_Adaptation(CRNewFlags,Par.GenRatio,stgCount,CRRatio,CR);
    
    mut_prop=rand(NP,1)<=0.5; % Both
    
    r=genR_EADE(Fit);
    
    F1=rand(NP,1);
    F2=rand(NP,1);
    
    X(r(mut_prop,1),:)=Pop(r(mut_prop,4),:) + F1(mut_prop, ones(1, D)).*(Pop(r(mut_prop,2),:)-(Pop(r(mut_prop,4),:))) + F2(mut_prop, ones(1, D)).*((Pop(r(mut_prop,4),:))-(Pop(r(mut_prop,3),:)));
    
    r=Gen_R(NP,4);
    
    temp=sum(~mut_prop);
    
    F=rand(temp,1);
    F(F>0.5)=-1.*F(F>0.5);
    
    F1(~mut_prop)=F;
    
    X(r(~mut_prop,1),:) = Pop(r(~mut_prop,4),:) + F1(~mut_prop, ones(1, D)).* (Pop(r(~mut_prop,2),:) - Pop(r(~mut_prop,3),:));
    
    mask = rand(NP, D) > CR(:, ones(1, D)); %mask is used to indicate which elements of ui comes from the parent
    Rnd=ceil(D* rand(NP, 1)); %choose one position where the element of X doesn't come from the parent
    jrand = sub2ind([NP D], (1:NP)', Rnd);
    mask(jrand)=false;
    X(mask) = Pop(mask);
    
    Temp_Pop = repmat(lu(1, :), NP, 1) + rand(NP, D) .* (repmat(lu(2, :) - lu(1, :), NP, 1));
    
    %% check the lower bound
    xl = repmat(lu(1, :), NP, 1);
    pos = X < xl;
    X(pos)= Temp_Pop(pos);
    
    %% check the upper bound
    xu = repmat(lu(2, :), NP, 1);
    pos = X > xu;
    X(pos)= Temp_Pop(pos);
    
    Eval_Pop(:,CC_Alg_Ind)=X;
    Child_Fit = EvalPop(Eval_Pop);
    
    nfes = nfes+NP;
    if nfes > CC_nfes;
        PopOld(:,CC_Alg_Ind)=Pop;
        Par.stgCount=stgCount;
        Par.CRNewFlags=CRNewFlags;
        Par.CR=CR;
        Par.CRRatio=CRRatio;
        return;
    end
    
    
    Fit_imp_inf = (Child_Fit<=Fit);
    
    CRNewFlags(Fit_imp_inf)=1;
    CRNewFlags(~Fit_imp_inf)=0;
    
    val= Child_Fit./Fit;
    
    val=1-val;
    
    Pop(Fit_imp_inf,:) = X(Fit_imp_inf,:); % replace current by trial
    Fit(Fit_imp_inf) = Child_Fit(Fit_imp_inf) ;
    
    
    for j=1:length(A)
        A_ind=A(j)==CR;
        CRRatio(j)= CRRatio(j)+sum(val(and(A_ind,Fit_imp_inf)));
    end
    
    %     fprintf('FES_Perc:%.2f%% FES_No:%1.3e FitiBest:%1.3e EADE\n',(Cnfes+nfes)*100/3.00E+06,Cnfes+nfes,min(Fit));
    
end

PopOld(:,CC_Alg_Ind)=Pop;
Par.stgCount=stgCount;
Par.CRNewFlags=CRNewFlags;
Par.CR=CR;
Par.CRRatio=CRRatio;

end


function [PopOld,Fit,Par]= MMTS(CC_nfes,Pop,Fit,lu,func_num,CC_Alg_Ind,Par,EvalPop)
global initial_flag
initial_flag=initial_flag;

Lbound=lu(1,CC_Alg_Ind);
Ubound=lu(2,CC_Alg_Ind);

PopOld=Pop;
Eval_Pop=Pop;
Pop=Pop(:,CC_Alg_Ind);
[NP,D]=size(Pop);

%%%%%% select the MMTS search agents from the DE population by Clearing procedure
[~,in]=sort(Fit);
LS_ind = in(1);
Eval_Pop=Eval_Pop(LS_ind,:);
LS_Pop=Pop(LS_ind,:);
LS_Fit=Fit(LS_ind);
LS_SR=(max(Pop,[],1)-min(Pop,[],1)).*rand(1,D);
LS_SR=min(LS_SR,0.2*(Ubound(1,1:D)-Lbound(1,1:D)));

dim=randperm(D);

nfes=0;

LS_Imp_Flag=1;

while (nfes<CC_nfes)
    LS_Last_Fit=LS_Fit;
    if LS_Imp_Flag==0
        LS_SR=LS_SR.*rand(1,D);
    end
    for i=1:D
        k=0;
        LS_Flag=1;
        while LS_Flag
            k=k+1;
            LS_Child_pos=LS_Pop;
            LS_Child_pos(dim(i))=LS_Child_pos(dim(i))+k*LS_SR(dim(i));
            if LS_Child_pos(dim(i))>Ubound(i)
                break;
            end
            Eval_Pop(CC_Alg_Ind)=LS_Child_pos;
            LS_Child_fit = EvalPop(Eval_Pop);
            nfes=1+nfes;
            if LS_Child_fit<=LS_Fit
                LS_Fit=LS_Child_fit;
                LS_Pop=LS_Child_pos;
            else
                LS_Flag=0;
            end
            
        end
        if k<=1
            k=0;
            LS_Flag=1;
            while LS_Flag;
                k=k+1;
                LS_Child_pos=LS_Pop;
                LS_Child_pos(dim(i))=LS_Child_pos(dim(i))-k*LS_SR(dim(i));
                if LS_Child_pos(dim(i))<Lbound(i)
                    break;
                end
                Eval_Pop(CC_Alg_Ind)=LS_Child_pos;
                LS_Child_fit = EvalPop(Eval_Pop);
                nfes=1+nfes;
                if LS_Child_fit<=LS_Fit
                    LS_Fit=LS_Child_fit;
                    LS_Pop=LS_Child_pos;
                else
                    LS_Flag=0;
                end
                
            end
        end
    end
    if LS_Last_Fit<=LS_Fit
        LS_Imp_Flag=0;
    else
        LS_Imp_Flag=1;
    end
end


Fit(LS_ind)=LS_Fit;
Pop(LS_ind,:)=LS_Pop;
PopOld(:,CC_Alg_Ind)=Pop;
end


function [A,CR,stgCount]=Cr_Adaptation(CRNewFlags,GenRatio,stgCount,CRRatio,CRs)

if(GenRatio<=(1/10))
    if(GenRatio<=(1/60))
        A=[0.05 0.1];
    elseif(GenRatio<=(1/40) && GenRatio>(1/60))
        A=[0.05 0.1 0.2 0.3];
    elseif(GenRatio<=(1/30) && GenRatio>(1/40))
        A=[0.05 0.1 0.2 0.3 0.4 0.5];
    elseif(GenRatio<=(1/24) && GenRatio>(1/30))
        A=[0.05 0.1 0.2 0.3 0.4 0.5 0.6 0.7];
    elseif(GenRatio<=(1/20) && GenRatio>(1/24))
        A=[0.05 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9];
    elseif(GenRatio<=(1/10) && GenRatio>(1/20))
        A=[0.05 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 0.95];
    end
else
    A=[0.05 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 0.95];
end


CR=CRs;
CR_New_ind=(CRNewFlags==0);

if(sum(CR_New_ind)>0)
    if(GenRatio<=(1/10))
        paraIndex=ceil(length(A)* rand(sum(CR_New_ind), 1));
        CR(CR_New_ind)=A(paraIndex);
    else
        stgCount(CR_New_ind)=stgCount(CR_New_ind)+1;
        stgCount_ind=(stgCount==21);
        paraIndex=ceil(length(A)* rand(sum(stgCount_ind), 1));
        stgCount(stgCount_ind)=0;
        CR(stgCount_ind)=A(paraIndex);
    end
end

CR_Imp_ind=(CRNewFlags==1);

if(sum(CR_Imp_ind)>0)
    CR(CR_Imp_ind)=A(abs(CRRatio)==max(abs(CRRatio)));
end


end




function vi = boundConstraint (vi, pop, lu)

% if the boundary constraint is violated, set the value to be the middle
% of the previous value and the bound
%
% Version: 1.1   Date: 11/20/2007
% Written by Jingqiao Zhang, jingqiao@gmail.com

[NP, D] = size(pop);  % the population size and the problem's dimension

%% check the lower bound
xl = repmat(lu(1, :), NP, 1);
pos = vi < xl;
vi(pos) = (pop(pos) + xl(pos)) / 2;

%% check the upper bound
xu = repmat(lu(2, :), NP, 1);
pos = vi > xu;
vi(pos) = (pop(pos) + xu(pos)) / 2;
end


function archive = updateArchive(archive, pop, funvalue)
% Update the archive with input solutions
%   Step 1: Add new solution to the archive
%   Step 2: Remove duplicate elements
%   Step 3: If necessary, randomly remove some solutions to maintain the archive size
%
% Version: 1.1   Date: 2008/04/02
% Written by Jingqiao Zhang (jingqiao@gmail.com)

if archive.NP == 0, return; end

if size(pop, 1) ~= size(funvalue,1), error('check it'); end

% Method 2: Remove duplicate elements
popAll = [archive.Pop; pop ];
funvalues = [archive.funvalues; funvalue ];
[~, IX]= unique(popAll, 'rows');
if length(IX) < size(popAll, 1) % There exist some duplicate solutions
  popAll = popAll(IX, :);
  funvalues = funvalues(IX, :);
end

if size(popAll, 1) <= archive.NP   % add all new individuals
  archive.Pop = popAll;
  archive.funvalues = funvalues;
else                % randomly remove some solutions
  rndpos = randperm(size(popAll, 1)); % equivelent to "randperm";
  rndpos = rndpos(1 : archive.NP);
  
  archive.Pop = popAll  (rndpos, :);
  archive.funvalues = funvalues(rndpos, :);
end
end



function R = Gen_R(NP_Size,N)

% Gen_R generate N column vectors r1, r2, ..., rN of size NP_Size
%    R's elements are choosen from {1, 2, ..., NP_Size} & R(j,i) are unique per row

% Call:
%    [R] = Gen_R(NP_Size)   % N is set to be 1;
%    [R] = Gen_R(NP_Size,N) 
%
% Version: 0.1  Date: 2018/02/01
% Written by Anas A. Hadi (anas1401@gmail.com)


R(1,:)=1:NP_Size;

for i=2:N+1
    
    R(i,:) = ceil(rand(NP_Size,1) * NP_Size);
    
    flag=0;
    while flag ~= 1
        pos = (R(i,:) == R(1,:));
        for w=2:i-1
            pos=or(pos,(R(i,:) == R(w,:)));
        end
        if sum(pos) == 0
            flag=1;
        else
            R(i,pos)= floor(rand(sum(pos),1 ) * NP_Size) + 1;
        end
    end
end

R=R';

end



function r=genR_EADE(Fit)

NP=length(Fit);
r(:,1)=1:NP;

[srt, Fit_index]=sort(Fit,'ascend');

T=ceil(length(Fit_index)/10);
Best=Fit_index(1:T);
Mid=Fit_index(T+1:end-T);
Worest=Fit_index(end-T+1:end);

% choose three random individuals from population mutually different
r(:,2) = Best(ceil(length(Best)* rand(NP, 1)));

r(:,3) = Worest(ceil(length(Worest)* rand(NP, 1)));

r(:,4) = Mid(ceil(length(Mid)* rand(NP, 1)));

pos=r(:,2)==r(:,3);

while(sum(pos)~=0)
    r(pos,3) = Worest(ceil(length(Worest)* rand(sum(pos), 1)));
    pos=r(:,2)==r(:,3);
end

pos=r(:,3)==r(:,4);
while(sum(pos)~=0)
    r(pos,4) = Mid(ceil(length(Mid)* rand(sum(pos), 1)));
    pos=r(:,3)==r(:,4);
end

end




function [r1, r2] = gnR1R2(NP1, NP2, r0)

% gnA1A2 generate two column vectors r1 and r2 of size NP1 & NP2, respectively
%    r1's elements are choosen from {1, 2, ..., NP1} & r1(i) ~= r0(i)
%    r2's elements are choosen from {1, 2, ..., NP2} & r2(i) ~= r1(i) & r2(i) ~= r0(i)
%
% Call:
%    [r1 r2 ...] = gnA1A2(NP1)   % r0 is set to be (1:NP1)'
%    [r1 r2 ...] = gnA1A2(NP1, r0) % r0 should be of length NP1
%
% Version: 2.1  Date: 2008/07/01
% Written by Jingqiao Zhang (jingqiao@gmail.com)

NP0 = length(r0);

r1 = floor(rand(1, NP0) * NP1) + 1;
%for i = 1 : inf
for i = 1 : 99999999
    pos = (r1 == r0);
    if sum(pos) == 0
        break;
    else % regenerate r1 if it is equal to r0
        r1(pos) = floor(rand(1, sum(pos)) * NP1) + 1;
    end
    if i > 1000, % this has never happened so far
        error('Can not genrate r1 in 1000 iterations');
    end
end

r2 = floor(rand(1, NP0) * NP2) + 1;
%for i = 1 : inf
for i = 1 : 99999999
    pos = ((r2 == r1) | (r2 == r0));
    if sum(pos)==0
        break;
    else % regenerate r2 if it is equal to r0 or r1
        r2(pos) = floor(rand(1, sum(pos)) * NP2) + 1;
    end
    if i > 1000, % this has never happened so far
        error('Can not genrate r2 in 1000 iterations');
    end
end


end
