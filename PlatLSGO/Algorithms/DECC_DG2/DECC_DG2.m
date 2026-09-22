function [gbestx,bestever,gbesthistory] = DECC_DG2(~,popsize,dimension,xmax,xmin,~,~,~,fCalculation,FuncId,~)
% DECC_DG2
% -------------------------------------------------------------------------
% DECC-DG2 adapted to the same interface style as the user's CSO code.
%
% Output:
%   gbestx       - best solution found (1 x dimension)
%   bestever     - best objective value found
%   gbesthistory - best-so-far trace, length = 3e6
%
% Main references:
%   1) Omidvar et al., Cooperative Co-evolution with Differential Grouping
%      for Large Scale Optimization, IEEE TEC, 2014.
%   2) Omidvar et al., DG2: A Faster and More Accurate Differential Grouping
%      for Large-Scale Black-Box Optimization, IEEE TEC, 2017.
%   3) Yang et al., Self-adaptive Differential Evolution with Neighborhood
%      Search, CEC 2008.
%
% Notes:
%   - DG2 is executed once per benchmark function if no cache exists.
%   - The grouping result is cached automatically under ./DG2_results/.
%   - The DG2 decomposition cost (evaluations.count) is counted in the
%     3e6 total FE budget every independent run, even when a cached grouping
%     result is loaded.
%   - The SaNSDE section follows the DECC_MDG source supplied by the user.
% -------------------------------------------------------------------------

    ComputeFitness = fCalculation;
    MaxFEs = 3e6;

    % The DECC/DG2 experiments use SaNSDE with population size 50.
    % Keep this fixed for paper-style reproduction.
    popsize = 50;

    % Convert scalar/vector bounds to row vectors.
    xminVec = local_expand_bound(xmin, dimension);
    xmaxVec = local_expand_bound(xmax, dimension);

    % Determine whether the current benchmark is CEC2010 or CEC2013.
    problem = local_detect_benchmark_year(ComputeFitness, FuncId, dimension);

    % Reset the standard CEC MATLAB benchmark initialization flag when used.
    global initial_flag;
    initial_flag = 0;

    % ---------------------------------------------------------------------
    % 1. DG2 decomposition (load cache or generate it)
    % ---------------------------------------------------------------------
    [group, decompositionFEs] = local_dg2_grouping( ...
        ComputeFitness, FuncId, problem, dimension, xminVec, xmaxVec);

    group_num = numel(group);
    if group_num == 0
        group = {1:dimension};
        group_num = 1;
    end

    % Reset benchmark state before the optimization phase.
    initial_flag = 0;

    % ---------------------------------------------------------------------
    % 2. Initial full-dimensional population
    % ---------------------------------------------------------------------
    Lbound = repmat(xminVec, popsize, 1);
    Ubound = repmat(xmaxVec, popsize, 1);

    pop = Lbound + rand(popsize, dimension) .* (Ubound - Lbound);
    val = local_eval_population(ComputeFitness, pop, FuncId);

    [bestval, ibest] = min(val);
    bestmem = pop(ibest, :);

    % The DG2 grouping budget is part of the 3e6 FE budget.
    FE = decompositionFEs + popsize;

    % Fitness trace. During the decomposition stage no optimizer best exists;
    % for platform compatibility we fill this prefix with the first bestval,
    % consistent with the style of the supplied DECC_MDG implementation.
    firstTraceEnd = min(FE, MaxFEs);
    gbesthistory = bestval * ones(firstTraceEnd, 1);

    % ---------------------------------------------------------------------
    % 3. Canonical cooperative co-evolution: round-robin components
    % ---------------------------------------------------------------------
    ccm = 0.5;
    sansde_iter = 1;
    Cycle = 0;

    while FE < MaxFEs

        for i = 1:group_num
            if FE >= MaxFEs
                break;
            end

            dim_index = group{i};
            if isempty(dim_index)
                continue;
            end

            subpop = pop(:, dim_index);
            subLbound = Lbound(:, dim_index);
            subUbound = Ubound(:, dim_index);

            [subpopnew, bestmemnew, bestvalnew, ~, ccm] = local_sansde( ...
                ComputeFitness, FuncId, dim_index, subpop, bestmem, bestval, ...
                subLbound, subUbound, sansde_iter, ccm, Cycle);

            pop(:, dim_index) = subpopnew;
            bestmem = bestmemnew;
            bestval = bestvalnew;

            % Keep the same FE convention as the supplied DECC_MDG code:
            % one SaNSDE generation consumes NP offspring evaluations.
            oldFE = FE;
            FE = FE + popsize;

    fprintf('DECC-DG2: FE = %d / %d, best = %e, Cycle = %d\n', ...
        FE, MaxFEs, bestval, Cycle);


            traceEnd = min(FE, MaxFEs);
            if traceEnd > oldFE
                gbesthistory(oldFE+1:traceEnd, 1) = bestval;
            end
        end



        Cycle = Cycle + 1;
    end

    % ---------------------------------------------------------------------
    % 4. Platform-compatible outputs
    % ---------------------------------------------------------------------
    if numel(gbesthistory) < MaxFEs
        gbesthistory(end+1:MaxFEs, 1) = bestval;
    elseif numel(gbesthistory) > MaxFEs
        gbesthistory(MaxFEs+1:end) = [];
    end

    gbestx = bestmem;
    bestever = bestval;
end


% =========================================================================
% Detect CEC benchmark year
% =========================================================================
function problem = local_detect_benchmark_year(fname, func_num, dimension)

    % Preferred manual override if the platform uses an ambiguous function
    % handle name such as benchmark_func for both suites.
    global LSGO_BENCHMARK_YEAR;
    if ~isempty(LSGO_BENCHMARK_YEAR)
        if any(LSGO_BENCHMARK_YEAR == [2010, 2013])
            problem = LSGO_BENCHMARK_YEAR;
            return;
        else
            error('LSGO_BENCHMARK_YEAR must be 2010 or 2013.');
        end
    end

    % Try to infer the suite from the function location/name.
    try
        fstr = func2str(fname);
        fpath = which(fstr);
        probe = lower([fstr, ' ', fpath]);
        if ~isempty(strfind(probe, '2010')) || ~isempty(strfind(probe, 'cec10')) %#ok<STREMP>
            problem = 2010;
            return;
        elseif ~isempty(strfind(probe, '2013')) || ~isempty(strfind(probe, 'cec13')) %#ok<STREMP>
            problem = 2013;
            return;
        end
    catch
        % Continue with unambiguous rules below.
    end

    % IDs 16-20 only exist in CEC2010 LSGO.
    if func_num > 15
        problem = 2010;
        return;
    end

    % CEC2013 f13/f14 official DG2 scripts use 905 dimensions.
    if dimension == 905
        problem = 2013;
        return;
    end

    error([ ...
        'DECC_DG2 cannot distinguish CEC2010 from CEC2013 automatically. ', ...
        'Before running the suite, add these two lines in the platform main file: ', ...
        'global LSGO_BENCHMARK_YEAR; LSGO_BENCHMARK_YEAR = 2010; ', ...
        'or set it to 2013.' ...
    ]);
end


% =========================================================================
% DG2 grouping: load a cached result or run DG2 once and save it
% =========================================================================
function [group, decompositionFEs] = local_dg2_grouping( ...
        fname, func_num, problem, dim, lb, ub)

    thisFile = mfilename('fullpath');
    thisDir = fileparts(thisFile);
    resultDir = fullfile(thisDir, 'DG2_results', sprintf('CEC%d', problem));

    if ~exist(resultDir, 'dir')
        mkdir(resultDir);
    end

    cacheFile = fullfile(resultDir, sprintf('F%02d_D%d.mat', func_num, dim));

    if exist(cacheFile, 'file')
        data = load(cacheFile);
        nonseps = data.nonseps;
        seps = data.seps;
        evaluations = data.evaluations;
    else
        fprintf('DECC-DG2: generating DG2 grouping for CEC%d F%02d (D=%d)...\n', ...
            problem, func_num, dim);

        opts.lbound = lb;
        opts.ubound = ub;
        opts.dim = dim;

        objfun = @(x) local_eval_one(fname, x, func_num);

        [delta, lambda, evaluations] = local_ism(objfun, opts);
        [nonseps, seps, theta, epsilon] = local_dsm(evaluations, lambda, dim);

        save(cacheFile, 'delta', 'lambda', 'evaluations', 'nonseps', ...
            'seps', 'theta', 'epsilon', '-v7');

        fprintf('DECC-DG2: grouping saved to %s\n', cacheFile);
        fprintf('DECC-DG2: DG2 decomposition FEs = %d\n', evaluations.count);
    end

    group = nonseps;
    if ~isempty(seps)
        group{end+1} = seps;
    end

    decompositionFEs = evaluations.count;
end


% =========================================================================
% DG2 - Interaction Structure Matrix (official DG2 logic, embedded locally)
% =========================================================================
function [delta, lambda, evaluations] = local_ism(fun, options)

    ub = local_expand_bound(options.ubound, options.dim);
    lb = local_expand_bound(options.lbound, options.dim);
    dim = options.dim;
    temp = (ub + lb) / 2;

    FEs = 0;
    f_archive = nan(dim, dim);
    fhat_archive = nan(dim, 1);
    delta1 = nan(dim, dim);
    delta2 = nan(dim, dim);
    lambda = nan(dim, dim);

    p1 = lb;
    fp1 = fun(p1);
    FEs = FEs + 1;

    counter = 0;
    totalPairs = dim * (dim - 1) / 2;
    lastPrinted = -1;

    for i = 1:dim-1

        if ~isnan(fhat_archive(i))
            fp2 = fhat_archive(i);
        else
            p2 = p1;
            p2(i) = temp(i);
            fp2 = fun(p2);
            FEs = FEs + 1;
            fhat_archive(i) = fp2;
        end

        for j = i+1:dim
            counter = counter + 1;

            % Light progress output for the expensive first decomposition.
            progressPct = floor(counter / totalPairs * 100);
            if mod(progressPct, 10) == 0 && progressPct ~= lastPrinted
                fprintf('DECC-DG2 grouping progress: %d%%\n', progressPct);
                lastPrinted = progressPct;
            end

            if ~isnan(fhat_archive(j))
                fp3 = fhat_archive(j);
            else
                p3 = p1;
                p3(j) = temp(j);
                fp3 = fun(p3);
                FEs = FEs + 1;
                fhat_archive(j) = fp3;
            end

            p4 = p1;
            p4(i) = temp(i);
            p4(j) = temp(j);
            fp4 = fun(p4);
            FEs = FEs + 1;

            f_archive(i,j) = fp4;
            f_archive(j,i) = fp4;

            d1 = fp2 - fp1;
            d2 = fp4 - fp3;

            delta1(i,j) = d1;
            delta2(i,j) = d2;
            lambda(i,j) = abs(d1 - d2);
        end
    end

    evaluations.base = fp1;
    evaluations.fhat = fhat_archive;
    evaluations.F = f_archive;
    evaluations.count = FEs;

    delta.delta1 = delta1;
    delta.delta2 = delta2;
end


% =========================================================================
% DG2 - Differential Grouping Structure Matrix
% =========================================================================
function [nonseps, seps, theta, epsilon] = local_dsm(evaluations, lambda, dim)

    fhat_archive = evaluations.fhat;
    f_archive = evaluations.F;
    fp1 = evaluations.base;

    F1 = ones(dim, dim) * fp1;
    F2 = repmat(fhat_archive', dim, 1);
    F3 = repmat(fhat_archive, 1, dim);
    F4 = f_archive;

    FS = cat(3, F1, F2, F3, F4);
    Fmax = max(FS, [], 3);

    FS2 = cat(3, F1 + F4, F2 + F3);
    Fmax_inf = max(FS2, [], 3);

    theta = nan(dim);

    muM = eps / 2;
    gamma = @(n) ((n .* muM) ./ (1 - n .* muM));
    errlb = gamma(2) * Fmax_inf;
    errub = gamma(dim^0.5) * Fmax;

    I1 = lambda <= errlb;
    theta(I1) = 0;

    I2 = lambda >= errub;
    theta(I2) = 1;

    I0 = (lambda == 0);
    c0 = sum(I0(:));
    tmpSep = (~I0 & I1);
    count_seps = sum(tmpSep(:));
    count_nonseps = sum(I2(:));
    reliable_calcs = count_seps + count_nonseps;

    denom = c0 + reliable_calcs;
    if denom == 0
        % Extremely defensive fallback; for CEC LSGO this branch should not
        % normally be reached.
        epsilon = 0;
    else
        w1 = (count_seps + c0) / denom;
        w2 = count_nonseps / denom;
        epsilon = w1 .* errlb + w2 .* errub;
    end

    AdjTemp = lambda > epsilon;

    idx = isnan(theta);
    theta(idx) = AdjTemp(idx);
    theta = theta | theta';
    theta(logical(eye(dim))) = 1;

    components = local_find_conn_comp(theta);

    sizeone = cellfun(@(x) length(x) == 1, components);
    seps = components(sizeone);
    if isempty(seps)
        seps = [];
    else
        seps = cell2mat(seps);
    end

    components(sizeone) = [];
    nonseps = components;
end


% =========================================================================
% Connected components used by DG2
% =========================================================================
function components = local_find_conn_comp(C)

    L = size(C,1);
    labels = zeros(1,L);
    ccc = 0;

    while true
        ind = find(labels == 0, 1, 'first');
        if isempty(ind)
            break;
        end

        list = ind;
        ccc = ccc + 1;
        labels(ind) = ccc;

        while ~isempty(list)
            list_new = [];
            for lc = 1:length(list)
                p = list(lc);
                cp = find(C(p,:));
                cp1 = cp(labels(cp) == 0);
                labels(cp1) = ccc;
                list_new = [list_new, cp1]; %#ok<AGROW>
            end
            list = list_new;
        end
    end

    group_num = max(labels);
    components = cell(1, group_num);
    for i = 1:group_num
        components{i} = find(labels == i);
    end
end


% =========================================================================
% SaNSDE component optimizer
% Based on the DECC_MDG source supplied by the user.
% =========================================================================
function [popnew, bestmemnew, bestvalnew, tracerst, ccm] = local_sansde( ...
        fname, func_num, dim_index, pop, bestmem, bestval, ...
        Lbound, Ubound, itermax, ccm, cycle) %#ok<INUSD>

    [popsize, dim] = size(pop);
    NP = popsize;
    D = dim;
    tracerst = [];

    F = zeros(NP,1);

    linkp = 0.5;
    l1 = 1; l2 = 1; nl1 = 1; nl2 = 1;

    fp = 0.5;
    ns1 = 1; nf1 = 1; ns2 = 1; nf2 = 1;

    pm1 = zeros(NP,D);
    pm2 = zeros(NP,D);
    pm3 = zeros(NP,D);
    pm4 = zeros(NP,D);
    pm5 = zeros(NP,D); %#ok<NASGU>
    bm  = zeros(NP,D);
    ui  = zeros(NP,D);
    mui = zeros(NP,D);
    mpo = zeros(NP,D);

    rot = 0:NP-1;

    cc_rec = [];
    f_rec = [];

    % Context-vector evaluation of the current component population.
    gpop = repmat(bestmem, popsize, 1);
    gpop(:, dim_index) = pop;
    val = local_eval_population(fname, gpop, func_num);

    [best, ibest] = min(val);
    subbestmem = pop(ibest, :);

    if best < bestval
        bestval = best;
        bestmem = gpop(ibest, :);
    end

    iter = 0;

    while iter < itermax
        popold = pop;

        ind = randperm(4);

        a1 = randperm(NP);
        rt = rem(rot + ind(1), NP);
        a2 = a1(rt + 1);
        rt = rem(rot + ind(2), NP);
        a3 = a2(rt + 1);
        rt = rem(rot + ind(3), NP);
        a4 = a3(rt + 1);
        rt = rem(rot + ind(4), NP);
        a5 = a4(rt + 1); %#ok<NASGU>

        pm1 = popold(a1,:);
        pm2 = popold(a2,:);
        pm3 = popold(a3,:);
        pm4 = popold(a4,:);
        pm5 = popold(a5,:); %#ok<NASGU>

        bm = repmat(subbestmem, NP, 1);

        % Weighted CRm adaptation, retained from supplied DECC_MDG code.
        if rem(iter,24) == 0
            if iter ~= 0 && ~isempty(cc_rec) && sum(f_rec) > 0
                ccm = sum(f_rec .* cc_rec) / sum(f_rec);
            end
            cc_rec = [];
            f_rec = [];
        end

        % Generate CR values every 5 generations.
        if rem(iter,5) == 0
            cc = local_truncated_normal(ccm, 0.1, NP);
        end

        % Scale factor F: Gaussian or Cauchy neighborhood search.
        fst1 = (rand(NP,1) <= fp);
        fst2 = ~fst1;

        tmp = 0.5 + 0.3 .* randn(NP,1);
        F(fst1) = tmp(fst1);

        % Ratio of standard normals is a standard Cauchy random variable.
        tmpC = randn(NP,1) ./ randn(NP,1);
        F(fst2) = tmpC(fst2);

        F = abs(F);

        % Binomial crossover mask; ensure at least one mutant dimension.
        aa = rand(NP,D) < repmat(cc,1,D);
        zeroRows = find(sum(aa,2) == 0);
        for k = 1:numel(zeroRows)
            bb = randi(D);
            aa(zeroRows(k), bb) = 1;
        end

        mui = aa;
        mpo = mui < 0.5;

        aaa = (rand(NP,1) <= linkp);
        aindex = find(~aaa);
        bindex = find(aaa);

        % Strategy 1: current-to-best/2
        if ~isempty(bindex)
            ui(bindex,:) = popold(bindex,:) ...
                + repmat(F(bindex),1,D) .* (bm(bindex,:) - popold(bindex,:)) ...
                + repmat(F(bindex),1,D) .* ( ...
                    pm1(bindex,:) - pm2(bindex,:) ...
                    + pm3(bindex,:) - pm4(bindex,:));

            ui(bindex,:) = popold(bindex,:) .* mpo(bindex,:) ...
                + ui(bindex,:) .* mui(bindex,:);
        end

        % Strategy 2: rand/1
        if ~isempty(aindex)
            ui(aindex,:) = pm3(aindex,:) ...
                + repmat(F(aindex),1,D) .* (pm1(aindex,:) - pm2(aindex,:));

            ui(aindex,:) = popold(aindex,:) .* mpo(aindex,:) ...
                + ui(aindex,:) .* mui(aindex,:);
        end

        bbb = 1 - aaa;

        % Boundary handling: modulo reflection, as in supplied DECC_MDG.
        width = Ubound - Lbound;

        idxUpper = ui > Ubound;
        if any(idxUpper(:))
            ui(idxUpper) = Ubound(idxUpper) ...
                - mod(ui(idxUpper) - Ubound(idxUpper), width(idxUpper));
        end

        idxLower = ui < Lbound;
        if any(idxLower(:))
            ui(idxLower) = Lbound(idxLower) ...
                + mod(Lbound(idxLower) - ui(idxLower), width(idxLower));
        end

        % Evaluate trial component population in the current context vector.
        trialGpop = repmat(bestmem, popsize, 1);
        trialGpop(:, dim_index) = ui;
        tempval = local_eval_population(fname, trialGpop, func_num);

        for i = 1:NP
            if tempval(i) <= val(i)
                if tempval(i) < val(i)
                    cc_rec = [cc_rec, cc(i,1)]; %#ok<AGROW>
                    f_rec = [f_rec, (val(i) - tempval(i))]; %#ok<AGROW>
                end

                pop(i,:) = ui(i,:);
                val(i) = tempval(i);

                l1 = l1 + aaa(i);
                l2 = l2 + bbb(i);

                ns1 = ns1 + fst1(i);
                ns2 = ns2 + fst2(i);
            else
                nl1 = nl1 + aaa(i);
                nl2 = nl2 + bbb(i);

                nf1 = nf1 + fst1(i);
                nf2 = nf2 + fst2(i);
            end
        end

        if rem(iter,24) == 0 && iter ~= 0
            r1 = l1 / (l1 + nl1);
            r2 = l2 / (l2 + nl2);
            if r1 + r2 > 0
                linkp = r1 / (r1 + r2);
            end

            l1 = 1; l2 = 1; nl1 = 1; nl2 = 1;

            denom = ns2 * (ns1 + nf1) + ns1 * (ns2 + nf2);
            if denom > 0
                fp = (ns1 * (ns2 + nf2)) / denom;
            end
            ns1 = 1; nf1 = 1; ns2 = 1; nf2 = 1;
        end

        [best, ibest] = min(val);
        subbestmem = pop(ibest, :);

        if best < bestval
            bestval = best;
            bestmem(dim_index) = pop(ibest, :);
        end

        tracerst = [tracerst; bestval]; %#ok<AGROW>
        iter = iter + 1;
    end

    popnew = pop;
    bestmemnew = bestmem;
    bestvalnew = bestval;
end


% =========================================================================
% Objective function adapters
% =========================================================================
function y = local_eval_one(fname, x, func_num)
    % Platform convention in the user's CSO: fitness(x', FuncId), i.e. the
    % decision vector is passed as a column vector.
    y = fname(x(:), func_num);
    if numel(y) ~= 1
        y = y(1);
    end
    y = double(y);
end


function vals = local_eval_population(fname, pop, func_num)
    NP = size(pop,1);
    vals = zeros(1,NP);
    for ii = 1:NP
        vals(ii) = local_eval_one(fname, pop(ii,:), func_num);
    end
end


% =========================================================================
% Utilities
% =========================================================================
function b = local_expand_bound(b, dim)
    if isscalar(b)
        b = repmat(double(b), 1, dim);
    else
        b = double(b(:)');
        if numel(b) ~= dim
            error('Bound size does not match dimension.');
        end
    end
end


function x = local_truncated_normal(mu, sigma, n)
    % Generate N(mu,sigma) samples restricted to (0,1), matching the
    % acceptance logic used by the supplied DECC_MDG implementation.
    x = zeros(n,1);
    filled = 0;
    while filled < n
        candidates = mu + sigma .* randn(max(3*n, n-filled),1);
        candidates = candidates(candidates > 0 & candidates < 1);
        take = min(numel(candidates), n-filled);
        if take > 0
            x(filled+1:filled+take) = candidates(1:take);
            filled = filled + take;
        end
    end
end
