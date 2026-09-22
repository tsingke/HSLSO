function [gbestx,bestever,gbesthistory] = MOS(mainHandle,popsize,dimension,xmax,xmin,vmax,vmin,maxiter,fCalculation,FuncId,VisualSwitch)
% MOS  SOURCEPORT_V3_20260827 - source-grounded MATLAB port of MOS-CEC2013.
%
% Platform interface (kept identical to the user's other algorithms):
%   [gbestx,bestever,gbesthistory] = MOS(mainHandle,popsize,dimension,...
%       xmax,xmin,vmax,vmin,maxiter,fCalculation,FuncId,VisualSwitch)
%
% Objective call convention:
%   f = fCalculation(x', FuncId)
%
% -------------------------------------------------------------------------
% REPRODUCTION BASIS
% -------------------------------------------------------------------------
% Paper:
%   A. LaTorre, S. Muelas, J.-M. Pena,
%   "Large Scale Global Optimization: Experimental Results with
%   MOS-based Hybrid Algorithms", IEEE CEC 2013.
%
% Source tree supplied by the user:
%   Algorithms-ELSGO/MOS_CEC2013/
%   - cec2013.cfg
%   - techs_cec2013.cfg
%   - src/gaeda/MOSEA2.cc
%   - src/gaeda/MOSTechniqueGA.cc
%   - src/gaeda/MOSTechniqueLSObj.cc
%   - src/gaeda/MOSTechniqueLS.cc
%   - src/gaeda/MOSParticipationFunction.cc
%   - src/gaeda/MOSQualityFunction.cc
%   - src/gaeda/GARealOps.cc
%   - src/gaeda/PopElitism.cc
%   - src/gaeda/LSElitism.cc
%   - src/gaeda/solis.cc / solisrand.cc
%   - src/gaeda/RLImprovPerDimManager.cc
%
% Important: the supplied repository is a later research copy adapted for
% EEG experiments. Its README.long_double.txt explicitly states that the
% CEC2013 loader in that copy is not runnable after an automated long-double
% conversion, and cec2013.cfg currently says evaluations=100000 for EEG.
% Therefore benchmark-level settings are taken from the CEC2013 paper
% (3e6 FEs), while low-level operator mechanics are ported from the source.
%
% One inconsistency exists in the supplied generalized MTS source:
% cec2013.cfg comments record the paper-selected raw factors
%   adjustFailed=2, adjustMin=2.5,
% while GARealOps.cc in this later copy multiplies SR by those options.
% That would enlarge SR after failure and contradicts both the original
% MTS-LS1 algorithm (SR is reduced after failure) and the meaning of the
% paper's factors. Here the paper factors are applied with their original
% MTS semantics:
%   SR <- SR / 2;  if SR < 1e-14, SR <- searchRange / 2.5.
% No numerical value is invented for this reconciliation.
%
% The stochastic laws/operators are ported, but MATLAB and the C++ library do
% not share the same global RNG implementation. Thus identical random-number
% streams are not expected. Solis-Wets uses its own source-defined SRandom
% generator and seed 12345679 below.
%
% -------------------------------------------------------------------------
% Fixed benchmark / MOS settings
% -------------------------------------------------------------------------
ComputeFitness = fCalculation;
MaxFEs = 3e6;               % CEC2013 LSGO paper budget
printEvery = 1000;          % user-requested output interval

D = dimension;
NP = 400;                   % CEC2013 selected GA population size

% Compatibility-only inputs, intentionally unused by MOS-CEC2013:
% mainHandle, popsize, vmax, vmin, maxiter, VisualSwitch

% Bounds -> 1-by-D vectors.
lb = xmin;
ub = xmax;
if isscalar(lb), lb = repmat(lb,1,D); else, lb = reshape(lb,1,[]); end
if isscalar(ub), ub = repmat(ub,1,D); else, ub = reshape(ub,1,[]); end
if numel(lb) ~= D || numel(ub) ~= D
    error('MOS:BoundSizeMismatch','xmin/xmax must be scalar or contain dimension elements.');
end
rangeVec = ub - lb;
if any(rangeVec <= 0)
    error('MOS:InvalidBounds','Every decision variable must have xmax > xmin.');
end

% MOS controller (paper + source).
stepFactor = 36000;
minPart = 0.20;
partAdjust = 0.05;          % GAEDAConfig::getParticipationFunction()
nTech = 3;
GA_ID  = 1;
SW_ID  = 2;
MTS_ID = 3;
participation = ones(1,nTech) / nTech;
prevQuality = zeros(1,nTech);
prevTimesImproved = zeros(1,nTech);
mosGeneration = 0;

% GA technique: techs_cec2013.cfg + RealBlend/RealGaussian source.
gaPcx = 0.90;
gaPmut = 0.01;
gaAlpha = 0.50;             % hard-coded in RealBlendCrossover as +/- 0.5*distance

% Solis-Wets: CEC2013 selected configuration (paper and cfg comments).
swMaxSuccess = 5;
swMaxFailed = 5;
swAdjustSuccess = 4.0;
swAdjustFailed = 0.75;
swDelta0 = 2.4;

% MTS-LS1-Reduced: CEC2013 selected configuration.
mtsAdjustFailedDivisor = 2.0;
mtsAdjustMinDivisor = 2.5;
mtsMoveLeft = 0.25;
mtsMoveRight = 0.50;
mtsSearchProb = 0.90;
mtsMinProb = 0.025;
mtsMinSR = 1e-14;
mtsRLAlpha = 0.70;          % RLImprovPerDimManager default in supplied source

% GAEDALib MTS uses one scalar SR based on allele set 0. CEC LSGO uses a
% common range for all coordinates. Keep source behavior by using dim 1.
mtsInitialSR = rangeVec(1) / 2.0;

% -------------------------------------------------------------------------
% Global FE / best state
% -------------------------------------------------------------------------
FEs = 0;
bestever = inf;
gbestx = zeros(1,D);
gbesthistory = inf(MaxFEs,1);

% -------------------------------------------------------------------------
% Initial population + per-real-genome MTS state
% GA1DArrayAlleleGenome initializes SR=range/2 and improve=true.
% -------------------------------------------------------------------------
P = zeros(NP,D);
fit = inf(NP,1);
pSR = repmat(mtsInitialSR,NP,1);
pImprove = true(NP,1);

for ii = 1:NP
    P(ii,:) = lb + rangeVec .* rand(1,D);
    fit(ii) = evaluatePoint(P(ii,:));
end

% -------------------------------------------------------------------------
% Persistent state of the two local-search techniques.
% The C++ source keeps one _currentGenome per local search across MOS steps.
% -------------------------------------------------------------------------
swInitialized = false;
swCurrentX = zeros(1,D);
swCurrentF = inf;
swCurrentSR = mtsInitialSR;     % carried only because MOSGenome carries real-genome state
swCurrentImprove = true;
swBias = zeros(1,D);
swDelta = swDelta0;
swNumSuccess = 0;
swNumFailed = 0;

mtsInitialized = false;
mtsCurrentX = zeros(1,D);
mtsCurrentF = inf;
mtsCurrentSR = mtsInitialSR;
mtsCurrentImprove = true;

% MTS reduced dimension-performance manager.
mtsDimValues = ones(1,D);
mtsDimExplored = false(1,D);
mtsAllDimsExplored = false;

% Solis-Wets dedicated SRandom state (solisrand.cc).
swRandM = 714025;
swRandIA = 1366;
swRandIC = 150889;
swRandIdum = 12345679;
swRandIy = 0;
swRandIr = zeros(1,97);
swRandInitialized = false;

% -------------------------------------------------------------------------
% MOS HRH main loop.
% Source order from techs_cec2013.cfg / use-tech:
%   GA_t1 -> LS_t1 (Solis-Wets) -> MTS_t1 (MTS-LS1-Reduced)
% Participation is updated BEFORE the techniques of the next MOS step,
% using qualities measured in the previous step.
% -------------------------------------------------------------------------
while FEs < MaxFEs

    participation = updateParticipationSource( ...
        participation,prevQuality,prevTimesImproved,mosGeneration);

    evalsToShare = min(stepFactor, MaxFEs - FEs);
    allocation = floor(participation .* evalsToShare ./ sum(participation));

    % MOSEA2.cc distributes integer leftovers by repeatedly incrementing
    % the currently smallest allocation (NOT by largest fractional remainder).
    totalAllocated = sum(allocation);
    while totalAllocated < evalsToShare
        [~,idxMin] = min(allocation);
        allocation(idxMin) = allocation(idxMin) + 1;
        totalAllocated = totalAllocated + 1;
    end

    thisQuality = zeros(1,nTech);
    thisImproved = zeros(1,nTech);

    if allocation(GA_ID) > 0 && FEs < MaxFEs
        [thisQuality(GA_ID),thisImproved(GA_ID)] = runGATechnique(allocation(GA_ID));
    end

    if allocation(SW_ID) > 0 && FEs < MaxFEs
        [thisQuality(SW_ID),thisImproved(SW_ID)] = runSolisWetsTechnique(allocation(SW_ID));
    end

    if allocation(MTS_ID) > 0 && FEs < MaxFEs
        [thisQuality(MTS_ID),thisImproved(MTS_ID)] = runMTSTechnique(allocation(MTS_ID));
    end

    prevQuality = thisQuality;
    prevTimesImproved = thisImproved;
    mosGeneration = mosGeneration + 1;
end

% Defensive exact-length guarantee.
if FEs < MaxFEs
    gbesthistory(FEs+1:MaxFEs) = bestever;
elseif numel(gbesthistory) > MaxFEs
    gbesthistory(MaxFEs+1:end) = [];
end

% =========================================================================
% Objective evaluation helper
% =========================================================================
    function f = evaluatePoint(x)
        if FEs >= MaxFEs
            error('MOS:FEBudgetExceeded','Attempted an objective evaluation after MaxFEs.');
        end

        f = ComputeFitness(x',FuncId);
        if ~isscalar(f)
            error('MOS:NonScalarFitness','fCalculation(x'',FuncId) must return a scalar.');
        end
        f = double(f);
        FEs = FEs + 1;

        if f < bestever
            bestever = f;
            gbestx = x;
        end
        gbesthistory(FEs) = bestever;

        if mod(FEs,printEvery) == 0
            fprintf('MOS算法，第%d次评价，最佳适应度 = %e\n',FEs,bestever);
        end
    end

% =========================================================================
% Source fitness increment: positive relative improvement only
% GAGenome::computeFitnessIncrement(old_score)
% =========================================================================
    function inc = fitnessIncrement(newScore,oldScore)
        if newScore < oldScore
            inc = abs((newScore-oldScore)/oldScore);
        else
            inc = 0.0;
        end
    end

% =========================================================================
% Dynamic participation function (MOSParticipationFunction.cc)
% Includes the source's post-generation-10 improvement override.
% =========================================================================
    function part = updateParticipationSource(part,quality,timesImproved,generation)

        % Reproduce MOSTechniqueSet::getBestTechniqueIds().
        bestQuality = 0.0;
        bestQIds = [];
        for t = 1:nTech
            if quality(t) == bestQuality
                bestQIds(end+1) = t; %#ok<AGROW>
            elseif quality(t) > bestQuality
                bestQuality = quality(t);
                bestQIds = t;
            end
        end

        ratioImproved = timesImproved ./ (1500.0 .* part);
        bestImprovement = 0.0;
        bestIIds = [];
        for t = 1:nTech
            if ratioImproved(t) == bestImprovement
                bestIIds(end+1) = t; %#ok<AGROW>
            elseif ratioImproved(t) > bestImprovement
                bestImprovement = ratioImproved(t);
                bestIIds = t;
            end
        end

        useImprovement = false;
        if numel(bestQIds) ~= numel(bestIIds) && generation > 10
            useImprovement = true;
            bestIds = bestIIds;
        else
            equalSets = numel(bestQIds) == numel(bestIIds) && all(bestQIds == bestIIds);
            if equalSets || generation <= 10
                bestIds = bestQIds;
            else
                useImprovement = true;
                bestIds = bestIIds;
            end
        end

        % No adjustment if all techniques tie (or no best technique).
        if isempty(bestIds) || numel(bestIds) == nTech
            return;
        end

        if useImprovement
            qUsed = ratioImproved;
        else
            qUsed = quality;
        end

        bestQ = qUsed(bestIds(1));
        baseQ = 0.0; % fitIncAvg => DynamicParticipation::setBaseQual returns 0

        for t = 1:nTech
            if any(bestIds == t)
                continue;
            end

            diffQ = (bestQ - qUsed(t)) / (bestQ - baseQ);
            currentPart = part(t);
            ratioInc = currentPart * partAdjust * diffQ;

            if (currentPart - ratioInc) > minPart
                shared = ratioInc / numel(bestIds);
                part(bestIds) = part(bestIds) + shared;
                part(t) = part(t) - ratioInc;
            elseif currentPart > minPart
                shared = (currentPart - minPart) / numel(bestIds);
                part(bestIds) = part(bestIds) + shared;
                part(t) = minPart;
            end
        end

        % Source only clips floating-point drift; it does not renormalize.
        part(part < 0) = 0;
        part(part > 1) = 1;
    end

% =========================================================================
% GA technique (MOSTechniqueGA + RealBlendCrossover + RealGaussianMutator)
% =========================================================================
    function [quality,timesImproved] = runGATechnique(maxNominalEvals)

        % MOSTechnique::evolve() clones the current population once and then
        % alternates the two population buffers after each internal GA iteration.
        curP = P;
        curFit = fit;
        curSR = pSR;
        curImprove = pImprove;

        auxP = curP;
        auxFit = curFit;
        auxSR = curSR;
        auxImprove = curImprove;

        totalNominal = 0;
        qualAcum = 0.0;
        nIters = 0;
        timesImproved = 0; % MOSTechniqueGA never increments _times_improved

        while totalNominal < maxNominalEvals
            remaining = maxNominalEvals - totalNominal;
            chunk = min(NP,remaining);

            incSum = 0.0;
            markerCount = 0;
            pos = 1;

            while pos <= chunk
                if pos < chunk
                    nChildren = 2;
                else
                    nChildren = 1;
                end

                momIdx = tournament2(curFit);
                dadIdx = tournament2(curFit);
                bestParentF = min(curFit(momIdx),curFit(dadIdx));

                if nChildren == 2
                    child1 = auxP(pos,:);
                    child2 = auxP(pos+1,:);
                    child1SR = auxSR(pos); child1Imp = auxImprove(pos);
                    child2SR = auxSR(pos+1); child2Imp = auxImprove(pos+1);

                    crossed = rand < gaPcx;
                    if crossed
                        [child1,child2] = blendTwo(curP(momIdx,:),curP(dadIdx,:));
                        changed1 = true;
                        changed2 = true;
                    else
                        child1 = curP(momIdx,:);
                        child2 = curP(dadIdx,:);
                        child1SR = curSR(momIdx); child1Imp = curImprove(momIdx);
                        child2SR = curSR(dadIdx); child2Imp = curImprove(dadIdx);
                        changed1 = false;
                        changed2 = false;
                    end

                    [child1,nmut1] = gaussianMutate(child1);
                    [child2,nmut2] = gaussianMutate(child2);
                    changed1 = changed1 || (nmut1 > 0);
                    changed2 = changed2 || (nmut2 > 0);

                    if changed1
                        f1 = evaluatePoint(child1);
                        inc1 = fitnessIncrement(f1,bestParentF);
                        incSum = incSum + inc1;
                        markerCount = markerCount + 1;
                    else
                        f1 = curFit(momIdx);
                    end

                    if changed2
                        f2 = evaluatePoint(child2);
                        inc2 = fitnessIncrement(f2,bestParentF);
                        incSum = incSum + inc2;
                        markerCount = markerCount + 1;
                    else
                        f2 = curFit(dadIdx);
                    end

                    auxP(pos,:) = child1;       auxFit(pos) = f1;
                    auxSR(pos) = child1SR;      auxImprove(pos) = child1Imp;
                    auxP(pos+1,:) = child2;     auxFit(pos+1) = f2;
                    auxSR(pos+1) = child2SR;    auxImprove(pos+1) = child2Imp;
                    pos = pos + 2;
                else
                    child = auxP(pos,:);
                    childSR = auxSR(pos); childImp = auxImprove(pos);

                    crossed = rand < gaPcx;
                    if crossed
                        child = blendOne(curP(momIdx,:),curP(dadIdx,:));
                        changed = true;
                    else
                        if rand < 0.5
                            child = curP(momIdx,:);
                            childSR = curSR(momIdx); childImp = curImprove(momIdx);
                            inheritedF = curFit(momIdx);
                        else
                            child = curP(dadIdx,:);
                            childSR = curSR(dadIdx); childImp = curImprove(dadIdx);
                            inheritedF = curFit(dadIdx);
                        end
                        changed = false;
                    end

                    [child,nmut] = gaussianMutate(child);
                    changed = changed || (nmut > 0);

                    if changed
                        fc = evaluatePoint(child);
                        inc = fitnessIncrement(fc,bestParentF);
                        incSum = incSum + inc;
                        markerCount = markerCount + 1;
                    else
                        fc = inheritedF;
                    end

                    auxP(pos,:) = child; auxFit(pos) = fc;
                    auxSR(pos) = childSR; auxImprove(pos) = childImp;
                    pos = pos + 1;
                end
            end

            % AverageFitnessIncrementQuality divides by nominal newInds
            % (usedEvals), even if a generated clone did not require reevaluation.
            if markerCount > 0
                qAct = incSum / chunk;
            else
                qAct = 0.0;
            end
            qualAcum = qualAcum + qAct;
            nIters = nIters + 1;
            totalNominal = totalNominal + chunk;

            % PopElitism: append ALL old population to aux, sort 2N, remove worst N.
            allP = [auxP; curP];
            allFit = [auxFit; curFit];
            allSR = [auxSR; curSR];
            allImprove = [auxImprove; curImprove];
            [~,ord] = sort(allFit,'ascend');
            ord = ord(1:NP);
            newP = allP(ord,:);
            newFit = allFit(ord);
            newSR = allSR(ord);
            newImprove = allImprove(ord);

            % Source swaps pop and aux buffers. The old cur buffer becomes the
            % destination buffer for the next internal GA iteration.
            oldCurP = curP; oldCurFit = curFit; oldCurSR = curSR; oldCurImprove = curImprove;
            curP = newP; curFit = newFit; curSR = newSR; curImprove = newImprove;
            auxP = oldCurP; auxFit = oldCurFit; auxSR = oldCurSR; auxImprove = oldCurImprove;
        end

        P = curP; fit = curFit; pSR = curSR; pImprove = curImprove;
        quality = qualAcum / nIters;
    end

    function idx = tournament2(fvec)
        a = randi(NP);
        b = randi(NP);
        if fvec(a) == fvec(b)
            if rand < 0.5, idx = a; else, idx = b; end
        elseif fvec(a) < fvec(b)
            idx = a;
        else
            idx = b;
        end
    end

    function [c1,c2] = blendTwo(mom,dad)
        dist = abs(mom-dad);
        lo = min(mom,dad) - gaAlpha.*dist;
        hi = max(mom,dad) + gaAlpha.*dist;
        lo = max(lo,lb);
        hi = min(hi,ub);
        c1 = lo + (hi-lo).*rand(1,D);
        c2 = lo + (hi-lo).*rand(1,D);
    end

    function c = blendOne(mom,dad)
        dist = abs(mom-dad);
        lo = max(min(mom,dad) - gaAlpha.*dist,lb);
        hi = min(max(mom,dad) + gaAlpha.*dist,ub);
        c = lo + (hi-lo).*rand(1,D);
    end

    function [x,nMut] = gaussianMutate(x)
        nMutReal = gaPmut * D;
        nMut = floor(nMutReal);
        frac = nMutReal - nMut;
        if rand < frac
            nMut = nMut + 1;
        end

        for mm = 1:nMut
            idx = randi(D); % source chooses indices with replacement
            oldGene = x(idx);
            alleleRange = rangeVec(idx);
            while true
                % RealGaussianMutator: gene + N(0,1) * full allele range,
                % rejection-sampled until it lies inside the bounds.
                newGene = oldGene + randn * alleleRange;
                if newGene >= lb(idx) && newGene <= ub(idx)
                    break;
                end
            end
            x(idx) = newGene;
        end
    end

% =========================================================================
% Solis-Wets technique (MOSTechniqueLSObj + solis.cc)
% =========================================================================
    function [quality,timesImproved] = runSolisWetsTechnique(maxEvals)

        [bestPopF,bestIdx] = min(fit);

        if ~swInitialized
            swCurrentX = P(bestIdx,:);
            swCurrentF = bestPopF;
            swCurrentSR = pSR(bestIdx);
            swCurrentImprove = pImprove(bestIdx);
            swInitialized = true;
        end

        % Source starts from the better of population best and persistent
        % local-search _currentGenome. Bias/delta state itself is NOT reset.
        if bestPopF < swCurrentF
            x = P(bestIdx,:); f = bestPopF;
            xSR = pSR(bestIdx); xImprove = pImprove(bestIdx);
        else
            x = swCurrentX; f = swCurrentF;
            xSR = swCurrentSR; xImprove = swCurrentImprove;
        end

        used = 0;
        incTotal = 0.0;
        timesImproved = 0;

        while used < maxEvals
            oldF = f;
            dif = zeros(1,D);
            for dd = 1:D
                dif(dd) = swNormal(swDelta);
            end

            trial = x + swBias + dif;
            trial = min(max(trial,lb),ub);
            fTrial = evaluatePoint(trial);
            used = used + 1;
            incTotal = incTotal + fitnessIncrement(fTrial,oldF);

            if fTrial < oldF
                x = trial; f = fTrial;
                timesImproved = timesImproved + 1;
                swBias = 0.6.*swBias + 0.4.*dif;
                swNumSuccess = swNumSuccess + 1;
                swNumFailed = 0;
            elseif used < maxEvals
                trial2 = x - swBias - dif;
                trial2 = min(max(trial2,lb),ub);
                fTrial2 = evaluatePoint(trial2);
                used = used + 1;
                incTotal = incTotal + fitnessIncrement(fTrial2,oldF);

                if fTrial2 < oldF
                    x = trial2; f = fTrial2;
                    timesImproved = timesImproved + 1;
                    swBias = 0.6.*swBias - 0.4.*dif;
                    swNumSuccess = swNumSuccess + 1;
                    swNumFailed = 0;
                else
                    swBias = 0.5.*swBias;
                    swNumFailed = swNumFailed + 1;
                    swNumSuccess = 0;
                end
            end

            if swNumSuccess >= swMaxSuccess
                swNumSuccess = 0;
                swDelta = swDelta * swAdjustSuccess;
            elseif swNumFailed >= swMaxFailed
                swNumFailed = 0;
                swDelta = swDelta * swAdjustFailed;
            end
        end

        quality = incTotal / used;

        swCurrentX = x; swCurrentF = f;
        swCurrentSR = xSR; swCurrentImprove = xImprove;

        % LSElitism: keep the LS-modified first individual, copy all other
        % positions from the old population, then sort.
        newP = P; newFit = fit; newSR = pSR; newImprove = pImprove;
        newP(bestIdx,:) = x; newFit(bestIdx) = f;
        newSR(bestIdx) = xSR; newImprove(bestIdx) = xImprove;
        [newFit,ord] = sort(newFit,'ascend');
        P = newP(ord,:); pSR = newSR(ord); pImprove = newImprove(ord);
        fit = newFit;
    end

% =========================================================================
% MTS-LS1-Reduced technique (MOSTechniqueLS + GARealOps.cc)
% =========================================================================
    function [quality,timesImproved] = runMTSTechnique(maxEvals)

        [bestPopF,bestIdx] = min(fit);

        if ~mtsInitialized
            mtsCurrentX = P(bestIdx,:);
            mtsCurrentF = bestPopF;
            mtsCurrentSR = pSR(bestIdx);
            mtsCurrentImprove = pImprove(bestIdx);
            mtsInitialized = true;
        end

        % Better of population best and persistent MTS _currentGenome,
        % including the real-genome SR/improve state copied with the genome.
        if bestPopF < mtsCurrentF
            x = P(bestIdx,:); f = bestPopF;
            SR = pSR(bestIdx); improveFlag = pImprove(bestIdx);
        else
            x = mtsCurrentX; f = mtsCurrentF;
            SR = mtsCurrentSR; improveFlag = mtsCurrentImprove;
        end

        fitBest = f;
        used = 0;
        incTotal = 0.0;
        timesImproved = 0;

        % MOSTechniqueLS::offspring repeatedly calls the LS until its full
        % technique FE allocation is consumed.
        while used < maxEvals
            [x,f,fitBest,SR,improveFlag,usedNow,incNow,impNow] = ...
                mtsReducedCall(x,f,fitBest,SR,improveFlag,maxEvals-used);

            used = used + usedNow;
            incTotal = incTotal + incNow;
            timesImproved = timesImproved + impNow;

            if usedNow == 0
                error('MOS:MTSNoEvaluation','MTS-LS1-Reduced consumed zero evaluations.');
            end
        end

        quality = incTotal / used;

        mtsCurrentX = x; mtsCurrentF = f;
        mtsCurrentSR = SR; mtsCurrentImprove = improveFlag;

        % LSElitism: replace only the population-best slot with the LS result,
        % preserve the other individuals, then sort.
        newP = P; newFit = fit; newSR = pSR; newImprove = pImprove;
        newP(bestIdx,:) = x; newFit(bestIdx) = f;
        newSR(bestIdx) = SR; newImprove(bestIdx) = improveFlag;
        [newFit,ord] = sort(newFit,'ascend');
        P = newP(ord,:); pSR = newSR(ord); pImprove = newImprove(ord);
        fit = newFit;
    end

    function [x,f,fitBest,SR,improveFlag,evals,fitIncAcum,improvements] = ...
            mtsReducedCall(x,f,fitBest,SR,improveFlag,maxIters)

        evals = 0;
        fitIncAcum = 0.0;
        improvements = 0;

        % Paper-selected adjustFailed=2 and adjustMin=2.5 interpreted with
        % original MTS semantics (shrink after failure; reset after minimum).
        if ~improveFlag
            SR = SR / mtsAdjustFailedDivisor;
            if SR < mtsMinSR
                SR = rangeVec(1) / mtsAdjustMinDivisor;
            end
        end
        improveFlag = false;

        dims = mtsSelectDimensions();
        dims = sort(dims,'ascend'); % source explicitly sorts numeric indices

        for kk = 1:numel(dims)
            dim = dims(kk);
            if evals == maxIters
                return;
            end

            oldGene = x(dim);
            oldScore = f;
            newScore = oldScore;
            trialWasEvaluated = false;

            isAtLower = (oldGene == lb(dim));

            if ~isAtLower
                newGene = oldGene - mtsMoveLeft*SR;
                if newGene < lb(dim), newGene = lb(dim); end
                trial = x;
                trial(dim) = newGene;
                newScore = evaluatePoint(trial);
                evals = evals + 1;
                trialWasEvaluated = true;
                fitIncAcum = fitIncAcum + fitnessIncrement(newScore,oldScore);
                if newScore < oldScore
                    improvements = improvements + 1;
                end
                if newScore < fitBest
                    fitBest = newScore;
                end

                if newScore < oldScore
                    x = trial; f = newScore;
                elseif newScore == oldScore
                    % Equal first move is restored and no right move is tried.
                    x(dim) = oldGene; f = oldScore;
                end
            end

            % Source tries the right direction if at lower bound OR first move
            % was worse. Equality does not enter this branch.
            if isAtLower || (trialWasEvaluated && newScore > oldScore)
                x(dim) = oldGene; f = oldScore;

                % Source returns here before per-dimension manager update if
                % the FE budget was exhausted by the failed left trial.
                if evals == maxIters
                    return;
                end

                if oldGene ~= ub(dim)
                    newGene = oldGene + mtsMoveRight*SR;
                    if newGene > ub(dim), newGene = ub(dim); end
                    trial = x;
                    trial(dim) = newGene;
                    newScore = evaluatePoint(trial);
                    evals = evals + 1;
                    fitIncAcum = fitIncAcum + fitnessIncrement(newScore,oldScore);
                    if newScore < oldScore
                        improvements = improvements + 1;
                    end
                    if newScore < fitBest
                        fitBest = newScore;
                    end

                    if newScore < oldScore
                        x = trial; f = newScore;
                        improveFlag = true;
                    else
                        x(dim) = oldGene; f = oldScore;
                    end
                end
            elseif trialWasEvaluated && newScore < oldScore
                improveFlag = true;
            end

            % CEC2013 source manager stores absolute score improvement for
            % the final evaluated move of this dimension, otherwise zero.
            if newScore < oldScore
                dimImprovement = abs(newScore-oldScore);
            else
                dimImprovement = 0.0;
            end
            mtsUpdateDimValue(dim,dimImprovement);
        end
    end

    function dims = mtsSelectDimensions()
        perf = mtsDimValues;

        if ~mtsAllDimsExplored
            maxV = max(mtsDimValues);
            perf(~mtsDimExplored) = maxV + 1.0;
        end

        s = sum(perf);
        if s > 0
            probs = perf ./ s;
        else
            % Exact fallback present in ImprovPerDimManager::getProbs().
            probs = (1:D) ./ D;
        end

        [sortedProb,sortedDims] = sort(probs,'ascend');
        selected = [];
        sumProb = 0.0;
        pos = D;
        for jj = D:-1:1
            selected(end+1) = sortedDims(jj); %#ok<AGROW>
            sumProb = sumProb + sortedProb(jj);
            pos = jj-1; % number of lower-probability dimensions remaining
            if sumProb > mtsSearchProb
                break;
            end
        end

        nOther = floor(D*mtsMinProb);
        if nOther == 0, nOther = 1; end

        if nOther > 0 && pos > 0
            nOther = min(nOther,pos);
            lowDims = sortedDims(1:pos);
            order = randperm(numel(lowDims));
            selected = [selected, lowDims(order(1:nOther))]; %#ok<AGROW>
        end

        dims = selected;
    end

    function mtsUpdateDimValue(dim,newValue)
        if ~mtsDimExplored(dim)
            mtsDimValues(dim) = newValue;
        else
            mtsDimValues(dim) = mtsRLAlpha*mtsDimValues(dim) + ...
                                (1-mtsRLAlpha)*newValue;
        end
        mtsDimExplored(dim) = true;

        if ~mtsAllDimsExplored
            mtsAllDimsExplored = all(mtsDimExplored);
        end
    end

% =========================================================================
% Solis-Wets dedicated RNG (solisrand.cc + solis.cc::normal)
% =========================================================================
    function u = swUniform()
        if ~swRandInitialized
            swRandIdum = rem(swRandIC - swRandIdum,swRandM);
            if swRandIdum < 0, swRandIdum = -swRandIdum; end
            for jj = 1:97
                swRandIdum = rem(swRandIA*swRandIdum + swRandIC,swRandM);
                swRandIr(jj) = swRandIdum;
            end
            swRandIdum = rem(swRandIA*swRandIdum + swRandIC,swRandM);
            swRandIy = swRandIdum;
            swRandInitialized = true;
        end

        jj = floor(1 + 97.0*swRandIy/swRandM);
        if jj < 1 || jj > 97
            error('MOS:SolisRNG','Solis-Wets SRandom index out of range.');
        end
        swRandIy = swRandIr(jj);
        swRandIdum = rem(swRandIA*swRandIdum + swRandIC,swRandM);
        swRandIr(jj) = swRandIdum;

        % C++ returns (float)iy/M; use a single cast before converting back.
        u = double(single(swRandIy/swRandM));
    end

    function z = swNormal(delta)
        u1 = 0.0;
        while u1 == 0.0
            u1 = swUniform();
        end
        u2 = swUniform();
        z = delta * sqrt(-2.0*log(u1)) * sin(2.0*pi*u2);
    end

end
