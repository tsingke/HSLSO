function summarize_dataset(datasetDir,cfg)
%SUMMARIZE_DATASET Combine per-run descriptive summaries for one dataset.
% This function reports raw observations and descriptive statistics only.
% It does not perform hypothesis tests or pairwise algorithm rankings.

raw = table();

for ai = 1:numel(cfg.algorithms)
    alg = cfg.algorithms{ai};
    f = fullfile(datasetDir,alg,'RawResults.csv');
    if ~exist(f,'file')
        continue;
    end
    T = readtable(f);
    T.Algorithm = repmat({alg},height(T),1);
    raw = [raw; T]; %#ok<AGROW>
end

if isempty(raw)
    return;
end

writetable(raw,fullfile(datasetDir,'RawResults_AllAlgorithms.csv'));

Algorithm = {};
Runs = [];
PopulationSizeInitial = [];
PopulationSizeMinimum = [];
MaxFEs = [];
FEs_Mean = [];
FEs_Std = [];
InternalSPS_Mean = [];
InternalSPS_Std = [];
PreAlignmentSP = [];
InitialHMMSP = [];
SP_Mean = [];
SP_Std = [];
DeltaSP_HMM_Mean = [];
DeltaSP_HMM_Std = [];
PreAlignmentTC = [];
InitialHMMTC = [];
TC_Mean = [];
TC_Std = [];
DeltaTC_HMM_Mean = [];
DeltaTC_HMM_Std = [];
Time_Mean = [];

for ai = 1:numel(cfg.algorithms)
    alg = cfg.algorithms{ai};
    idx = strcmp(raw.Algorithm,alg);
    if ~any(idx)
        continue;
    end
    T = raw(idx,:);

    Algorithm{end+1,1} = alg; %#ok<AGROW>
    Runs(end+1,1) = height(T); %#ok<AGROW>
    PopulationSizeInitial(end+1,1) = T.PopulationSizeInitial(1); %#ok<AGROW>
    PopulationSizeMinimum(end+1,1) = T.PopulationSizeMinimum(1); %#ok<AGROW>
    MaxFEs(end+1,1) = T.MaxFEs(1); %#ok<AGROW>
    FEs_Mean(end+1,1) = mean(T.FitnessEvaluations); %#ok<AGROW>
    FEs_Std(end+1,1) = std(T.FitnessEvaluations,0); %#ok<AGROW>

    [InternalSPS_Mean(end+1,1),InternalSPS_Std(end+1,1)] = ...
        mean_std(T.DecodedSPS); %#ok<AGROW>
    [PreAlignmentSP(end+1,1),~] = mean_std(T.PreAlignmentSP); %#ok<AGROW>
    [InitialHMMSP(end+1,1),~] = mean_std(T.InitialHMMSP); %#ok<AGROW>
    [SP_Mean(end+1,1),SP_Std(end+1,1)] = mean_std(T.SP); %#ok<AGROW>
    [DeltaSP_HMM_Mean(end+1,1),DeltaSP_HMM_Std(end+1,1)] = ...
        mean_std(T.DeltaSP_HMM); %#ok<AGROW>
    [PreAlignmentTC(end+1,1),~] = mean_std(T.PreAlignmentTC); %#ok<AGROW>
    [InitialHMMTC(end+1,1),~] = mean_std(T.InitialHMMTC); %#ok<AGROW>
    [TC_Mean(end+1,1),TC_Std(end+1,1)] = mean_std(T.TC); %#ok<AGROW>
    [DeltaTC_HMM_Mean(end+1,1),DeltaTC_HMM_Std(end+1,1)] = ...
        mean_std(T.DeltaTC_HMM); %#ok<AGROW>
    [Time_Mean(end+1,1),~] = mean_std(T.TimeSeconds); %#ok<AGROW>
end

Summary = table(Algorithm,Runs,PopulationSizeInitial,PopulationSizeMinimum, ...
    MaxFEs,FEs_Mean,FEs_Std,InternalSPS_Mean,InternalSPS_Std, ...
    PreAlignmentSP,InitialHMMSP,SP_Mean,SP_Std, ...
    DeltaSP_HMM_Mean,DeltaSP_HMM_Std,PreAlignmentTC,InitialHMMTC, ...
    TC_Mean,TC_Std,DeltaTC_HMM_Mean,DeltaTC_HMM_Std,Time_Mean);
writetable(Summary,fullfile(datasetDir,'Summary_AllAlgorithms.csv'));

Paper = table(Algorithm);
Paper.InternalSPS = cell(size(Algorithm));
Paper.PreAlignmentSP = PreAlignmentSP;
Paper.InitialHMMSP = InitialHMMSP;
Paper.SP = cell(size(Algorithm));
Paper.DeltaSP_HMM = cell(size(Algorithm));
Paper.PreAlignmentTC = PreAlignmentTC;
Paper.InitialHMMTC = InitialHMMTC;
Paper.TC = cell(size(Algorithm));
Paper.DeltaTC_HMM = cell(size(Algorithm));

for ii = 1:numel(Algorithm)
    Paper.InternalSPS{ii} = sprintf('%.6g +/- %.3g', ...
        InternalSPS_Mean(ii),InternalSPS_Std(ii));
    Paper.SP{ii} = sprintf('%.4f +/- %.4f',SP_Mean(ii),SP_Std(ii));
    Paper.DeltaSP_HMM{ii} = sprintf('%+.4f +/- %.4f', ...
        DeltaSP_HMM_Mean(ii),DeltaSP_HMM_Std(ii));
    Paper.TC{ii} = sprintf('%.4f +/- %.4f',TC_Mean(ii),TC_Std(ii));
    Paper.DeltaTC_HMM{ii} = sprintf('%+.4f +/- %.4f', ...
        DeltaTC_HMM_Mean(ii),DeltaTC_HMM_Std(ii));
end

writetable(Paper,fullfile(datasetDir,'PaperTable_MeanStd.csv'));
end

function [m,s] = mean_std(x)
x = x(isfinite(x));
if isempty(x)
    m = NaN;
    s = NaN;
else
    m = mean(x);
    s = std(x,0);
end
end
