function summarize_algorithm_runs(datasetDir,alg,cfg)
%SUMMARIZE_ALGORITHM_RUNS Write per-run data, descriptive summaries and FE histories.
algDir=fullfile(datasetDir,alg);
rows=[];
for r=1:cfg.MaxRun
    f=fullfile(algDir,sprintf('run_%02d.mat',r));
    if ~exist(f,'file'), continue; end
    s=load(f,'result'); q=s.result;
    preSPS=getfield_default(q,'PreAlignmentSPS',NaN);
    preSP=getfield_default(q,'PreAlignmentSP',NaN);
    preTC=getfield_default(q,'PreAlignmentTC',NaN);
    hSPS=getfield_default(q,'InitialHMMSPS',getfield_default(q,'InitialSPS',NaN));
    hSP=getfield_default(q,'InitialHMMSP',getfield_default(q,'InitialSP',NaN));
    hTC=getfield_default(q,'InitialHMMTC',getfield_default(q,'InitialTC',NaN));
    nFEs=getfield_default(q,'FitnessEvaluations',NaN);
    pop0=getfield_default(q,'PopulationSizeInitial',NaN);
    popMin=getfield_default(q,'PopulationSizeMinimum',NaN);
    maxFEs=getfield_default(q,'MaxFEs',NaN);

    rows=[rows; q.Run q.Seed pop0 popMin maxFEs nFEs q.OptimizerFitness q.DecodedSPS ...
        preSPS preSP preTC hSPS hSP hTC ...
        q.SP q.SP-hSP q.SP-preSP q.TC q.TC-hTC q.TC-preTC q.TimeSeconds ...
        q.CorrectPairs q.ReferencePairs q.CorrectColumns q.ReferenceColumns]; %#ok<AGROW>
end
if isempty(rows), return; end

T=array2table(rows,'VariableNames',{'Run','Seed','PopulationSizeInitial','PopulationSizeMinimum', ...
    'MaxFEs','FitnessEvaluations','OptimizerFitness','DecodedSPS', ...
    'PreAlignmentSPS','PreAlignmentSP','PreAlignmentTC', ...
    'InitialHMMSPS','InitialHMMSP','InitialHMMTC', ...
    'SP','DeltaSP_HMM','DeltaSP_PreAlignment','TC','DeltaTC_HMM','DeltaTC_PreAlignment','TimeSeconds', ...
    'CorrectPairs','ReferencePairs','CorrectColumns','ReferenceColumns'});
writetable(T,fullfile(algDir,'RawResults.csv'));

metrics={'OptimizerFitness','DecodedSPS','SP','TC','DeltaSP_HMM','DeltaTC_HMM', ...
    'PopulationSizeInitial','PopulationSizeMinimum','MaxFEs','FitnessEvaluations','TimeSeconds'};
Metric=cell(numel(metrics),1); N=zeros(numel(metrics),1); Mean=N; Std=N; Median=N; Min=N; Max=N;
for i=1:numel(metrics)
    x=T.(metrics{i}); x=x(isfinite(x));
    Metric{i}=metrics{i}; N(i)=numel(x);
    if isempty(x)
        Mean(i)=NaN; Std(i)=NaN; Median(i)=NaN; Min(i)=NaN; Max(i)=NaN;
    else
        Mean(i)=mean(x); Std(i)=std(x,0); Median(i)=median(x); Min(i)=min(x); Max(i)=max(x);
    end
end
S=table(Metric,N,Mean,Std,Median,Min,Max);
writetable(S,fullfile(algDir,'Summary.csv'));

if cfg.saveConvergence
    H=nan(cfg.historyPoints,cfg.MaxRun);
    for r=1:cfg.MaxRun
        f=fullfile(algDir,sprintf('run_%02d.mat',r));
        if exist(f,'file')
            s=load(f,'result');
            H(:,r)=normalize_history(s.result.History,cfg.historyPoints);
        end
    end
    avg=mean_omitnan(H,2); med=median_omitnan(H,2);
    axisValues=round(linspace(1,cfg.MaxFEs,cfg.historyPoints))';
    C=table(axisValues,avg,med,'VariableNames', ...
        {'FitnessEvaluations','MeanBestFitness','MedianBestFitness'});
    writetable(C,fullfile(algDir,'Convergence.csv'));
end
end

function v=getfield_default(s,name,default)
if isfield(s,name), v=s.(name); else, v=default; end
end
function y=mean_omitnan(X,dim)
mask=isfinite(X); X(~mask)=0; n=sum(mask,dim); y=sum(X,dim)./n; y(n==0)=NaN;
end
function y=median_omitnan(X,dim)
if dim~=2, error('Internal helper supports dim=2 only.'); end
y=nan(size(X,1),1);
for i=1:size(X,1)
    v=X(i,isfinite(X(i,:)));
    if ~isempty(v), y(i)=median(v); end
end
end
