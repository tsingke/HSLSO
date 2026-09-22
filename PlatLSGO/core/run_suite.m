function run_suite(cfg)
root = setup_paths(cfg.suite);
if ~exist(cfg.resultDir,'dir'), mkdir(cfg.resultDir); end
suiteDir = fullfile(cfg.resultDir,upper(cfg.suite));
if ~exist(suiteDir,'dir'), mkdir(suiteDir); end
errorRows = {};

for a = 1:numel(cfg.algorithms)
    algorithm = cfg.algorithms{a};
    algOut = fullfile(suiteDir,algorithm);
    if ~exist(algOut,'dir'), mkdir(algOut); end

    for fid = cfg.funcIds
        funcOut = fullfile(algOut,sprintf('F%02d',fid));
        if ~exist(funcOut,'dir'), mkdir(funcOut); end

        vals = nan(cfg.numRuns,1);
        times = nan(cfg.numRuns,1);
        dims = nan(cfg.numRuns,1);
        status = strings(cfg.numRuns,1);

        for r = 1:cfg.numRuns
            try
                result = run_single(cfg,algorithm,fid,r);
                vals(r) = result.bestFitness;
                times(r) = result.runtime;
                dims(r) = result.dimension;
                status(r) = "OK";
                save(fullfile(funcOut,sprintf('run_%02d.mat',r)),'result');
                fprintf('%s %s F%02d run %02d/%02d  %.8e\n',upper(cfg.suite),algorithm,fid,r,cfg.numRuns,result.bestFitness);
            catch ME
                status(r) = "ERROR";
                errorRows(end+1,:) = {upper(cfg.suite),algorithm,fid,r,ME.message}; %#ok<AGROW>
                fprintf(2,'%s %s F%02d run %02d ERROR: %s\n',upper(cfg.suite),algorithm,fid,r,ME.message);
            end
        end

        T = table((1:cfg.numRuns)',dims,vals,times,status,'VariableNames',{'Run','Dimension','BestFitness','Runtime','Status'});
        writetable(T,fullfile(funcOut,'runs.csv'));

        good = isfinite(vals);
        if any(good)
            S = table(fid,dims(find(good,1)),sum(good),mean(vals(good)),std(vals(good),0),median(vals(good)),min(vals(good)),max(vals(good)),mean(times(good)), ...
                'VariableNames',{'Function','Dimension','Runs','Mean','Std','Median','Best','Worst','MeanRuntime'});
            writetable(S,fullfile(funcOut,'summary.csv'));
        end
    end
end

if ~isempty(errorRows)
    E = cell2table(errorRows,'VariableNames',{'Suite','Algorithm','Function','Run','Message'});
    writetable(E,fullfile(suiteDir,'errors.csv'));
end
build_suite_summary(cfg);
end
