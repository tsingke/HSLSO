function build_suite_summary(cfg)
suiteDir = fullfile(cfg.resultDir,upper(cfg.suite));
rows = {};
for a = 1:numel(cfg.algorithms)
    alg = cfg.algorithms{a};
    for fid = cfg.funcIds
        f = fullfile(suiteDir,alg,sprintf('F%02d',fid),'summary.csv');
        if exist(f,'file')
            T = readtable(f);
            rows(end+1,:) = {alg,fid,T.Dimension(1),T.Runs(1),T.Mean(1),T.Std(1),T.Median(1),T.Best(1),T.Worst(1),T.MeanRuntime(1)}; %#ok<AGROW>
        end
    end
end
if ~isempty(rows)
    S = cell2table(rows,'VariableNames',{'Algorithm','Function','Dimension','Runs','Mean','Std','Median','Best','Worst','MeanRuntime'});
    writetable(S,fullfile(suiteDir,'summary_all.csv'));
end
end
