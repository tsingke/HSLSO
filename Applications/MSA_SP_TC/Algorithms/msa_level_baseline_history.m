function sampled = msa_level_baseline_history(feBest, nPoints)
% Sample an evaluation-based best-so-far history at evenly spaced FE counts.
% Do NOT interpret these samples as outer iterations.
if isempty(feBest), error('No fitness evaluations were recorded.'); end
idx = max(1,round(linspace(1,numel(feBest),nPoints)));
sampled = feBest(idx(:));
end
