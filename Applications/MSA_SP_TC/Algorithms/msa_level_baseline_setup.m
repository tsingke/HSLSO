function [N,MaxFEs] = msa_level_baseline_setup(historyPoints,opts)
% Shared MSA population and FE-budget settings for LLSO/DLLSO/RLLPSO.
N=50;
if nargin>=2 && isstruct(opts) && isfield(opts,'levelBaselinePopulation')
    N=opts.levelBaselinePopulation;
end
validateattributes(N,{'numeric'},{'scalar','integer','>=',4});
MaxFEs=N*(historyPoints+1);
if nargin>=2 && isstruct(opts)
    if isfield(opts,'MaxFEs') && ~isempty(opts.MaxFEs)
        MaxFEs=opts.MaxFEs;
    elseif isfield(opts,'levelBaselineMaxFEs') && ~isempty(opts.levelBaselineMaxFEs)
        MaxFEs=opts.levelBaselineMaxFEs;
    end
end
validateattributes(MaxFEs,{'numeric'},{'scalar','integer','>=',N});
end
