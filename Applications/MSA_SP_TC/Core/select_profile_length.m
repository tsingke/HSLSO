function K = select_profile_length(lengths,cfg)
%SELECT_PROFILE_LENGTH Choose the number of profile match states.
%
% A profile HMM match-state count should represent a typical family length,
% not an artificial value longer than every sequence.  Without using a
% reference alignment, the median input length is a robust neutral choice.

if nargin < 2 || ~isfield(cfg,'profileLengthMode')
    mode = 'median';
else
    mode = lower(cfg.profileLengthMode);
end

switch mode
    case 'median'
        K = round(median(lengths));
    case 'mean'
        K = round(mean(lengths));
    case 'max'
        K = max(lengths);
    case 'legacy120max'
        K = ceil(1.2*max(lengths));
    otherwise
        error('Unknown cfg.profileLengthMode: %s',mode);
end
K = max(1,K);
end
