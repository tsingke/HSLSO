function root = setup_paths(suite)
%SETUP_PATHS  Put the platform folders required by a suite on the MATLAB path.
root = platform_root();
addpath(root);
addpath(fullfile(root,'core'));
if strcmpi(suite,'CEC2010')
    addpath(fullfile(root,'benchmarks','CEC2010'));
elseif strcmpi(suite,'CEC2013')
    addpath(fullfile(root,'benchmarks','CEC2013'));
else
    error('Unknown suite.');
end
end
