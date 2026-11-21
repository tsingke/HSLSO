function s = profile_hmm_settings()
%PROFILE_HMM_SETTINGS Central settings for the v4 profile-HMM MSA model.
%
% No benchmark reference information is used here.  The profile prior is
% estimated only from the unaligned input sequences.

% Candidate genes x in [0,1] perturb a reference-free empirical prior as
%   w_eff = w_prior .* exp(T * (2*x-1)).
% T=0 would ignore the optimizer; larger T gives a broader search.
s.perturbationTemperature = 1.75;

% Pseudocount for nucleotide emission priors.
s.emissionPseudoCount = 0.5;

% Reference-free affine sum-of-pairs objective.
s.matchScore    = 2.0;
s.mismatchScore = -1.0;
s.gapOpen       = -2.5;
s.gapExtend     = -0.5;
end
