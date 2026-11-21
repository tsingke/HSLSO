function score = SPS(aln)
%SPS Reference-free affine sum-of-pairs objective for nucleotide MSA.
%
% This function is an INTERNAL optimization objective.  It is not the
% BAliBASE reference SP/Q score.  No reference alignment is read here.
%
% For each induced pairwise alignment (gap-gap columns removed):
%   match       : +2.0
%   mismatch    : -1.0
%   first gap in a run : -2.5
%   additional gap residues in the same run : -0.5 each
%
% Scores are configured in profile_hmm_settings.m.

if isempty(aln)
    score = -inf;
    return;
end

s = profile_hmm_settings();
[m,~] = size(aln);
score = 0;

for i = 1:m-1
    for j = i+1:m
        a = upper(aln(i,:));
        b = upper(aln(j,:));
        ga = is_gap_vec(a);
        gb = is_gap_vec(b);

        % Remove gap-gap columns before forming the induced pairwise
        % alignment.  Such columns are not biological residue pairs and
        % should neither score nor interrupt a gap run.
        keep = ~(ga & gb);
        a = a(keep); b = b(keep);
        ga = ga(keep); gb = gb(keep);

        if isempty(a), continue; end

        bothResidue = ~ga & ~gb;
        if any(bothResidue)
            eq = a(bothResidue) == b(bothResidue);
            score = score + s.matchScore*sum(eq) + ...
                s.mismatchScore*sum(~eq);
        end

        gapA = ga & ~gb;
        gapB = gb & ~ga;
        score = score + affine_gap_score(gapA,s.gapOpen,s.gapExtend);
        score = score + affine_gap_score(gapB,s.gapOpen,s.gapExtend);
    end
end
end

function val = affine_gap_score(mask,gopen,gext)
if ~any(mask)
    val = 0;
    return;
end
starts = mask & [true ~mask(1:end-1)];
nOpen = sum(starts);
nGap = sum(mask);
val = gopen*nOpen + gext*(nGap-nOpen);
end

function tf = is_gap_vec(x)
tf = (x=='_' | x=='-' | x=='.');
end
