function scores = score_sp_tc(testFile, refFile)
%SCORE_SP_TC Reference-based BAliBASE/BALI2DNA SP (Q) and TC scores.
%
% SP = fraction of aligned residue pairs in the assessed reference region
%      that are reproduced in the test alignment.
% TC = fraction of assessed reference columns reproduced exactly.
%
% BALI2DNA reference FASTA files use upper-case letters for assessed/core
% regions and lower-case letters for regions excluded from assessment. If a
% reference contains no lower-case letters, all non-empty columns are used.

T = read_fasta_records(testFile);
R = read_fasta_records(refFile);
[testSeqs, refSeqs, ids] = reorder_test_to_reference(T,R); %#ok<ASGLU>

n = numel(refSeqs);
refLen = length(refSeqs{1});
testLen = length(testSeqs{1});
for i=1:n
    if length(refSeqs{i}) ~= refLen
        error('Reference alignment contains unequal row lengths.');
    end
    if length(testSeqs{i}) ~= testLen
        error('Test alignment contains unequal row lengths.');
    end
    refUngap = upper(remove_gaps(refSeqs{i}));
    testUngap = upper(remove_gaps(testSeqs{i}));
    if ~strcmp(refUngap,testUngap)
        error('Ungapped sequence mismatch for sequence %s.', ids{i});
    end
end

% Coordinate maps: each alignment column stores the ungapped residue index.
refPos = zeros(n,refLen);
testPos = zeros(n,testLen);
testColOfResidue = cell(n,1);
for s=1:n
    p=0;
    for c=1:refLen
        if ~is_gap(refSeqs{s}(c))
            p=p+1; refPos(s,c)=p;
        end
    end
    p=0;
    colmap=zeros(1,length(remove_gaps(testSeqs{s})));
    for c=1:testLen
        if ~is_gap(testSeqs{s}(c))
            p=p+1; testPos(s,c)=p; colmap(p)=c;
        end
    end
    testColOfResidue{s}=colmap;
end

% Determine assessed reference columns. In BALI2DNA, core/reference regions
% are upper-case. Gaps do not affect case detection.
hasAnyLower = false;
for s=1:n
    letters = refSeqs{s}(~arrayfun(@is_gap,refSeqs{s}));
    if any(letters>='a' & letters<='z'), hasAnyLower=true; break; end
end
core = false(1,refLen);
for c=1:refLen
    col = repmat(' ',n,1);
    for ss=1:n, col(ss)=refSeqs{ss}(c); end
    nongap = col(~arrayfun(@is_gap,col));
    if isempty(nongap), continue; end
    if hasAnyLower
        core(c) = any(nongap>='A' & nongap<='Z') && ~any(nongap>='a' & nongap<='z');
    else
        core(c) = true;
    end
end

% SP/Q score. Assessment is residue-pair based: when lower-case
% annotation exists, only upper-case reference residues participate.
denPairs = 0;
correctPairs = 0;
for c=1:refLen
    if hasAnyLower
        assessed=false(n,1);
        for ss=1:n
            ch=refSeqs{ss}(c);
            assessed(ss)=~is_gap(ch) && ch>='A' && ch<='Z';
        end
        present=find(assessed);
    else
        present=find(refPos(:,c)>0);
    end
    for ii=1:numel(present)-1
        s1=present(ii); r1=refPos(s1,c);
        tc1=testColOfResidue{s1}(r1);
        for jj=ii+1:numel(present)
            s2=present(jj); r2=refPos(s2,c);
            tc2=testColOfResidue{s2}(r2);
            denPairs=denPairs+1;
            if tc1==tc2, correctPairs=correctPairs+1; end
        end
    end
end
if denPairs==0
    SP=NaN;
else
    SP=correctPairs/denPairs;
end

% TC score: exact residue-coordinate signature of each assessed reference
% column must appear in the test alignment.
testKeys = containers.Map('KeyType','char','ValueType','logical');
for c=1:testLen
    key = signature(testPos(:,c));
    testKeys(key)=true;
end
coreCols=find(core);
correctCols=0;
for c=coreCols
    key=signature(refPos(:,c));
    if isKey(testKeys,key), correctCols=correctCols+1; end
end
if isempty(coreCols)
    TC=NaN;
else
    TC=correctCols/numel(coreCols);
end

scores.SP=SP;
scores.TC=TC;
scores.CorrectPairs=correctPairs;
scores.ReferencePairs=denPairs;
scores.CorrectColumns=correctCols;
scores.ReferenceColumns=numel(coreCols);
end

function [testSeqs, refSeqs, ids] = reorder_test_to_reference(T,R)
nt=numel(T); nr=numel(R);
if nt~=nr, error('Test/reference sequence count mismatch: %d vs %d.',nt,nr); end
used=false(1,nt);
testSeqs=cell(nr,1); refSeqs=cell(nr,1); ids=cell(nr,1);
for r=1:nr
    rid=first_token(R(r).Header);
    idx=[];
    for t=1:nt
        if ~used(t) && strcmpi(first_token(T(t).Header),rid)
            idx=t; break;
        end
    end
    if isempty(idx)
        target=upper(remove_gaps(R(r).Sequence));
        for t=1:nt
            if ~used(t) && strcmp(upper(remove_gaps(T(t).Sequence)),target)
                idx=t; break;
            end
        end
    end
    if isempty(idx), error('Cannot match reference sequence: %s',R(r).Header); end
    used(idx)=true;
    testSeqs{r}=T(idx).Sequence;
    refSeqs{r}=R(r).Sequence;
    ids{r}=R(r).Header;
end
end

function x=first_token(h)
x=regexp(strtrim(h),'\S+','match','once');
end
function s=remove_gaps(s)
s=s(~arrayfun(@is_gap,s));
end
function tf=is_gap(ch)
tf=(ch=='-' || ch=='_' || ch=='.');
end
function k=signature(v)
k=sprintf('%d,',v);
end
