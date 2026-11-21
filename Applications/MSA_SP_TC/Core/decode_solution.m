function alignedSeq = decode_solution(sequence, a, lengthdata, L, gbestx)
%DECODE_SOLUTION Decode one optimized HMM parameter vector using Viterbi.
dimA = 3*(3*L+1);
dimB = 4*(2*L+1);
if numel(gbestx) ~= dimA + dimB
    error('gbestx length %d does not match expected dimension %d.', numel(gbestx), dimA+dimB);
end
A = reshape(gbestx(1:dimA), 3*L+1, 3);
B = reshape(gbestx(dimA+1:dimA+dimB), 2*L+1, 4);
alignedSeq = Viterbi(sequence,a,lengthdata,A,B,L);
end
