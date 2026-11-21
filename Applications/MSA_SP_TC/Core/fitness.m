% Internal optimization objective for the original HMM-MSA formulation.
% IMPORTANT: SPS.m here is the search fitness, not the reference-based BAliBASE SP/Q score.
% Fitness function for the HMM-based MSA formulation.

function T = fitness(seq,a,lengthdata,L,data)
    msa_fitness_counter('increment');
    A=data(:,1:3*(3*L+1));
    A=reshape(A,3*L+1,3);
	
    B=data(:,3*(3*L+1)+1:3*(3*L+1)+4*(2*L+1));
    B=reshape(B,2*L+1,4);
	
    seq1=Viterbi(seq,a,lengthdata,A,B,L);
	
    T=SPS(seq1);
end

