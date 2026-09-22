% HMM模型和MSA封装成的适应度函数

function T = fitness(seq,a,lengthdata,L,data)
    A=data(:,1:3*(3*L+1));
    A=reshape(A,3*L+1,3);
	
    B=data(:,3*(3*L+1)+1:3*(3*L+1)+4*(2*L+1));
    B=reshape(B,2*L+1,4);
	
    seq1=Viterbi(seq,a,lengthdata,A,B,L);
	
    T=SPS(seq1);
end

