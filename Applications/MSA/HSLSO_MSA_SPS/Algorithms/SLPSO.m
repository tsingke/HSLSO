function [bestp,bestever,gbesthistory]=SLPSO(sequence,a,lengthdata,L,maxiter,dimension)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%  Implementation of a social learning PSO (SL-PSO) for scalable optimization
%%
%%  See the details of SL-PSO in the following paper
%%  R. Cheng and Y. Jin, A Social Learning Particle Swarm Optimization Algorithm for Scalable Pptimization,
%%  Information Sicences, 2014
%%
%%  The source code SL-PSO is implemented by Ran Cheng 
%%
%%  If you have any questions about the code, please contact: 
%%  Ran Cheng at r.cheng@surrey.ac.uk 
%%  Prof. Yaochu Jin at yaochu.jin@surrey.ac.uk
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

xmax=1;
xmin=0;
%d: dimensionality
d = dimension;
%maxfe: maximal number of fitness evaluations
% maxfe = 3e6;

% lu: define the upper and lower bounds of the variables
lu = [xmin * ones(1, d); xmax * ones(1, d)];

%parameter initiliaztion
M = 100;
m = 50;% 种群规模，传入的popsize用不上
c3 = d/M*0.01;
PL = zeros(m,1);

for i = 1 : m
    PL(i) = (1 - (i - 1)/m)^log(sqrt(ceil(d/M)));
end


% initialization
XRRmin = repmat(lu(1, :), m, 1);
XRRmax = repmat(lu(2, :), m, 1);
rand('seed', sum(100 * clock));
p = XRRmin + (XRRmax - XRRmin) .* rand(m, d);
for i = 1:m
    fitnessp(i,:) = fitness(sequence,a,lengthdata,L,p(i,:)); 
end
v = zeros(m,d);
bestever = 0;

% FES = m;
gen = 1;

%% main loop
while(gen <= maxiter)

    % population sorting
    [fitnessp rank] = sort(fitnessp);
    p = p(rank,:);
    v = v(rank,:);
    besty = fitnessp(m);
    bestp = p(m, :);
    bestever = max(besty, bestever);
    
    % center position
    center = ones(m,1)*mean(p);
    
    %random matrix 
    %rand('seed', sum(100 * clock));
    randco1 = rand(m, d);
    %rand('seed', sum(100 * clock));
    randco2 = rand(m, d);
    %rand('seed', sum(100 * clock));
    randco3 = rand(m, d);
    winidxmask = repmat([1:m]', [1 d]);
    winidx = winidxmask + ceil(rand(m, d).*(m - winidxmask));
    pwin = p;
    for j = 1:d
            pwin(:,j) = p(winidx(:,j),j);
    end
    
    % social learning
     lpmask = repmat(rand(m,1) < PL, [1 d]);
     lpmask(m,:) = 0;
     v1 =  1*(randco1.*v + randco2.*(pwin - p) + c3*randco3.*(center - p));
     p1 =  p + v1;   
     
     
     v = lpmask.*v1 + (~lpmask).*v;         
     p = lpmask.*p1 + (~lpmask).*p;
     
    % boundary control
    for i = 1:m - 1
        p(i,:) = max(p(i,:), lu(1,:));
        p(i,:) = min(p(i,:), lu(2,:));
        fitnessp(i,:) = fitness(sequence,a,lengthdata,L,p(i,:));
%         FES = FES + 1;
         % Update Personal Best
        if fitnessp(i,:)>bestever
            
            bestp=p(i,:);
            bestever=fitnessp(i,:);
            
        end
        
    end
    gbesthistory(gen)=bestever;
%         if mod(FES, maxfe/10) == 0 && FES <= maxfe
            fprintf("SLPSO 第%d代，最佳适应度 = %e\n",gen,bestever);
%         end
    gen = gen + 1;
end



end

function T = fitness(seq,a,lengthdata,L,data)
    A=data(:,1:3*(3*L+1));
    A=reshape(A,3*L+1,3);
	
    B=data(:,3*(3*L+1)+1:3*(3*L+1)+4*(2*L+1));
    B=reshape(B,2*L+1,4);
	
    seq1=Viterbi(seq,a,lengthdata,A,B,L);
	
    T=SPS(seq1);
end

    

