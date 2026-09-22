function [gbestx,bestever,gbesthistory]=CSO(sequence,a,lengthdata,L,maxiter,dimension)
popsize=50;
xmax=1;
xmin=0;
vmax=xmax;
vmin=xmin;
m = popsize;
% gbesthistory=rand(maxiter,1);

phi = 0.2;% Unclear here: check the original paper for how phi is set

FEs = 0;
MaxFEs = 10000*dimension;

% population initialization
for i =1:m
    p(i,:)=xmin+(xmax-xmin).*rand(1,dimension);
    v(i,:) = vmin+(vmax-vmin).*rand(1,dimension);
    fitnessx(i)= fitness(sequence,a,lengthdata,L,p(i,:)); % individual fitness    
end

FEs = FEs+popsize;

[bestever,id] = max(fitnessx);
gbestx = p(id,:); 

gbesthistory = [bestever*ones((FEs),1)]; 

gen = 1;

% main loop
while gen <= maxiter

    % randomly build competing pairs
    rlist = randperm(m);
    rpairs = [rlist(1:ceil(m/2)); rlist(floor(m/2) + 1:m)]';
        
    % compute the center position
    center = ones(ceil(m/2),1)*mean(p);
        
    % pairwise competition between particles
    mask = (fitnessx(rpairs(:,1))<fitnessx(rpairs(:,2)));
    for k = 1:ceil(m/2)
        if mask(k)==0
            losers(k)=rpairs(k,2);
            winners(k)=rpairs(k,1);
        else
            losers(k)=rpairs(k,1);
            winners(k)=rpairs(k,2);
        end
    end
   
        
%     %random matrix 
%     randco1 = rand(ceil(m/2), d);
%     randco2 = rand(ceil(m/2), d);
%     randco3 = rand(ceil(m/2), d);
         
    % boundary control
    for i = 1:ceil(m/2)
        r = rand(3,dimension);
        v(losers(i),:) = r(1).*v(losers(i),:)+r(2).*(p(winners(i),:)-p(losers(i),:))+ phi*r(3).*(center(i,:)-p(losers(i),:));
        p(losers(i),:) = p(losers(i),:) + v(losers(i),:);
        p(losers(i),:) = max(p(losers(i),:), xmin);
        p(losers(i),:) = min(p(losers(i),:), xmax);
        fitnessx(losers(i)) = fitness(sequence,a,lengthdata,L,p(losers(i),:));
        FEs = FEs+1;
        if  fitnessx(i)>= bestever
            bestever = fitnessx(i);
            gbestx = p(i,:);
        end

        
    end
    gbesthistory(gen) = bestever;
    fprintf("CSO  gen %d  best = %e\n",gen,bestever);
    gen = gen+1;

end
end