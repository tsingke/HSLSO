% DE algorithm

function [gbestx,gbestfitness,gbesthistory]=DE(sequence,a,lengthdata,L,maxiter,dimension)
% Differential Evolution Algorithm: DE/rand/1
% popsize: population size
% dimension:  dimension of an individual
% maxiter:  number of iterations
% Funcid:  test function index

%% position space allocation
popsize=50;
xmax=1;
xmin=0;

x=rand(popsize,dimension); %position vector
v=rand(popsize,dimension); %mutation vector
u=rand(popsize,dimension); %trial vector
fitnessx= rand(1,popsize);
gbestx = rand(1,dimension); % stores the global best position

gbesthistory=rand(maxiter,1); %record the best fitness value of each generation, for plotting the convergence curve

F = 0.5;
CR= 0.9;

%% population initialization
for i =1:popsize
    x(i,:)=xmin+(xmax-xmin).*rand(1,dimension);
    fitnessx(i)=fitness(sequence,a,lengthdata,L,x(i,:)); % individual fitness
    
end

%% initialize the global best fitness and the corresponding position
[gbestfitness, id] = min(fitnessx);
gbestx = x(id,:);

%% iteration loop
iter =1;
while iter<=maxiter
    %******************[Part 1: population evolution] ****************** 
    for i =1:popsize
        
        %% 1. individual i performs mutation to generate the mutant vector v (does the position go out of bounds after this?)
        
         %randomly select 2 integers different from i
          r=selectID(popsize,i,2);
          
          r1=r(1);
          r2=r(2);
          
        v(i,:)= gbestx(1,:) +  F*(x(r1,:) - x(r2,:));
        
        %% 2. individual i performs crossover to generate the trial vector u
        temp =randi([1,popsize],1);
        
        for j =1:dimension
            if(rand <= CR || j==temp)
                u(i,j)=v(i,j);
            else
                u(i,j)=x(i,j);
            end
        end
        
        
        %% 3. individual i performs greedy selection, generating the next-generation population of i (involves fitness evaluation)
        ufitness = fitness(sequence,a,lengthdata,L,u(i,:));
        if ufitness >= fitnessx(i) 
            fitnessx(i)= ufitness; % update fitness
            x(i,:) = u(i,:);      % update individual position
           
        end
        
        %% 4 update gbest in time; individuals with larger indices benefit from this update
        if  fitnessx(i)>= gbestfitness
            gbestfitness = fitnessx(i);
            gbestx =x(i,:);
        end
        
    end % end each individual
    
     %******************[Part 2: record the best] ****************** 
      gbesthistory(iter)=gbestfitness;
 
    fprintf("DE/best/1  gen %d  best = %e\n",iter,gbestfitness);
    
    iter = iter+1;
 
end % end iter

end % end function


function [r]=selectID(popsize,i,count)
% function purpose: randomly generate count mutually distinct integers in [1,popsize] that exclude i

if count<= popsize
    
    %1. remove the value i and build a new vector vec
    vec=[1:i-1,i+1:popsize]; 

    %2. randomly generate count distinct values
    r=zeros(1,count);
    for j =1:count
        n = popsize-j;   %current number of entries in vec
        t = randi(n,1,1);%generate a random integer
        r(j) = vec(t);   %take a random value
        vec(t)=[]; %delete the current element from the array so that it cannot be chosen again
    end
    
end

end
