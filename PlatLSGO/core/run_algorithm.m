function [bestX,bestFitness,bestHistory] = run_algorithm(name,suite,dimension,xmax,xmin,vmax,vmin,f,funcId)
root = platform_root();
popsize = get_population(name);
maxiter = ceil(3e6/popsize);
algDir = fullfile(root,'Algorithms',name);

switch upper(name)
    case 'AHLSO'
        addpath(algDir);
        c = onCleanup(@() rmpath(algDir));
        [bestX,bestFitness,bestHistory] = zAHLSO([],popsize,dimension,xmax,xmin,vmax,vmin,maxiter,f,funcId,[]);
    case 'DCSO'
        addpath(algDir);
        c = onCleanup(@() rmpath(algDir));
        [bestX,bestFitness,bestHistory] = D_CSO([],popsize,dimension,xmax,xmin,vmax,vmin,maxiter,f,funcId,[]);
    case 'CCOS'
        [bestX,bestFitness,bestHistory] = run_ccos(suite,popsize,dimension,xmax,xmin,vmax,vmin,maxiter,f,funcId);
    case 'DECC_MDG'
        % Holds the staged decomposition file until this function returns.
        mdgCleanup = check_mdg_data(suite,funcId); %#ok<NASGU>
        addpath(algDir);
        c = onCleanup(@() rmpath(algDir));
        [bestX,bestFitness,bestHistory] = DECC_MDG([],popsize,dimension,xmax,xmin,vmax,vmin,maxiter,f,funcId,[]);
    otherwise
        addpath(algDir);
        c = onCleanup(@() rmpath(algDir));
        funName = name;
        [bestX,bestFitness,bestHistory] = feval(funName,[],popsize,dimension,xmax,xmin,vmax,vmin,maxiter,f,funcId,[]);
end
end
