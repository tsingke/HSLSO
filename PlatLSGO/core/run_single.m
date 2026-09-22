function result = run_single(cfg,algorithm,funcId,runId)
setup_paths(cfg.suite);
[xmax,xmin,vmax,vmin,dimension] = get_ranges(cfg.suite,funcId,cfg.dimension);
f = get_benchmark(cfg.suite);
global initial_flag
initial_flag = 0;
rng(cfg.baseSeed + 100000*find(strcmp(cfg.algorithms,algorithm),1) + 1000*funcId + runId,'twister');
t = tic;
[bestX,bestFitness,bestHistory] = run_algorithm(algorithm,cfg.suite,dimension,xmax,xmin,vmax,vmin,f,funcId);
runtime = toc(t);
result.algorithm = algorithm;
result.funcId = funcId;
result.runId = runId;
result.dimension = dimension;
result.bestFitness = bestFitness;
result.bestX = bestX;
result.bestHistory = bestHistory;
result.runtime = runtime;
end
