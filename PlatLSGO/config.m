function cfg = config()
cfg.suite = 'CEC2010';
cfg.funcIds = 1:20;
cfg.dimension = 1000;
cfg.numRuns = 30;
cfg.baseSeed = 1;
cfg.algorithms = { ...
    'HSLSO','TPCSO','HCLPSO','SCLDPSO','DCSO','WGA','CSO','SLPSO', ...
    'APSO_DEE','EAPSO','CCOS','DECC_MDG','LLSO','DLLSO','RLLPSO', ...
    'AHLSO','PCLSO','DPCLSO','RCIPSO','DECC_DG2','MOS','MLSHADE_SPA'};
cfg.resultDir = fullfile(pwd,'results');
end