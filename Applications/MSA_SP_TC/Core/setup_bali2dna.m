function setup_bali2dna(cfg)
%SETUP_BALI2DNA Download/extract official benchmark references when missing.
required = cfg.datasets;
allFound = true;
for i=1:numel(required)
    if isempty(find_reference_file(cfg.referenceDir,required{i}))
        allFound = false;
        break;
    end
end
if allFound, return; end
if ~cfg.autoDownloadReference
    error(['BALI2DNA reference alignments are missing. Put the official files ' ...
           'from bench.tar.gz/bali2dna/ref into: %s'], cfg.referenceDir);
end
fprintf('\nBALI2DNA reference files are missing. Attempting official download...\n');
if ~exist(cfg.referenceDir,'dir'), mkdir(cfg.referenceDir); end
downloadDir = fullfile(cfg.rootDir,'Data','Downloads');
if ~exist(downloadDir,'dir'), mkdir(downloadDir); end
archive = fullfile(downloadDir,'bench.tar.gz');
extractDir = fullfile(downloadDir,'bench_extract');
try
    if ~exist(archive,'file')
        websave(archive,cfg.benchmarkURL);
    end
    if exist(extractDir,'dir'), rmdir(extractDir,'s'); end
    mkdir(extractDir);
    untar(archive,extractDir);
catch ME
    error(['Automatic BALI2DNA download failed: %s\n\nManual fallback:\n' ...
           '1) Download https://www.drive5.com/bench/bench.tar.gz\n' ...
           '2) Extract it.\n' ...
           '3) Copy the folder bali2dna/ref into:\n   %s\n' ...
           '4) Run the experiment again.'], ME.message, cfg.referenceDir);
end
refDirs = dir(fullfile(extractDir,'**','bali2dna','ref'));
if isempty(refDirs)
    error('Downloaded benchmark archive does not contain bali2dna/ref.');
end
sourceRef = fullfile(refDirs(1).folder,refDirs(1).name);
copyfile(fullfile(sourceRef,'*'),cfg.referenceDir);
infoDirs = dir(fullfile(extractDir,'**','bali2dna','info'));
if ~isempty(infoDirs)
    infoTarget = fullfile(cfg.rootDir,'Data','BALI2DNA','info');
    if ~exist(infoTarget,'dir'), mkdir(infoTarget); end
    copyfile(fullfile(infoDirs(1).folder,infoDirs(1).name,'*'),infoTarget);
end
for i=1:numel(required)
    if isempty(find_reference_file(cfg.referenceDir,required{i}))
        error('Reference alignment %s was not found after benchmark extraction.',required{i});
    end
end
fprintf('BALI2DNA reference alignments installed successfully.\n');
end
