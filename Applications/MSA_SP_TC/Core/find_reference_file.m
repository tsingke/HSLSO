function refFile = find_reference_file(referenceDir, datasetName)
%FIND_REFERENCE_FILE Locate a BALI2DNA reference alignment by dataset id.
refFile = '';
if ~exist(referenceDir,'dir'), return; end
candidates = {datasetName, [datasetName '.fa'], [datasetName '.fasta'], ...
              [datasetName '.fas'], [datasetName '.tfa'], [datasetName '.aln']};
for i=1:numel(candidates)
    p = fullfile(referenceDir,candidates{i});
    if exist(p,'file')
        refFile = p;
        return;
    end
end
% recursive fallback
listing = dir(fullfile(referenceDir,'**','*'));
for i=1:numel(listing)
    if listing(i).isdir, continue; end
    [~,base,~] = fileparts(listing(i).name);
    if strcmpi(listing(i).name,datasetName) || strcmpi(base,datasetName)
        refFile = fullfile(listing(i).folder,listing(i).name);
        return;
    end
end
end
