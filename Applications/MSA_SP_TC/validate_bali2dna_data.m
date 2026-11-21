function validate_bali2dna_data()
%VALIDATE_BALI2DNA_DATA Check that each input sequence set matches its
%corresponding BALI2DNA reference alignment after removing alignment gaps.

rootDir = fileparts(mfilename('fullpath'));
addpath(fullfile(rootDir,'Core'));
cfg = config_msa();

for d = 1:numel(cfg.datasets)
    dataset = cfg.datasets{d};
    inputFile = fullfile(cfg.inputDir,dataset);
    refFile = find_reference_file(cfg.referenceDir,dataset);

    if exist(inputFile,'file') ~= 2
        error('Missing input sequence file: %s',inputFile);
    end
    if isempty(refFile) || exist(refFile,'file') ~= 2
        error('Missing BALI2DNA reference alignment for %s.',dataset);
    end

    I = read_fasta_records(inputFile);
    R = read_fasta_records(refFile);
    if numel(I) ~= numel(R)
        error('%s: input/reference sequence counts differ (%d vs %d).', ...
              dataset,numel(I),numel(R));
    end

    inputIDs = cell(numel(I),1);
    for k = 1:numel(I), inputIDs{k} = first_token(I(k).Header); end

    for k = 1:numel(R)
        rid = first_token(R(k).Header);
        idx = find(strcmpi(inputIDs,rid),1);
        if isempty(idx)
            error('%s: reference sequence ID not found in input: %s',dataset,rid);
        end
        refUngapped = upper(remove_gaps(R(k).Sequence));
        inputUngapped = upper(remove_gaps(I(idx).Sequence));
        if ~strcmp(refUngapped,inputUngapped)
            error('%s: ungapped input/reference sequence mismatch for %s.',dataset,rid);
        end
    end

    fprintf('Validated %-12s: %d sequences; reference=%s\n', ...
            dataset,numel(I),refFile);
end
fprintf('All six input/reference dataset pairs passed validation.\n');
end

function t = first_token(h)
q = regexp(strtrim(h),'\S+','match','once');
t = q;
end

function s = remove_gaps(s)
s = s(~(s=='-' | s=='_' | s=='.'));
end
