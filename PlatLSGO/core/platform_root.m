function root = platform_root()
%PLATFORM_ROOT  Absolute path of the PlatLSGO platform root folder.
%   All platform helpers live in <root>/core, so the root is one level up.
%   The benchmark suites and every algorithm load their data through
%   CWD-relative paths, so the platform must be run from <root>.
root = fileparts(fileparts(mfilename('fullpath')));
end
