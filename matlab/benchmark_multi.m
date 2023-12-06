% Add path for loading npy files.
addpath npy-matlab/npy-matlab/

% Load mock data.
pinvDesignMtx = readNPY("../mock-data/x_pinv.npy");
residual = readNPY("../mock-data/resid.npy");
groupIds = readNPY("../mock-data/block_ids.npy");
numGroupIDs = readNPY("../mock-data/n_blocks.npy");

% Number of repetitions.
numRep = 20;

% Spin up parallel pool.
if isempty(gcp("nocreate"))
    local = parcluster('local');
    local.NumWorkers = maxNumCompThreads();
    saveProfile(local)
    pool = local.parpool(maxNumCompThreads());
end

% Compute numRep copies of covB in parallel.
tic;
parfor i=1:numRep
    covB = halfSandwichWald(pinvDesignMtx, residual, groupIds, numGroupIDs);
end
timeElapsed = toc;
