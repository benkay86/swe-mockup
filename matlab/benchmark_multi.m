fprintf("Benchmark of multiple, parallel SwE computations.\n");
fprintf("Loading mock data from mock-data/*.npy...");

% Add path for loading npy files.
addpath npy-matlab/npy-matlab/

% Load mock data.
pinvDesignMtx = readNPY("../mock-data/x_pinv.npy");
residual = readNPY("../mock-data/resid.npy");
groupIds = readNPY("../mock-data/block_ids.npy");
numGroupIDs = readNPY("../mock-data/n_blocks.npy");
fprintf(" done.\n");

% Number of repetitions.
numRep = 20;

% Display parameters.
fprintf("Mock data parameters:\n");
fprintf("Number of observations: %d\n", size(residual, 1));
fprintf("Number of features: %d\n", size(residual, 2));
fprintf("Number of predictors: %d\n", size(pinvDesignMtx, 1));
fprintf("Number of blocks: %d\n", numGroupIDs);
fprintf("Number of parallel repetitions: %d\n", numRep);

% Spin up parallel pool.
if isempty(gcp("nocreate"))
    local = parcluster('local');
    local.NumWorkers = maxNumCompThreads();
    saveProfile(local)
    pool = local.parpool(maxNumCompThreads());
end
fprintf("Parallel pool has %d workers.\n", maxNumCompThreads());

% Compute numRep copies of covB in parallel.
fprintf("Computing SwE...");
tic;
parfor i=1:numRep
    covB = halfSandwichWald(pinvDesignMtx, residual, groupIds, numGroupIDs);
end
timeElapsed = toc;
fprintf(" done.\n");

% How did we do?
fprintf("Time elapsed: %f seconds.\n", timeElapsed);
fprintf("That's %f seconds per repetition.\n", timeElapsed / numRep);